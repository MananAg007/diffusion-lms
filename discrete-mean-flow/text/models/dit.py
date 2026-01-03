import math
import typing

import einops
import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

env_val = os.getenv('DIT_USE_COMPILE', '0').lower()
USE_COMPILE = env_val in ['1', 'true', 'yes', 'on']
print(f"DIT: USE_COMPILE={USE_COMPILE}")

if USE_COMPILE:
    torch_compile_deco = torch.compile(model=None, mode=None, dynamic=False, options={"max_autotune": True, "triton.cudagraphs": False})
    jit_deco = lambda x: x
else:
    torch_compile_deco = lambda x: x
    jit_deco = torch.jit.script
    
# Import JVP-enabled attention if available
try:
    from utils.jvp import attention as jvp_attention
    JVP_ATTENTION_AVAILABLE = True
except ImportError:
    JVP_ATTENTION_AVAILABLE = False
    jvp_attention = None

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


def bias_dropout_add_scale(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float,
        training: bool) -> torch.Tensor:
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)

    if residual is not None:
        out = residual + out
    return out


def get_bias_dropout_add_scale(training):
    def _bias_dropout_add(x, bias, scale, residual, prob):
        return bias_dropout_add_scale(
            x, bias, scale, residual, prob, training)

    return _bias_dropout_add


# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


@jit_deco
def bias_dropout_add_scale_fused_train(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, True)


@jit_deco
def bias_dropout_add_scale_fused_inference(
        x: torch.Tensor,
        bias: typing.Optional[torch.Tensor],
        scale: torch.Tensor,
        residual: typing.Optional[torch.Tensor],
        prob: float) -> torch.Tensor:
    return bias_dropout_add_scale(
        x, bias, scale, residual, prob, False)


@jit_deco
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim],
                             device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos(
            )[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin(
            )[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # This makes the transformation on v an identity.
            self.cos_cached[:, :, 2, :, :].fill_(1.)
            self.sin_cached[:, :, 2, :, :].fill_(0.)

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def split_and_apply_rotary_pos_emb(qkv, rotary_cos_sin,):
    with torch.amp.autocast(device_type=qkv.device.type, enabled=False):
        cos, sin = rotary_cos_sin
        cos = cos.to(qkv.dtype)
        sin = sin.to(qkv.dtype)
        cos = cos[0, :, 0, 0, :cos.shape[-1]//2]
        sin = sin[0, :, 0, 0, :sin.shape[-1]//2]
        q, k, v = qkv.chunk(3, dim=2)
        q = flash_attn.layers.rotary.apply_rotary_emb_torch(
            q.squeeze(dim=2), cos, sin)
        k = flash_attn.layers.rotary.apply_rotary_emb_torch(
            k.squeeze(dim=2), cos, sin)
        v = v.squeeze(dim=2)
    return q, k, v


def apply_rotary_pos_emb(qkv, cos, sin):
    cos = cos[0, :, 0, 0, :cos.shape[-1]//2]
    sin = sin[0, :, 0, 0, :sin.shape[-1]//2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


def regular_attention_multi_headed(q, k, v, tq=None, tk=None, tv=None, use_jvp=False):
    """
    Multi-headed attention with optional JVP support.

    Args:
        q, k, v: (batch, seq_len, num_heads, head_dim)
        tq, tk, tv: Optional tangent vectors for JVP computation
        use_jvp: Whether to use JVP-enabled attention (requires tq, tk, tv)

    Returns:
        output: (batch, seq_len, num_heads * head_dim)
        If use_jvp and tangent vectors provided, also returns tangent output
    """
    # If JVP is requested and available, use JVP-enabled attention
    if use_jvp and JVP_ATTENTION_AVAILABLE and tq is not None and tk is not None and tv is not None:
        # jvp_attention expects (batch, nheads, seqlen, headdim)
        # Current shape: (batch, seqlen, nheads, headdim)
        q_t = q.transpose(1, 2)  # (B, H, L, D)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        tq_t = tq.transpose(1, 2)
        tk_t = tk.transpose(1, 2)
        tv_t = tv.transpose(1, 2)

        # jvp_attention returns (output, tangent_output)
        output, tangent_output = jvp_attention(
            q_t, k_t, v_t, tq_t, tk_t, tv_t,
            causal=False, sm_scale=None
        )

        # Convert back to (batch, seqlen, nheads, headdim)
        output = output.transpose(1, 2)
        tangent_output = tangent_output.transpose(1, 2)

        # Reshape to (batch, seqlen, nheads * headdim)
        output = einops.rearrange(output, 'b s h d -> b s (h d)')
        tangent_output = einops.rearrange(tangent_output, 'b s h d -> b s (h d)')
        return output, tangent_output

    # Standard attention (no JVP)
    attention_output = F.scaled_dot_product_attention(
        query=q.transpose(1, 2),
        key=k.transpose(1, 2),
        value=v.transpose(1, 2),
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False)
    # [batch_size, seq_len, num_heads, head_dim]
    attention_output = attention_output.transpose(1, 2)
    return einops.rearrange(attention_output, 'b s h d -> b s (h d)')


#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale).view(*x.shape[:-1], dim_out)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            - math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding,
                 torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations.

    Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core Model                                    #
#################################################################################

class DDiTBlockCausal(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, rotary_cos_sin, **kwargs):
        del kwargs
        batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # attention operation
        x_skip = x
        x = self.norm1(x)

        qkv = self.attn_qkv(x)
        qkv = einops.rearrange(
            qkv,
            'b s (three h d) -> b s three h d',
            three=3,
            h=self.n_heads)
        with torch.amp.autocast(device_type=qkv.device.type, enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        qkv = einops.rearrange(qkv, 'b s ... -> (b s) ...')
        cu_seqlens = torch.arange(
            0, (batch_size + 1) * seq_len,
            step=seq_len, dtype=torch.int32, device=qkv.device)
        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0.0, causal=True)

        x = einops.rearrange(x, '(b s) h d -> b s (h d)',
                             b=batch_size)

        scale = torch.ones(1, device=x.device, dtype=x.dtype)
        x = bias_dropout_scale_fn(
            self.attn_out(x), None, scale, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(
            self.mlp(self.norm2(x)), None, scale, x, self.dropout)
        return x


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, adaLN,
                 cond_dim=None, mlp_ratio=4,
                 dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.adaLN = adaLN
        self.softcap=50
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference
    
    
    def forward(self, x, rotary_cos_sin=None, c=None, seqlens=None, exclude_last_token=False):
        # def forward(self, x, c=None, seqlens=None, exclude_last_token=False):

        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        x_skip = x
        x = self.norm1(x)

        if self.adaLN:
            # self.adaLN_modulation(c): (128, 1536)
            # self.adaLN_modulation(c)[:, None]: (128, 1, 1536)
            # "" .chunk(6, dim=2) returns 6 tuples of shapes (128, 1, 256)
            (shift_msa, scale_msa, gate_msa, shift_mlp,
             scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
            x = modulate_fused(x, shift_msa, scale_msa)

        # qkv = einops.rearrange(
        #     self.attn_qkv(x),
        #     'b s (three h d) -> b s three h d',
        #     three=3,
        #     h=self.n_heads)
        # if rotary_cos_sin is not None:
        #     if exclude_last_token:
        #         q, k, v = split_and_apply_rotary_pos_emb(
        #             qkv[:, :-1], rotary_cos_sin)  # exclude_last_token
        #         _q, _k, _v = qkv[:, -1].chunk(3, dim=1)
        #         # import ipdb; ipdb.set_trace()
        #         q = torch.cat((q, _q), dim=1)
        #         k = torch.cat((k, _k), dim=1)
        #         v = torch.cat((v, _v), dim=1)
        #         assert q.shape[1] == k.shape[1] == v.shape[1] == qkv.shape[1]
        #     else:
        #         q, k, v = split_and_apply_rotary_pos_emb(
        #             qkv, rotary_cos_sin)  # exclude_last_token
        # else:
        #     if exclude_last_token:
        #         q, k, v = qkv[:, :-1].unbind(dim=2)  # shapes (B, L-1, H, Dh)
        #         _q, _k, _v = qkv[:, -1].unbind(dim=1)  # shapes (B, H, Dh)
        #         q = torch.cat([q, _q.unsqueeze(1)], dim=1)
        #         k = torch.cat([k, _k.unsqueeze(1)], dim=1)
        #         v = torch.cat([v, _v.unsqueeze(1)], dim=1)
        #     else:
        #         q, k, v = qkv.unbind(dim=2)
        # x = regular_attention_multi_headed(q, k, v)

        qkv = self.attn_qkv(x)
        qkv = einops.rearrange(
            qkv,
            'b s (three h d) -> b s three h d',
            three=3,
            h=self.n_heads)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        x = flash_attn.flash_attn_qkvpacked_func(
            qkv, 0.0, causal=False,
            softcap=self.softcap,
            )

        x = einops.rearrange(x, 'b s h d -> b s (h d)',)
        

        if self.adaLN:
            x = bias_dropout_scale_fn(self.attn_out(x),
                                      None,
                                      gate_msa,
                                      x_skip,
                                      self.dropout)
            x = bias_dropout_scale_fn(
                self.mlp(modulate_fused(
                    self.norm2(x), shift_mlp, scale_mlp)),
                None, gate_mlp, x, self.dropout)
        else:
            scale = torch.ones(1, device=x.device, dtype=x.dtype)
            x = bias_dropout_scale_fn(
                self.attn_out(x), None, scale, x_skip, self.dropout)
            x = bias_dropout_scale_fn(
                self.mlp(self.norm2(x)), None, scale, x, self.dropout)
        return x


class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        if x.ndim == 2:
            return self.embedding[x]
        assert x.ndim == 3
        return torch.einsum(
            "blv,ve->ble",
            torch.nn.functional.softmax(x, dim=-1).float(),
            self.embedding.float()).to(x.dtype)


def build_sinusoidal_embeddings(n_pos: int, dim: int, base: int = 10000, device=None, dtype=torch.float32):
    position = torch.arange(n_pos, device=device, dtype=dtype).unsqueeze(
        1)              # [n_pos, 1]
    base = max(base, n_pos * 10)
    div_term = torch.exp(torch.arange(
        0, dim, 2, device=device, dtype=dtype) * (-math.log(base) / dim))
    # [n_pos, dim]
    pe = torch.zeros(n_pos, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class DDiTFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim,
                 adaLN):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        # self.linear.weight.data.normal_(mean=0.0, std=0.02)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()
        self.adaLN = adaLN
        if self.adaLN:
            self.adaLN_modulation = nn.Linear(cond_dim,
                                              2 * hidden_size,
                                              bias=True)
            self.adaLN_modulation.weight.data.zero_()
            self.adaLN_modulation.bias.data.zero_()

    def forward(self, x, c):
        x = self.norm_final(x)
        if self.adaLN:
            shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
            x = modulate_fused(x, shift, scale)
        x = self.linear(x)
        return x


class DIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        if type(config) == dict:
            config = omegaconf.OmegaConf.create(config)
        self.causal = config.algo.causal_attention
        self.adaLN = not self.causal
        self.config = config
        self.vocab_size = vocab_size
        self.use_curriculum = self.config.algo.use_curriculum
        dim = config.model.hidden_size
        cond_dim = config.model.cond_dim
        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        if not self.causal:
            self.sigma_map = TimestepEmbedder(cond_dim)
        if self.config.algo.name == 'dos':
            if self.config.algo.double_temb:
                self.sigma_map_prime = TimestepEmbedder(cond_dim)
            else:
                self.sigma_map_prime = None
        self.rotary_emb = Rotary(dim // config.model.n_heads)
        
        # Get number of blocks to separate (default 0 for backward compatibility)
        # -1 means no separation at all (single output layer only)
        self.n_separated_blocks = getattr(config.algo, 'n_separated_blocks', 0)
        self.shortcut_mix_logit = getattr(config.algo, 'shortcut_mix_logit', False)
        self.shortcut_mix_type = getattr(config.algo, 'shortcut_mix_type', 'interpolate')
        if self.n_separated_blocks == 0:
            print("Model Initialized with separated output layer only")
        else:
            print(f"Model Initialized with {self.n_separated_blocks} separated blocks")
        
        # Calculate number of common blocks
        # If n_separated_blocks == -1, all blocks are common (no separation)
        if self.n_separated_blocks == -1:
            n_common_blocks = config.model.n_blocks
        else:
            n_common_blocks = config.model.n_blocks - self.n_separated_blocks
        
        # Common blocks (shared between both heads)
        common_blocks = []
        for _ in range(n_common_blocks):
            if self.causal:
                block = DDiTBlockCausal(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    dropout=config.model.dropout)
            else:
                block = DDiTBlock(
                    dim=dim,
                    n_heads=config.model.n_heads,
                    cond_dim=cond_dim,
                    adaLN=self.adaLN,
                    dropout=config.model.dropout)
            common_blocks.append(block)
        self.blocks = nn.ModuleList(common_blocks)
        
        # Separated blocks for each head (only if n_separated_blocks > 0)
        if self.n_separated_blocks > 0:
            blocks_head1 = []
            blocks_head2 = []
            for _ in range(self.n_separated_blocks):
                if self.causal:
                    block1 = DDiTBlockCausal(
                        dim=dim,
                        n_heads=config.model.n_heads,
                        dropout=config.model.dropout)
                    block2 = DDiTBlockCausal(
                        dim=dim,
                        n_heads=config.model.n_heads,
                        dropout=config.model.dropout)
                else:
                    block1 = DDiTBlock(
                        dim=dim,
                        n_heads=config.model.n_heads,
                        cond_dim=cond_dim,
                        adaLN=self.adaLN,
                        dropout=config.model.dropout)
                    block2 = DDiTBlock(
                        dim=dim,
                        n_heads=config.model.n_heads,
                        cond_dim=cond_dim,
                        adaLN=self.adaLN,
                        dropout=config.model.dropout)
                blocks_head1.append(block1)
                blocks_head2.append(block2)
            self.blocks_head1 = nn.ModuleList(blocks_head1)
            self.blocks_head2 = nn.ModuleList(blocks_head2)
        else:
            self.blocks_head1 = None
            self.blocks_head2 = None

        self.abs_pos_max_len = 1024
        abs_pe = build_sinusoidal_embeddings(
            n_pos=self.abs_pos_max_len, dim=dim, base=10000,
            device=None, dtype=torch.float32
        )
        self.register_buffer("abs_pos", abs_pe, persistent=False)

        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=self.adaLN)
        
        # Only create output_layer_sc if n_separated_blocks != -1
        # -1 means no separation at all (single output layer only)
        if self.n_separated_blocks != -1:
            self.output_layer_sc = DDiTFinalLayer(
                hidden_size=dim,
                out_channels=vocab_size,
                cond_dim=cond_dim,
                adaLN=self.adaLN)
        else:
            self.output_layer_sc = None
        
        self.sigma = 1e-5
        self.scale_by_sigma = config.model.scale_by_sigma
        if "is_di4c" in config:
            self.is_di4c = config.is_di4c
        else:
            self.is_di4c = config.is_di4c = False

        if "is_di4c_deterministic" in config:
            self.is_di4c_deterministic = config.is_di4c_deterministic
        else:
            self.is_di4c_deterministic = config.is_di4c_deterministic = False

        if self.is_di4c:
            print("Using Di4C")
            # Added for Di4C:
            self.latent_feature_dim = 128
            self.latent_projection = nn.Sequential(
                nn.Linear(in_features=self.latent_feature_dim,
                          out_features=self.latent_feature_dim*4),
                nn.GELU(),
                nn.Linear(self.latent_feature_dim*4, config.model.hidden_size)
            )

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference
    
    @torch_compile_deco
    def forward(self, x, sigma, sigma_prime=None, use_auxiliary_head=False, c_d=None, c_t=None):
        if self.use_curriculum == True:
            x = self.vocab_embed(x)
        else:
            vocab_embed_weight = self.vocab_embed.embedding
            x = torch.matmul(x, vocab_embed_weight)  # skip softmax
            
        if self.causal:
            t_cond = None
        else:
            t_emb = self.sigma_map(sigma)
            if sigma_prime is not None:
                if self.sigma_map_prime is not None:
                    t_prime_emb = self.sigma_map_prime(sigma_prime)
                else:
                    t_prime_emb = self.sigma_map(sigma_prime)
                t_emb = t_emb + t_prime_emb
                
            t_cond = F.silu(t_emb)

        rotary_cos_sin = self.rotary_emb(x)
        
        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c=t_cond,
                                    seqlens=None, exclude_last_token=self.is_di4c)
            
            if self.n_separated_blocks == -1:
                return self.output_layer(x, c=t_cond)

            if sigma_prime is not None:
                d = torch.clamp(sigma_prime.to(x.dtype), 0.0, 1.0)
                d = d.view(-1, 1, 1) if d.ndim <= 1 else d
                
            if not use_auxiliary_head: # used for flow matching loss
                if self.n_separated_blocks > 0:
                    for i in range(len(self.blocks_head1)):
                        x = self.blocks_head1[i](x, rotary_cos_sin, c=t_cond,
                                                 seqlens=None, exclude_last_token=self.is_di4c)
                return self.output_layer(x, c=t_cond)

            else:
                x1 = x
                if self.n_separated_blocks > 0:
                    for i in range(len(self.blocks_head1)):
                        x1 = self.blocks_head1[i](x1, rotary_cos_sin, c=t_cond,
                                                    seqlens=None, exclude_last_token=self.is_di4c)
                output1 = self.output_layer(x1, c=t_cond)

                x2 = x
                if self.n_separated_blocks > 0:
                    for i in range(len(self.blocks_head2)):
                        x2 = self.blocks_head2[i](x2, rotary_cos_sin, c=t_cond,
                                                    seqlens=None, exclude_last_token=self.is_di4c)
                output2 = self.output_layer_sc(x2, c=t_cond)

                
                if self.shortcut_mix_logit and c_d is not None and c_t is not None:
                    if self.shortcut_mix_type == 'interpolate':
                        output = (1.0 - c_d) * output1 + c_d * output2
                    elif self.shortcut_mix_type == 'residual':
                        weight = 0.5 * c_d * (1.0 - c_t)
                        output = output1 + weight * output2
                    else:
                        output = output1
                    return output

                return output1, output2


        return x

# From https://github.com/yang-song/score_sde_pytorch/ which is from
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py


def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32,
                    device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
