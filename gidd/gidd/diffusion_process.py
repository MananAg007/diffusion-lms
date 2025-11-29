from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gidd.utils import sample_categorical


def sample_t(config, batch_size, eps=None, device=None):
    if eps is None:
        eps = config.model.t_eps

    if config.training.low_discrepancy_sampling:
        t = torch.arange(batch_size, device=device) / batch_size
        t = (t + torch.rand(1, device=device)).fmod(1.0)
    else:
        t = torch.rand(batch_size, device=device)

    t = (1 - 2 * eps) * t + eps
    return t


class NoiseSchedule(nn.Module, ABC):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.mask_id = tokenizer.mask_token_id
        self.vocab_size = len(tokenizer)

        self.register_buffer("log_prior", self.get_log_prior())

    def get_log_prior(self):
        pr = torch.full((self.vocab_size,), -1e3)
        pr[self.mask_id] = 0
        return pr - pr.logsumexp(-1, keepdim=True)
    
    def sample_prior(self, shape):
        return torch.full(shape, self.mask_id, dtype=torch.long, device=self.log_prior.device)
    
    @abstractmethod
    def logits_at_t(self, features, t):
        raise NotImplementedError
    
    @abstractmethod
    def probs_at_t(self, prs, t):
        raise NotImplementedError

    @abstractmethod
    def sample_zt(self, input_ids, t):
        raise NotImplementedError


class HybridDiffusion(NoiseSchedule):
    def __init__(self, tokenizer, clip_noise=20, gamma=1.0, p_uniform=0.0):
        super().__init__(tokenizer)
        self.clip_noise = clip_noise
        self.p_uniform = max(np.exp(-clip_noise), p_uniform)

        log_B = gamma*np.log(2) + np.log(self.p_uniform) - np.log(1 - self.p_uniform)
        self.register_buffer("log_B", torch.tensor(float(log_B)).clip(-clip_noise))
        self.register_buffer("log_gamma", torch.tensor(float(gamma)).log())

        mask = torch.zeros(self.vocab_size)
        mask[self.mask_id] = 1
        self.register_buffer("mask", mask, persistent=False)

        unif = (1 - self.mask) / (self.vocab_size - 1)
        self.register_buffer("unif", unif, persistent=False)
    
    def get_alpha_betapi(self, t, eps=1e-4):
        t = t[:, None]
        t1m = 1 - t

        gamma = self.log_gamma.exp()
        B = self.log_B.exp()
        # .pow() autocasts to fp32
        c_t = t.pow(gamma/2) * t1m.pow(gamma/2) * B
        C_t = 1 + c_t
        # C_t should never be much smaller than 1,
        # but just in case it is, we clip it to avoid numerical instability
        C_t = C_t.clip(eps)

        alpha_t = t1m / C_t
        beta_pi = (t * self.mask + c_t * self.unif) / C_t
        return alpha_t, beta_pi

    def logits_at_t(self, features, t):
        raise NotImplementedError("logits_at_t is not implemented for HybridDiffusion. Use probs_at_t instead.")
    
    def probs_at_t(self, prs, t, eps=1e-4):
        orig_dtype = prs.dtype
        alpha_t, beta_pi = self.get_alpha_betapi(t, eps=eps)

        probs = prs.mul(alpha_t.unsqueeze(-1))
        probs[..., :beta_pi.shape[-1]].add_(beta_pi.unsqueeze(1))
        return probs.to(orig_dtype)

    def sample_zt(self, input_ids, t):
        x = F.one_hot(input_ids, num_classes=self.vocab_size).to(dtype=t.dtype)
        probs = self.probs_at_t(x, t)
        z_t = sample_categorical(probs)
        return z_t
    
class MuGIDD(NoiseSchedule):
    """
    Constant-μ interpolation between Masked and Uniform diffusion.

    q_t(z | x) = Cat( alpha(t) * one_hot(x) + beta(t) * pi_mu )

    alpha(t) = 1 - t
    beta(t)  = t

    pi_mu = (1 - mu) * mask_onehot + mu * unif_nonmask
      - mu=0   => pure masked (prior all [MASK])
      - mu=1   => pure uniform over non-mask tokens (mask prob = 0)
      - mu=0.5 => constant 50/50 mixture for all t
    """
    def __init__(self, tokenizer, mu=0.0):
        # Must set mu BEFORE super().__init__ because super registers log_prior via get_log_prior()
        self.mu = float(mu)
        if not (0.0 <= self.mu <= 1.0):
            raise ValueError(f"mu must be in [0,1], got {self.mu}")

        super().__init__(tokenizer)

        mask = torch.zeros(self.vocab_size)
        mask[self.mask_id] = 1.0
        self.register_buffer("mask", mask, persistent=False)

        unif = (1.0 - self.mask) / (self.vocab_size - 1)
        self.register_buffer("unif", unif, persistent=False)

    def get_log_prior(self):
        # Prior at t=1 is pi_mu (constant in time)
        # mask prob = (1-mu), others = mu/(V-1)
        pr = torch.full((self.vocab_size,), -1e3)  # finite "minus inf"
        if self.mu < 1.0:
            pr[self.mask_id] = float(np.log(max(1.0 - self.mu, 1e-12)))
        if self.mu > 0.0:
            val = float(np.log(max(self.mu / (self.vocab_size - 1), 1e-12)))
            pr[:] = pr.where(torch.arange(self.vocab_size) == self.mask_id, torch.tensor(val))
            pr[self.mask_id] = float(np.log(max(1.0 - self.mu, 1e-12))) if self.mu < 1.0 else -1e3

        # Normalize (so downstream code can rely on logsumexp=0)
        return pr - pr.logsumexp(-1, keepdim=True)

    def sample_prior(self, shape):
        # Sample from pi_mu. For mu=0, this returns all masks (fast path).
        if self.mu == 0.0:
            return torch.full(shape, self.mask_id, dtype=torch.long, device=self.log_prior.device)

        probs = self.log_prior.exp()
        # sample_categorical expects probs broadcastable to [..., V]
        probs = probs.view(*([1] * len(shape)), -1).expand(*shape, -1)
        return sample_categorical(probs)

    def get_alpha_betapi(self, t):
        """
        Returns:
          alpha_t: [B, 1]
          beta_pi: [B, V]
        """
        t = t[:, None]
        alpha_t = 1.0 - t
        pi_mu = (1.0 - self.mu) * self.mask + self.mu * self.unif  # [V]
        beta_pi = t * pi_mu[None, :]  # [B, V]
        return alpha_t, beta_pi

    def logits_at_t(self, features, t):
        # Mirror HybridDiffusion behavior: this schedule is used via probs_at_t.
        raise NotImplementedError("logits_at_t is not implemented for MuGIDD. Use probs_at_t instead.")

    def probs_at_t(self, prs, t):
        """
        prs is typically [B, L, V] (e.g., one-hot of x0) or any distribution over vocab.
        """
        orig_dtype = prs.dtype
        alpha_t, beta_pi = self.get_alpha_betapi(t)

        probs = prs.mul(alpha_t.unsqueeze(-1))          # [B, L, V]
        probs[..., :beta_pi.shape[-1]].add_(beta_pi.unsqueeze(1))  # add to all positions
        return probs.to(orig_dtype)

    def sample_zt(self, input_ids, t):
        x = F.one_hot(input_ids, num_classes=self.vocab_size).to(dtype=t.dtype)
        probs = self.probs_at_t(x, t)
        return sample_categorical(probs)

class MaskedDiffusion(NoiseSchedule):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        # required to be able to interchangeably mix our/mdlm schedule/loss
        self.register_buffer("log_gamma", torch.tensor(0.0))
        self.register_buffer("log_B", torch.tensor(-20.0))

    def get_sigmas(self, t, eps=1e-4):
        dsigma = (1 - eps) / (1 - (1 - eps) * t.clip(eps, 1))
        sigma = -torch.log1p(-(1 - eps) * t.clip(eps, 1))
        return dsigma, sigma

    def logits_at_t(self, features, t):
        _, sigma = self.get_sigmas(t)
        move_chance = 1 - torch.exp(-sigma)
        log_1m_move_chance = -sigma
        logits = (features + 1e-8).clip(1e-8).log().log_softmax(-1) + log_1m_move_chance[..., None, None]
        logits[:, :, self.mask_id] = move_chance.log().clip(-1e6)[..., None]
        return logits
    
    def probs_at_t(self, prs, t):
        _, sigma = self.get_sigmas(t)
        alpha_t = torch.exp(-sigma)
        probs = alpha_t[..., None, None] * prs
        probs[..., self.mask_id] = 1 - alpha_t.unsqueeze(-1)
        return probs

    def sample_zt(self, input_ids, t):
        _, sigma = self.get_sigmas(t)
        move_chance = 1 - torch.exp(-sigma)
        is_masked = torch.rand_like(input_ids.float()) < move_chance.unsqueeze(-1)
        z_t = torch.where(is_masked, self.mask_id, input_ids)
        return z_t


def get_noise_schedule(config, tokenizer):
    if config.model.type == "autoregressive":
        return None
    elif config.model.diffusion_process == "gidd":
        noise_schedule = HybridDiffusion(tokenizer, p_uniform=config.model.p_uniform)
    elif config.model.diffusion_process == "mugidd":
        noise_schedule = MuGIDD(tokenizer, mu=config.model.mu)
    elif config.model.diffusion_process == "mdlm":
        noise_schedule = MaskedDiffusion(tokenizer)
    else:
        raise ValueError(f"Unknown diffusion process: {config.model.diffusion_process}")

    return noise_schedule
