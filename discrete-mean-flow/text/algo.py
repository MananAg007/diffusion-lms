import os
import collections
import copy
import pickle

import fsspec
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
import trainer_base
import utils
import math
from models.dit import modulate_fused


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


def stopgrad(x):
    """Stop gradient for x."""
    return x.detach()


class AR(trainer_base.TrainerBase):
    def __init__(self, config, tokenizer):
        vocab_size = tokenizer.vocab_size
        if (not hasattr(tokenizer, 'mask_token')
                or tokenizer.mask_token is None):
            self.mask_index = vocab_size
            vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id
        super().__init__(config, tokenizer,
                         vocab_size=vocab_size)
        self.save_hyperparameters()
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert not self.config.algo.time_conditioning
        assert self.config.prior.type == 'none'

    def _process_model_input(self, x0, valid_tokens):
        input_tokens = x0[:, :-1]
        output_tokens = x0[:, 1:]
        valid_tokens = valid_tokens[:, 1:]
        return input_tokens, output_tokens, valid_tokens

    def nll(self, input_tokens, output_tokens,
            current_accumulation_step):
        del current_accumulation_step
        output = self.backbone(input_tokens, None)
        output[:, :, self.mask_index] = self.neg_infinity
        output = output.log_softmax(-1)
        return - output.gather(
            -1, output_tokens[:, :, None])[:, :, 0]

    def generate_samples(self, num_samples, **kwargs):
        # precompute token buffer
        num_pred_tokens = self.num_tokens - 1
        x = torch.zeros(
            (num_samples, num_pred_tokens + 1),
            dtype=torch.long,
            device=self.device)
        x[:, 0] = self.tokenizer.bos_token_id
        # precompute noise
        noise = (torch.distributions.Gumbel(0, 1)
                 .sample((num_samples, num_pred_tokens, self.vocab_size))
                 .to(self.device))
        if self.config.sampling.use_float64:
            noise = noise.to(torch.float64)
        for i in range(num_pred_tokens):
            output = self.backbone(x[:, :i + 1], None)
            output[:, :, self.mask_index] = self.neg_infinity
            output = output.log_softmax(-1)
            y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
            x[:, i + 1] = y
        return x

    def _process_sigma(self, sigma):
        del sigma
        return None


class MDLM(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        # ancestral sampling isn't desirable because it's slow
        assert self.sampler == 'ancestral_cache'

    def _process_model_output(self, model_output, xt, sigma):
        del sigma
        model_output[:, :, self.mask_index] += self.neg_infinity

        # Normalize the model_output such that x.exp() is
        # a probability distribution over vocab_size.
        model_output = model_output - torch.logsumexp(
            model_output, dim=-1, keepdim=True)
        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = (xt != self.mask_index)
        model_output[unmasked_indices] = self.neg_infinity
        model_output[unmasked_indices, xt[unmasked_indices]] = 0
        return model_output

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                      dalpha_t, low_var=False):
        del xt
        log_p_theta = torch.gather(
            input=log_x_theta,
            dim=-1,
            index=x0[:, :, None]).squeeze(-1)
        return log_p_theta * dalpha_t / (1 - alpha_t)

    def _get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        # score(x, t) = p_t(y) / p_t(x)
        # => log score(x, t) = log p_t(y) - log p_t(x)

        # case 1: x = masked
        #   (i) y = unmasked
        #     log score(x, t) = log p_\theta(x)|_y + log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        #   (ii) y = masked
        #     log score(x, t) = 0

        # case 2: x = unmasked
        #   (i) y != masked, y != x
        #     log score(x_i, t) = - inf
        #   (ii) y = x
        #     log score(x_i, t) = 0
        #   (iii) y = masked token
        #     log score(x_i, t) = - log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))

        log_k = - torch.log(torch.expm1(sigma)).squeeze(-1)
        assert log_k.ndim == 1

        masked_score = model_output + log_k[:, None, None]
        masked_score[:, :, self.mask_index] = 0

        unmasked_score = self.neg_infinity * torch.ones_like(
            model_output)
        unmasked_score = torch.scatter(
            unmasked_score,
            -1,
            x[..., None],
            torch.zeros_like(unmasked_score[..., :1]))
        unmasked_score[:, :, self.mask_index] = - (
            log_k[:, None] * torch.ones_like(x))

        masked_indices = (x == self.mask_index).to(
            model_output.dtype)[:, :, None]
        model_output = (
            masked_score * masked_indices
            + unmasked_score * (1 - masked_indices))
        return model_output.exp()


class D3PMAbsorb(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.noise.type == 'log-linear'
        assert self.parameterization == 'mean'

    def _process_model_output(self, model_output, xt, sigma):
        del xt
        del sigma
        if self.subs_masking:
            model_output[:, :, self.mask_index] += self.neg_infinity
        return model_output.log_softmax(dim=-1)

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                      dalpha_t, low_var=False):
        del dalpha_t
        assert not low_var
        dt = 1 / self.T
        t = 1 - alpha_t  # Only valid for log-linear schedule.
        t = t.clamp(0., 1.0 - 1e-4)
        alpha_t = alpha_t + torch.zeros_like(xt)
        alpha_s = t - dt + torch.zeros_like(xt)
        assert alpha_s.shape == xt.shape
        assert alpha_t.shape == xt.shape
        log_x_theta_at_x0 = torch.gather(
            log_x_theta, -1, x0[:, :, None]).squeeze(-1)
        log_x_theta_at_m = log_x_theta[:, :, self.mask_index]
        x_theta_at_m = log_x_theta_at_m.exp()

        term_1_coef = dt / t
        term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
        term_1_log_dr = log_x_theta_at_x0

        term_2_coef = 1 - dt / t
        term_2_log_nr = term_1_log_nr
        term_2_log_dr = torch.log(
            alpha_s * x_theta_at_m / (t - dt) + 1)
        L_vb_masked = (
            term_1_coef * (term_1_log_nr - term_1_log_dr)
            + term_2_coef * (term_2_log_nr - term_2_log_dr))

        diffusion_loss = self.T * L_vb_masked * (xt == self.mask_index)
        return self._reconstruction_loss(x0) + diffusion_loss


class SEDDAbsorb(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.config.sampling.predictor == 'analytic'

    def _get_score(self, x, sigma):
        return self.forward(x, sigma).exp()

    def _process_model_output(self, model_output, xt, sigma):
        esigm1_log = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            sigma.exp() - 1).log().to(model_output.dtype)
        # logits shape
        # (batch_size, context_length, vocab_size)
        model_output = (model_output
                        - esigm1_log[:, None, None]
                        - np.log(model_output.shape[-1] - 1))
        # The below scatter operation sets the log score
        # for the input word to 0.
        model_output = torch.scatter(
            model_output, -1, xt[..., None],
            torch.zeros_like(model_output[..., :1]))
        return model_output

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                      dalpha_t, low_var=False):
        """Computes the SEDD loss for the Absorbing State Diffusion.

        Args:
          log_x_theta: float torch.Tensor with shape (batch_size,
              context_length, vocab_size),
              log score, output of the denoising network.
          xt: int torch.Tensor with shape (batch_size,
              context_length), input.
          x0: int torch.Tensor with shape (batch_size,
              context_length), input.
          alpha_t: float torch.Tensor with shape (batch_size, 1),
              signal level.
          alpha_t: float torch.Tensor with shape (batch_size, 1),
              signal level.
          dalpha_t: float or float torch.Tensor with shape (batch_size, 1),
              time derivative of signal level.
          low_var: bool, low variance loss during training.

        Returns:
          loss with shape (batch_size, context_length).
        """
        assert not low_var
        masked_indices = xt == self.mask_index
        sigma = self._sigma_from_alphat(alpha_t)
        dsigma = - dalpha_t / alpha_t

        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1 / expsig_minus_1[masked_indices]

        words_that_were_masked = x0[masked_indices]

        neg_term = q_ratio * torch.gather(
            log_x_theta[masked_indices],
            -1,
            words_that_were_masked[..., None]).squeeze(-1)
        score = log_x_theta[masked_indices].exp()
        if self.mask_index == self.vocab_size - 1:
            pos_term = score[:, :-1].sum(dim=-1)
        else:
            pos_term = score[:, : self.mask_index].sum(
                dim=-1) + score[:, self.mask_index + 1:].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)

        entropy = torch.zeros(* xt.shape, device=xt.device)
        entropy[masked_indices] += pos_term - neg_term + const
        return dsigma * entropy


def stopgrad(x):
    """Stop gradient for x."""
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2), keepdim=False)  # (B,)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


def mse_loss(error):
    per_sample = (error ** 2).mean(dim=(1, 2))  # [B]
    return per_sample.mean()


class DUO_BASE(trainer_base.UniformState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = collections.OrderedDict(
            (k, v) for k, v in checkpoint['state_dict'].items()
            if not k.startswith('teacher'))
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        # Handle _orig_mod prefix from torch.compile and filter teacher keys
        new_state_dict = collections.OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            # Filter out teacher keys
            if k.startswith('teacher'):
                continue
            # Strip _orig_mod prefix from torch.compile
            new_key = k.replace('._orig_mod.', '.')
            new_state_dict[new_key] = v
        checkpoint['state_dict'] = new_state_dict
        super().on_load_checkpoint(checkpoint)

    def _process_model_output(self, model_output, xt, sigma):
        del xt, sigma
        return model_output.log_softmax(dim=-1)

    def _compute_posterior(self, x, xt, alpha_s, alpha_t):
        """Computes the posterior / approximate posterior.

        Args:
          x: Either clean input `x0` (one-hot),
            or model's predicted `x_theta` of shape (B, L, V).
          xt: The noisy latent (as indices) of shape (B, L).
          alpha_s: Noise level at s of shape (B, [L | 1], 1).
          alpha_t: Noise level at t of shape (B, [L | 1], 1).

        Returns:
          Posterior / approximate posterior of shape (B, L, V).
        """
        if self.config.sampling.use_float64:
            x = x.to(torch.float64)
        if alpha_s.ndim == 2:
            alpha_s = alpha_s.unsqueeze(-1)
        if alpha_t.ndim == 2:
            alpha_t = alpha_t.unsqueeze(-1)
        alpha_ts = alpha_t / alpha_s
        d_alpha = alpha_s - alpha_t
        xt_one_hot = F.one_hot(xt, self.vocab_size).to(
            self.dtype).to(self.device)
        return (
            (alpha_t * self.vocab_size * x * xt_one_hot + (
                alpha_ts - alpha_t) * xt_one_hot + d_alpha * x + (
                1 - alpha_ts) * (1 - alpha_s) / self.vocab_size) / (
                alpha_t * self.vocab_size * torch.gather(
                    x, -1, xt[..., None]) + (1 - alpha_t)))

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                      dalpha_t, low_var=False):  # Computes Eq 5.
        assert alpha_t.ndim == 2
        assert x0.ndim == 2
        assert xt.ndim == 2
        if torch.is_tensor(dalpha_t) and dalpha_t.ndim == 1:
            dalpha_t = dalpha_t.unsqueeze(-1)
        assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
        x_reconst = log_x_theta.exp()  # convert logits to probabilities
        x_bar_theta = self.vocab_size * alpha_t[
            :, :, None] * x_reconst + 1 - alpha_t[:, :, None]
        coeff = dalpha_t / (self.vocab_size * alpha_t)
        x_eq_xt = (x0 == xt).float()
        x_neq_xt = 1 - x_eq_xt
        xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
        xbar_theta_xt = torch.gather(
            x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
        xbar_theta_x = torch.gather(
            x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
        term1 = self.vocab_size * (1 / xbar_xt
                                   - 1 / xbar_theta_xt)  # Eq 5. term 1

        const = (1 - alpha_t) / (self.vocab_size * alpha_t
                                 + 1 - alpha_t)
        term2_coefs = x_eq_xt * const + x_neq_xt
        term2_offset = ((self.vocab_size - 1) * const * x_eq_xt
                        - (1 / const) * x_neq_xt) * const.log()
        term2_theta = - term2_coefs * (
            x_bar_theta.log().sum(-1)
            - self.vocab_size * xbar_theta_xt.log())
        term2_theta = (
            term2_theta
            - self.vocab_size * alpha_t / (1 - alpha_t) * (
                xbar_theta_x.log() - xbar_theta_xt.log()) * x_neq_xt)
        term2 = term2_theta + term2_offset
        diffusion_loss = coeff * (term1 - term2)
        assert diffusion_loss.ndim == 2
        return diffusion_loss

    def _ancestral_update(self, x, t, dt, p_x0=None,
                          noise_removal_step=False, step_index=None):
        del p_x0
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        sigma_t = self._sigma_from_alphat(alpha_t)

        assert alpha_t.ndim == 2

        q_xs = self._compute_posterior(
            x=self.forward(x, sigma_t).exp(),
            xt=x,
            alpha_s=alpha_s,
            alpha_t=alpha_t)
        if self.p_nucleus < 1:
            q_xs = utils.top_k_top_p_filtering(
                q_xs.log(), top_p=self.p_nucleus)
        return None, trainer_base.sample_categorical(q_xs, self.config.sampling.temperature)


class Integral(torch.autograd.Function):
    """
    torch module calculating UDLM's p_t 
    """

    @staticmethod
    def forward(ctx, gamma_t, data):
        gamma_max = data['gamma_max']
        gamma_min = data['gamma_min']
        if (gamma_t.max() > gamma_max) or (
                gamma_t.min() < gamma_min):
            # print('max:{} {}'.format(gamma_t.max(), gamma_max))
            # print('min:{} {}'.format(gamma_t.min(), gamma_min))
            gamma_t = torch.clip(gamma_t, gamma_min, gamma_max)
        indices = torch.round(
            (data['num_points'] - 1) * (gamma_t - gamma_min) / (
                gamma_max - gamma_min)).long()
        grad_pt = data['grad_pt']
        ctx.grad_pt = grad_pt[indices]

        pt = data['pt'][indices]
        assert pt.shape == gamma_t.shape
        return pt

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grad_pt * grad_output, None


class DUO(DUO_BASE):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        with fsspec.open(self.config.algo.integral_cache_path,
                         'rb') as f:
            self.integral_cache = pickle.load(f)
        self.integral_cache['pt'] = torch.from_numpy(
            self.integral_cache['pt'])
        self.integral_cache['grad_pt'] = torch.from_numpy(
            self.integral_cache['grad_pt'])
        self.gamma_min = self.config.algo.gamma_min
        self.gamma_max = self.config.algo.gamma_max
        self.gumbel_tau_log10_start = self.config.algo.gumbel_tau_log10_start
        self.gumbel_tau_log10_end = self.config.algo.gumbel_tau_log10_end
        self.curriculum_start = self.config.algo.curriculum_start
        self.curriculum_end = self.config.algo.curriculum_end
        self.loss_type = self.config.training.loss_type
        # assert self.loss_type in {'flow', 'meanflow'}
        self._validate_configuration()
        self.log_flag = False

    def sample_t_t_prime(self, batch_size, device, accum_step=None):
        t_min = getattr(self.config.algo, 't_min', 0.0)
        t_max = getattr(self.config.algo, 't_max', 1.0)

        if accum_step is not None:
            local_batch_size = batch_size
            n = self.config.loader.global_batch_size
        else:
            n = batch_size

        eps = torch.rand(n, 2, device=device)

        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            offset = offset.unsqueeze(1)  # (N, 1)

            eps = (eps / n + offset) % 1.0

            perm = torch.randperm(n, device=device)
            eps[:, 1] = eps[perm, 1]

        if accum_step is not None:
            eps = eps.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            eps = eps.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            eps = eps.chunk(self.trainer.accumulate_grad_batches)[accum_step]

            eps = eps[:local_batch_size]

        samples = eps * (t_max - t_min) + t_min

        t = torch.minimum(samples[:, 0], samples[:, 1])
        r = torch.maximum(samples[:, 0], samples[:, 1])

        if self.config.algo.flow_ratio > 0:
            num_selected = int(self.config.algo.flow_ratio * eps.shape[0])
            indices = torch.randperm(eps.shape[0], device=device)[:num_selected]
            r[indices] = t[indices]

        return t, r

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.integral_cache['pt'] = self.integral_cache[
            'pt'].to(*args, **kwargs)
        self.integral_cache['grad_pt'] = self.integral_cache[
            'grad_pt'].to(*args, **kwargs)
        return self

    def _duo_alpha_and_dalpha_dt(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        t: (B,) in [0,1]
        returns:
        alpha_t: (B,)
        dalpha_dt: (B,)
        Uses the same finite-diff trick as DUO.nll so sampling matches training.
        """
        gamma_t = self.gamma_min + t * \
            (self.gamma_max - self.gamma_min)  # (B,) -3.75 ~ -1.75

        # this is problem... mapping to discrete alpha_t
        alpha_t = self._gamma_to_alphat(
            gamma_t)                           # (B,)

        # match training: dalpha_t = gamma_prime * T * (alpha(gamma+1/T) - alpha(gamma))
        T_fd = 200
        gamma_prime = (self.gamma_max - self.gamma_min)
        alpha_t_plus = self._gamma_to_alphat(gamma_t + 1.0 / T_fd)
        dalpha_dt = gamma_prime * T_fd * \
            (alpha_t_plus - alpha_t)          # (B,)
        return alpha_t, dalpha_dt

    def _compute_gumbel_tau_inverse(self):
        if self.config.mode == 'sample_eval':
            tau = -10
            return 10 ** (-tau)
        start = self.gumbel_tau_log10_start
        end = self.gumbel_tau_log10_end
        delta = end - start
        if self.global_step < self.curriculum_start:
            tau = start
        elif self.global_step < self.curriculum_end:
            frac = (self.global_step - self.curriculum_start) / (
                self.curriculum_end - self.curriculum_start)
            tau = start + frac * delta
        else:
            tau = end
        return 10 ** (-tau)

    def training_step(self, batch, batch_idx):
        self.log(name='gumbel_tau_log10',
                 value=1 / self._compute_gumbel_tau_inverse(),
                 on_step=True,
                 on_epoch=False,
                 sync_dist=True)
        return super().training_step(batch, batch_idx)

    def _gamma_to_alphat(self, gamma_t):  # eq 10.
        integral = Integral.apply(gamma_t, self.integral_cache)
        return (self.vocab_size * integral - 1) / (
            self.vocab_size - 1)

    def _prior_loss(self):
        alpha_1 = self._gamma_to_alphat(
            torch.tensor(self.gamma_max))
        loss = ((alpha_1 + (1 - alpha_1) / self.vocab_size) * torch.log(
            (self.vocab_size - 1) * alpha_1 + 1) + (
            1 - 1 / self.vocab_size) * (1 - alpha_1) * torch.log(1 - alpha_1))
        return loss.item()

    def _q_xt_gaussian(self, x, gamma_t):
        """Computes the noisy sample xt."""
        assert gamma_t.ndim == 1
        assert x.ndim == 3
        gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
        alpha_t = torch.sigmoid(-gamma_t).sqrt()
        sigma_t = torch.sigmoid(gamma_t).sqrt()
        epsilon = torch.randn(x.shape, dtype=torch.float32,
                              device=self.device)  # sample noise
        # import ipdb; ipdb.set_trace()
        self.log('alpha_t', alpha_t.mean(), prog_bar=True, sync_dist=True)
        self.log('sigma_t', sigma_t.mean(), prog_bar=True, sync_dist=True)
        return alpha_t * x + sigma_t * epsilon, epsilon  # add noise to x

    def flow_loss(self, x0, output_tokens,
                  current_accumulation_step=None, train_mode=False, xT=None, given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t
        del output_tokens
        B = x0.shape[0]
        V = self.vocab_size
        if self.global_step % 500 != 0:
            self.log_flag = False
        t = self._sample_t_interval(B, current_accumulation_step)  # sample t

        t_lin = t.unsqueeze(-1).unsqueeze(-1)

        x0_one_hot = F.one_hot(x0, V)
        noise_eps = torch.randn_like(x0_one_hot, dtype=torch.float32)
        x_t = (1-t_lin)*x0_one_hot + t_lin * noise_eps  # noise along alpha_t ? not t
        # atimport ipdb; ipdb.set_trace()
        self.log('t_lin', t_lin.mean(), prog_bar=True, sync_dist=True)

        if self.config.algo.use_curriculum == True:
            tau_inverse = self._compute_gumbel_tau_inverse()
            x_t_tempered = x_t * tau_inverse
        else:
            x_t_tempered = x_t

        if self.config.training.pred_type == 'velocity':
            v_pred = self.backbone.forward(x_t_tempered, t)
            v_tgt = x0_one_hot - noise_eps
            error = (v_tgt - v_pred)  # (B, L, V)
            v_pred_debug = self.tokenizer.decode((x_t+t_lin*v_pred).argmax(dim=-1)[0].cpu().numpy())
            if self.global_step % 1000 == 0 and self.log_flag == False:
                self.trainer.logger.log_table(
                    key=f'v_pred_debug@global_step{self.global_step}',
                    columns=['Generated_Samples'],
                    data=[[v_pred_debug]])
                self.log_flag = True
            self.log('error_mean', error.abs().mean(), prog_bar=True, sync_dist=True)
            loss_per_token = (error ** 2).mean(dim=-1)  # (B, L)
            loss_per_token = loss_per_token * self.vocab_size
            return loss_per_token
        elif self.config.training.pred_type == 'x0':
            x_0_pred = self.forward(x_t_tempered, t.unsqueeze(-1))
            error = x0_one_hot - x_0_pred.exp()
            # print(self.tokenizer.decode((x_0_pred.exp()).argmax(dim=-1)[0].cpu().numpy()))

        self.log('error_mean', error.abs().mean(),
                 prog_bar=True, sync_dist=True)
        loss_per_token = (error ** 2).mean(dim=-1)  # (B, L)
        loss_per_token = loss_per_token * self.vocab_size
        # loss_per_token = loss_per_token
        return loss_per_token

    def meanflow_loss(self, x0, output_tokens,
                      current_accumulation_step=None, train_mode=False, xT=None, given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t
        del output_tokens
        if self.global_step % self.config.algo.pred_log_interval != 0:
            self.log_flag = False
        B = x0.shape[0]
        V = self.vocab_size
        jvp_flag = True

        if self.global_step < self.config.algo.flow_warmup_steps:
            t = self._sample_t_interval(B, current_accumulation_step)
            r = torch.ones_like(t)
        else:
            t, r = self.sample_t_t_prime(B, self.device, accum_step=current_accumulation_step)
        t_lin = t.unsqueeze(-1).unsqueeze(-1).requires_grad_(True)
        r_lin = r.unsqueeze(-1).unsqueeze(-1)

        sigma_min = self.config.algo.sigma_min
        sigma_t = 1 - (1 - sigma_min) * t_lin

        target_data = F.one_hot(x0, V).float()  # x1 (Data)
        noise = torch.randn_like(target_data)  # x0 (Noise)

        x_t = t_lin * target_data + (1 - t_lin) * noise

        if self.config.algo.use_curriculum == True:
            tau_inverse = self._compute_gumbel_tau_inverse()
            x_t = x_t * tau_inverse
        else:
            x_t = x_t

        # weights = 1.0 / (sigma_t ** 2)
        weights = 1.0
        if self.global_step < self.config.algo.flow_warmup_steps:
            jvp_flag = False
            # During warmup, use single timestep forward
            f = self.forward(x_t, t.unsqueeze(-1)).exp()
        else:
            f = self.forward_double_timestep(x_t, t, r, jvp_flag).exp()
        # print(self.tokenizer.decode((f).argmax(dim=-1)[0].cpu().numpy()))
        diff = f - target_data
        tfm_loss = (diff ** 2 * weights).mean(dim=-1)*self.vocab_size
        self.log('error_mean', diff.abs().mean(), prog_bar=True, sync_dist=True)
        self.log('tfm_loss', tfm_loss.mean(), prog_bar=True, sync_dist=True)
        if self.global_step % self.config.algo.pred_log_interval == 0 and self.log_flag == False:
            f_debug = self.tokenizer.decode((f).argmax(dim=-1)[0].cpu().numpy())
            print(f_debug)
            self.log_flag = True

        if self.global_step < self.config.algo.flow_warmup_steps:
            return tfm_loss

        # print(t,r)

        # Calculate df/dt using JVP
        # NOTE: Do NOT torch.compile this inner function.
        # AOTAutograd currently does not support double backward,
        # and JVP is called inside the training/validation graph.
        def jvp_fn(t_scalar):
            xt_inner = t_scalar * target_data + (1 - t_scalar) * noise
            return self.forward_double_timestep(
                xt_inner,
                t_scalar.reshape(B),
                r.reshape(B)
            ).exp()

        _, df_dt = torch.autograd.functional.jvp(
            jvp_fn,
            (t_lin,),
            (torch.ones_like(t_lin),),
            create_graph=False
        )

        # Flow Matching Loss: || f - x1 ||^2

        # MeanFlow Consistency
        # 2 * (t - r) * (f - (1-sigma_min)x_t)^T * df_dt
        v_term = f - (1 - sigma_min) * x_t
        dot_product = (v_term * df_dt).sum(dim=-1, keepdim=True)
        cross_coeff = 2 * (t_lin - r_lin) * weights
        cross_term = (cross_coeff * dot_product).mean(dim=-1)

        self.log('cross_term', cross_term.mean(), prog_bar=True, sync_dist=True)
        self.log('total_loss', (tfm_loss + cross_term).mean(), prog_bar=True, sync_dist=True)
        loss = (tfm_loss + cross_term)
        return loss

    def nll(self, x0, output_tokens,
            current_accumulation_step=None, train_mode=False, xT=None, given_t=None, not_sampling_t=False):
        # TODO: use xT
        # TODO: check given_t and not_sampling_t
        del given_t, not_sampling_t
        B = x0.shape[0]
        V = self.vocab_size
        if self.global_step % 500 != 0:
            self.log_flag = False

        # use_true_nll = (self.global_step > self.curriculum_end
        #                 or not train_mode)  # after curriculum end, operate on discrete space (B, L)
        # if use_true_nll:
        #     return super().nll(x0, output_tokens,
        #                        current_accumulation_step)
        del output_tokens
        t = self._sample_t(B, current_accumulation_step)  # sample t
        gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)
        gamma_t_prime = self.gamma_max - self.gamma_min
        usdm_alpha_t = self._gamma_to_alphat(gamma_t)
        # import ipdb; ipdb.set_trace()
        T = 200
        usdm_dalpha_t = gamma_t_prime * T * (self._gamma_to_alphat(gamma_t + 1 / T) - usdm_alpha_t)
        usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
        usdm_dalpha_t = usdm_dalpha_t.unsqueeze(-1)
        assert usdm_alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(usdm_alpha_t)

        x0_one_hot = F.one_hot(x0, V)
        xt, epsilon = self._q_xt_gaussian(x0_one_hot, gamma_t)  # (B, L, V)
        xt = xt * self._compute_gumbel_tau_inverse()  # multiply 1/temperature before applying softmax
        xt_usdm = xt.argmax(-1)

        if self.loss_type == 'mse':
            # mse loss
            log_x_theta = self.forward(xt, sigma=sigma)  # output log prob
            error = x0_one_hot - log_x_theta.exp()
            loss_per_token = (error**2).mean(dim=-1)
            self.log('error_mean', error.abs().mean(),
                     prog_bar=True, sync_dist=True)

            loss_per_token = loss_per_token * self.vocab_size
            return loss_per_token
        elif self.loss_type == 'adaptive_l2':
            # adaptive l2 loss (per-token): sg(w) * ||Δ||^2 with w = 1/(||Δ||^2 + c)^(1-γ)
            # meanflow hyperparmaeters
            error = x0_one_hot - log_x_theta.exp()
            self.log('error_mean', error.abs().mean(),
                     prog_bar=True, sync_dist=True)

            delta_sq = (error ** 2).mean(dim=-1)  # (B, L)
            gamma = 0.5
            c = 1e-3
            p = 1.0 - gamma
            w = 1.0 / (delta_sq + c).pow(p)
            loss_per_token = stopgrad(w) * delta_sq  # (B, L)
            loss_per_token = loss_per_token * self.vocab_size
            return loss_per_token
        elif self.loss_type == 'flow':
            v_pred = self.forward(xt, sigma=sigma)
            error = epsilon - x0_one_hot - v_pred.exp()
            loss_per_token = (error**2).mean(dim=-1)
            self.log('error_mean', error.abs().mean(),
                     prog_bar=True, sync_dist=True)

            loss_per_token = loss_per_token * self.vocab_size
            return loss_per_token

        else:
            # original nll loss
            log_x_theta = self.forward(xt, sigma=sigma)  # output log prob

            log_x_theta_debug = self.tokenizer.decode(
                log_x_theta.argmax(-1)[0].cpu().numpy())
            log_x_theta_exp_debug = self.tokenizer.decode(
                log_x_theta.exp().argmax(-1)[0].cpu().numpy())

            if self.global_step % 500 == 0 and self.log_flag == False:
                # print(f'Global step: {self.global_step}, Logits: {log_x_theta_debug}')
                self.trainer.logger.log_table(
                    key=f'log_x_theta@global_step{self.global_step}',
                    columns=['Generated_Samples'],
                    data=[[log_x_theta_debug]])
                self.trainer.logger.log_table(
                    key=f'log_x_theta_exp@global_step{self.global_step}',
                    columns=['Generated_Samples'],
                    data=[[log_x_theta_exp_debug]])
                self.log_flag = True

            return self.nll_per_token(log_x_theta=log_x_theta,
                                      xt=xt_usdm,
                                      x0=x0,
                                      alpha_t=usdm_alpha_t,
                                      dalpha_t=usdm_dalpha_t,
                                      low_var=False)


class Distillation(DUO):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.update_teacher_every = config.algo.update_teacher_every
        self.save_hyperparameters()
        self.teacher = None
        self.teacher_ema = config.algo.teacher_ema
        self.linear_growth_dt = config.algo.linear_growth_dt
        self.linear_growth_min = config.algo.linear_growth_min
        self.linear_growth_max = config.algo.linear_growth_max

    def _validate_configuration(self):
        assert os.path.exists(
            self.config.algo.integral_cache_path), (
            'The integral cache (Eq. 10 in the paper) for '
            f'the {self.config.data.tokenizer_name_or_path} '
            ' tokenizer doesnt exist at '
            f'{self.config.algo.integral_cache_path}. '
            'Please generate it by running the utils.py script, '
            'and ensure the correct path is specified using the '
            'algo.integral_cache_path flag.')
        assert self.loss_type in {
            'kl-fwd', 'kl-bwd', 'posterior', 'kl-posterior'}

    def _maybe_update_teacher_weights(self):
        if self.global_step % self.update_teacher_every != 0:
            return
        if self.teacher_ema:
            self.ema.copy_to(self.teacher.parameters())
        else:
            for better_param, current_param in zip(
                    self.backbone.parameters(), self.teacher.parameters()):
                if current_param.requires_grad:
                    current_param.data.copy_(better_param.data)

    @torch.no_grad()
    def _teacher_logits(self, xt, sigma):
        if self.teacher is None:
            self.teacher = copy.deepcopy(self.backbone)
        self._maybe_update_teacher_weights()

        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.teacher(xt, sigma)
        logits = self._process_model_output(
            model_output=model_output, xt=xt, sigma=sigma)
        return logits.detach()

    def _sample_trajectory(self, x0, gamma_t, gamma_s):
        """Computes the noisy sample xt."""
        assert gamma_t.ndim == 1
        assert gamma_s.ndim == 1
        assert x0.ndim == 2
        x0 = F.one_hot(x0, self.vocab_size).to(
            self.dtype).to(self.device)
        gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
        alpha_t = torch.sigmoid(-gamma_t).sqrt()
        sigma_t = torch.sigmoid(gamma_t).sqrt()

        gamma_s = gamma_s.unsqueeze(-1).unsqueeze(-1)
        alpha_s = torch.sigmoid(-gamma_s).sqrt()
        sigma_s = torch.sigmoid(gamma_s).sqrt()

        epsilon = torch.randn(x0.shape, dtype=torch.float32,
                              device=self.device)
        xt = alpha_t * x0 + sigma_t * epsilon
        xs = alpha_s * x0 + sigma_s * epsilon
        return xt, xs

    def _compute_dt(self):
        if self.linear_growth_dt:
            scale = self.global_step / self.trainer.max_steps
            return self.linear_growth_min + scale * (
                self.linear_growth_max - self.linear_growth_min)
        n = self.global_step // self.update_teacher_every
        return 2 ** n / self.T

    def nll(self, x0, output_tokens,
            current_accumulation_step=None, train_mode=None, xT=None):
        # TODO: use xT
        del output_tokens, train_mode
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        dt = self._compute_dt()
        t = torch.clip(t + dt, 0, 1)

        gamma_t = self.gamma_min + t * (self.gamma_max
                                        - self.gamma_min)
        gamma_s = self.gamma_min + (
            t - dt) * (self.gamma_max - self.gamma_min)

        alpha_t = self._gamma_to_alphat(gamma_t)
        alpha_t = alpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        usdm_alpha_s = self._gamma_to_alphat(gamma_s)
        usdm_alpha_s = usdm_alpha_s.unsqueeze(-1)
        assert usdm_alpha_s.ndim == 2

        xt, xs = self._sample_trajectory(x0, gamma_t, gamma_s)
        xt_discrete = xt.argmax(-1)
        xs_discrete = xs.argmax(-1)
        log_x_theta_student = self.forward(
            xt_discrete, sigma=self._sigma_from_alphat(alpha_t))
        log_x_theta_teacher = self._teacher_logits(
            xs_discrete, sigma=self._sigma_from_alphat(usdm_alpha_s))
        if self.config.training.loss_precision == 'float64':
            log_x_theta_student = log_x_theta_student.to(torch.float64)
            log_x_theta_teacher = log_x_theta_teacher.to(torch.float64)
        if self.loss_type == 'kl-fwd':
            return (log_x_theta_teacher.exp() * (
                log_x_theta_teacher - log_x_theta_student)).sum(-1)
        elif self.loss_type == 'kl-bwd':
            return (log_x_theta_student.exp() * (
                log_x_theta_student - log_x_theta_teacher)).sum(-1)

    def training_step(self, batch, batch_idx):
        self.log(name='dt',
                 value=self._compute_dt(),
                 on_step=True,
                 on_epoch=False,
                 sync_dist=True)
        return super().training_step(batch, batch_idx)


class Rectification(DUO):  # Training as duo, without curriculum
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.save_hyperparameters()
        self.use_linear_schedule = config.algo.use_linear_schedule
        self.use_simple_loss = config.algo.use_simple_loss
        self.onestep_mode = config.algo.onestep_mode
        self.debug = getattr(config.algo, 'debug', False)

    def _compute_gumbel_tau_inverse(self):
        return 1e-10

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t,
                      dalpha_t, low_var=False, simple_loss=False):
        if simple_loss:
            loss = F.cross_entropy(
                log_x_theta.view(-1, self.vocab_size),
                x0.view(-1),
                reduction='none')
            loss = loss.view(xt.shape)
            return loss
        else:
            return super().nll_per_token(
                log_x_theta=log_x_theta,
                xt=xt,
                x0=x0,
                alpha_t=alpha_t,
                dalpha_t=dalpha_t,
                low_var=low_var
            )

    def nll(self, x0, output_tokens,
            current_accumulation_step=None, train_mode=False, xT=None, given_t=None, not_sampling_t=False):
        del output_tokens
        if given_t is not None:
            if not_sampling_t:
                assert torch.is_tensor(given_t)
                t = 1-given_t
            else:
                t = self._sample_t(
                    x0.shape[0], current_accumulation_step, given_t=1-given_t)
        else:
            t = self._sample_t(x0.shape[0], current_accumulation_step)
        assert t.shape[0] == x0.shape[0]
        if self.T > 0:
            assert 0

        dalpha_t, alpha_t = self.noise(t)

        alpha_t = alpha_t.unsqueeze(-1)
        dalpha_t = dalpha_t.unsqueeze(-1)
        assert alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(alpha_t)

        if given_t is not None and xT is not None:
            # x0 with alpha_t, xT with (1-alpha_t)
            random = torch.rand_like(x0, dtype=torch.float32)
            given_t = given_t.unsqueeze(1)
            random = given_t + random * (1 - given_t)
            if self.onestep_mode:
                # always larger than alpha_t
                random = torch.ones_like(random) + 1
            xt = torch.where(random <= alpha_t, x0, xT)
        elif xT is None or self.debug:
            if not self.debug:
                assert not self.training, 'xT should be provided during training'
            xT = self.prior_sample(x0.shape[0], x0.shape[1])
            random = torch.rand_like(x0, dtype=torch.float32)
            if self.onestep_mode:
                # always larger than alpha_t
                random = torch.ones_like(random) + 1
            xt = torch.where(random <= alpha_t, x0, xT)
        else:
            # x0 with alpha_t, xT with (1-alpha_t)
            random = torch.rand_like(x0, dtype=torch.float32)
            if self.onestep_mode:
                # always larger than alpha_t
                random = torch.ones_like(random) + 1
            xt = torch.where(random <= alpha_t, x0, xT)

        log_x_theta = self.forward(xt, sigma=sigma)

        return self.nll_per_token(log_x_theta=log_x_theta,
                                  xt=xt,
                                  x0=x0,
                                  alpha_t=alpha_t,
                                  dalpha_t=dalpha_t,
                                  low_var=train_mode and self.loss_type == 'low_var',
                                  simple_loss=self.use_simple_loss,
                                  )


class DOS(trainer_base.TrainerBase):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()
        self.flow_ratio = config.algo.flow_ratio
        self.jvp_api = config.algo.jvp_api
        self.gumbel_tau_log10_start = config.algo.gumbel_tau_log10_start
        self.gumbel_tau_log10_end = config.algo.gumbel_tau_log10_end
        self.curriculum_start = config.algo.curriculum_start
        self.curriculum_end = config.algo.curriculum_end
        self.sigma_min = config.algo.sigma_min
        self.t_min = config.algo.t_min
        self.t_max = config.algo.t_max
        self.use_curriculum = config.algo.use_curriculum
        self.log_flag = False

        assert self.jvp_api in [
            'funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if self.jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif self.jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

        self.lut_a2g, self.lut_g2a = utils.build_luts(K=self.vocab_size)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = collections.OrderedDict(
            (k, v) for k, v in checkpoint['state_dict'].items()
            if not k.startswith('teacher'))
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        new_state_dict = collections.OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if k.startswith('teacher'):
                continue
            new_key = k.replace('._orig_mod.', '.')
            new_state_dict[new_key] = v

        if self.config.mode != 'sample_eval':
            if self.config.algo.double_temb and self.backbone.sigma_map_prime is not None:
                if not any(k.startswith('backbone.sigma_map_prime') for k in new_state_dict.keys()):

                    print("[INFO] Adding sigma_map_prime to state_dict (Last-Layer Zero Init)")

                    for name, param in self.backbone.sigma_map_prime.named_parameters():
                        param_key = f'backbone.sigma_map_prime.{name}'

                        if 'mlp.2' in name:
                            print(name)
                            print("zero init mlp.2")
                            zero_tensor = torch.zeros_like(param.data)
                            new_state_dict[param_key] = zero_tensor
                            param.data.copy_(zero_tensor)
                        else:
                            new_state_dict[param_key] = param.data.clone()

            # Add output_layer_sc if missing
            if not any(k.startswith('backbone.output_layer_sc') for k in new_state_dict.keys()):
                print("[INFO] Adding output_layer_sc to state_dict")
                for name, param in self.backbone.output_layer_sc.named_parameters():
                    param_key = f'backbone.output_layer_sc.{name}'
                    new_state_dict[param_key] = param.data.clone()

        checkpoint['state_dict'] = new_state_dict
        super().on_load_checkpoint(checkpoint)

    def _compute_gumbel_tau_inverse(self):
        if self.config.mode == 'sample_eval':
            tau = self.gumbel_tau_log10_end
            return 10 ** (-tau)
        start = self.gumbel_tau_log10_start
        end = self.gumbel_tau_log10_end
        delta = end - start
        if self.global_step < self.curriculum_start:
            tau = start
        elif self.global_step < self.curriculum_end:
            frac = (self.global_step - self.curriculum_start) / (
                self.curriculum_end - self.curriculum_start)
            tau = start + frac * delta
        else:
            tau = end
        return 10 ** (-tau)

    def training_step(self, batch, batch_idx):
        self.log(name='gumbel_tau_log10',
                 value=1 / self._compute_gumbel_tau_inverse(),
                 on_step=True,
                 on_epoch=False,
                 sync_dist=True)
        return super().training_step(batch, batch_idx)

    def _validate_configuration(self):
        pass

    def _process_sigma(self, sigma):
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.config.algo.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _process_model_output(self, model_output, xt, sigma, cap_value = 30.0):
        del xt, sigma
        model_output = cap_value * torch.tanh(model_output / cap_value)
        return model_output.log_softmax(dim=-1)

    def forward_with_ema(self, *args, **kwargs):
        
        ema_to_use = self.shortcut_ema if self.shortcut_ema is not None else self.ema
        assert ema_to_use is not None, "Either shortcut_ema or ema must be available"
        
        ema_to_use.store(self._get_parameters())
        ema_to_use.copy_to(self._get_parameters())
        try:
            with torch.no_grad():
                self.backbone.eval()
                out = self.forward(*args, **kwargs)
            return out
        finally:
            ema_to_use.restore(self._get_parameters())
            self.backbone.train()

    def _sample_t_interval(self, n, accum_step, t_min=None, t_max=None):
        if t_min is None:
            t_min = self.t_min
        
        if t_max is None:
            t_max = self.t_max
        
        if accum_step is not None:
            # During training
            batch_dim = n
            n = self.config.loader.global_batch_size
        _eps_t = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
            perm = torch.randperm(n, device=self.device)
            _eps_t = _eps_t[perm]

        t = (t_max - t_min) * _eps_t + t_min
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[
                accum_step]
            # corner case for the last datapoint
            t = t[:batch_dim]
        return t

    def sample_t_t_prime(self, batch_size, device, accum_step=None):
        t_min = self.t_min
        t_max = self.t_max

        if accum_step is not None:
            local_batch_size = batch_size
            n = self.config.loader.global_batch_size
        else:
            n = batch_size

        eps = torch.rand(n, 2, device=device)

        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            offset = offset.unsqueeze(1)  # (N, 1)

            eps = (eps / n + offset) % 1.0

            perm = torch.randperm(n, device=device)
            eps[:, 1] = eps[perm, 1]
            perm_global = torch.randperm(n, device=device)
            eps = eps[perm_global]

        if accum_step is not None:
            eps = eps.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            eps = eps.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            eps = eps.chunk(self.trainer.accumulate_grad_batches)[accum_step]
            eps = eps[:local_batch_size]

        # Map to [t_min, t_max]
        samples = eps * (t_max - t_min) + t_min  # shape (n_local, 2)
        t_raw = samples[:, 0]
        r_raw = samples[:, 1]

        t = torch.minimum(t_raw, r_raw)
        r = torch.maximum(t_raw, r_raw)

        # flow_ratio fraction where r = t (pure FM points)
        if self.config.algo.flow_ratio > 0:
            num_selected = int(self.config.algo.flow_ratio * t.shape[0])
            indices = torch.randperm(t.shape[0], device=device)[:num_selected]
            r[indices] = t[indices]

        return t, r

    # convert discrete time schedule alpha_t to continuous time schedule gamma_t
    def _alpha_t_to_gamma(self, alpha_t):
        return utils.alpha_to_gamma(alpha_t, self.lut_a2g)

    def _gamma_to_alphat(self, gamma_t):
        return utils.gamma_to_alpha(gamma_t, self.lut_g2a)

    def corrupt_continuous(self, x0, t):
        t = t.unsqueeze(-1).unsqueeze(-1)

        target_data = F.one_hot(x0, self.vocab_size).float()
        noise = torch.randn_like(target_data, dtype=torch.float32)
        x_t = (1 - t) * noise + t * target_data
        return x_t, target_data
    
    def flow_loss(self, x0, output_tokens,
                  current_accumulation_step=None, train_mode=False, xT=None, given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t
        del output_tokens
        B = x0.shape[0]

        if self.global_step % 1000 != 0:
            self.log_flag = False

        if self.config.algo.use_discrete_schedule:              
            t = self._sample_t_interval(B, current_accumulation_step, t_min = 0.0, t_max = 1.0)
            c_t = self._alpha_t_to_gamma(t)
            self.log('t', t.mean(), prog_bar=True, sync_dist=True)
            self.log('c_t', c_t.mean(), prog_bar=True, sync_dist=True)
        else:  # use heuristic continuous time schedule
            c_t = self._sample_t_interval(B, current_accumulation_step, t_min = self.t_min, t_max = self.t_max)

        x_t, target_data = self.corrupt_continuous(x0, c_t)

        if self.config.algo.use_curriculum == True:
            tau_inverse = self._compute_gumbel_tau_inverse()
            if self.config.algo.scale_input:
                scale = 1.0 / (1.0 - c_t.view(-1, 1, 1) + 1e-5)
            else:
                scale = 1.0
            x_t = x_t * tau_inverse * scale

        if self.config.algo.use_discrete_schedule:
            if self.config.algo.time_condition == 'alpha_t':
                f = self.forward(x_t, t)
            else:
                f = self.forward(x_t, c_t)
        else:
            f = self.forward(x_t, c_t)

        if self.config.algo.flow_loss_type == 'mse':
            error = target_data - f.exp()
            tfm_loss = (error ** 2).mean(dim=-1) 
            tfm_loss = tfm_loss * self.vocab_size
        else: # cross-entropy            
            tfm_loss = -(target_data * f).sum(dim=-1)
            
        self.log('tfm_loss', tfm_loss.mean(), prog_bar=True, sync_dist=True)

        if self.global_step % 500 == 0 and self.log_flag == False:
            f_debug = self.tokenizer.decode(f.exp().argmax(dim=-1)[0])
            print(c_t[0], f_debug)
            self.log_flag = True

        if self.trainer.is_global_zero:
            c_final = c_t

            num_bins = 10
            for b in range(num_bins):
                lo = b / num_bins
                hi = (b + 1) / num_bins
                mask_fm = (c_final >= lo) & (c_final < hi)
                if mask_fm.any():
                    bin_loss_fm = tfm_loss[mask_fm].mean()
                    self.log(
                        f'tfm_loss_gamma_bin_{b}',
                        bin_loss_fm,
                        prog_bar=False,
                        sync_dist=False,
                    )

        return tfm_loss
    
    def shortcut_loss(self, x0, output_tokens,
                      current_accumulation_step=None, train_mode=False, xT=None, given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t
        del output_tokens
        B = x0.shape[0]
        L = x0.shape[1]
        V = self.vocab_size
        K_max = self.config.algo.shortcut_k_max
        d_min = 1.0 / K_max
        max_power = int(math.log2(K_max))

        num_fm = int(B * self.flow_ratio)
        num_sc = B - num_fm

        x0_fm = x0[:num_fm]
        x0_sc = x0[num_fm:]

        x_t_final = torch.empty((B, L, V), device=self.device, dtype=torch.float32)
        target_final = torch.empty((B, L, V), device=self.device, dtype=torch.float32)
        t_final = torch.empty((B,), device=self.device)
        d_final = torch.empty((B,), device=self.device)

        if self.config.algo.use_curriculum == True:
            tau_inverse_fm = 10**(-self.config.algo.tau_log10_fm)
            tau_inverse_sc = 10**(-self.config.algo.tau_log10_shortcut)
        else:
            tau_inverse = 10
        
        # Cache scale_input condition to avoid repeated checks
        scale_input = self.config.algo.scale_input
        
        # flow matching
        if num_fm > 0:
            
            t_fm = self._sample_t_interval(num_fm, current_accumulation_step, t_min = 0.0, t_max = 1.0)
            d_fm = self._sample_t_interval(num_fm, current_accumulation_step, t_min = 0.0, t_max = 1.0).clamp(min=1e-5, max=1.0)
            c_t_fm = self._alpha_t_to_gamma(t_fm)
            x_t_fm, target_data_fm = self.corrupt_continuous(x0_fm, c_t_fm)

            if scale_input:
                scale = 1.0 / (1.0 - c_t_fm.view(-1, 1, 1) + 1e-5)
            else:
                scale = 1.0
            
            x_t_final[:num_fm] = x_t_fm * tau_inverse_fm * scale
            target_final[:num_fm] = target_data_fm
            t_final[:num_fm] = t_fm
            # d_final[:num_fm] = 0.0
            d_final[:num_fm] = d_fm # random d for flow matching

            del x0_fm, x_t_fm, target_data_fm, t_fm, c_t_fm

        # shortcut
        if num_sc > 0:
            if self.config.algo.sample_d_on_grid:
                k_sc = torch.randint(1, max_power + 1, (num_sc,), device=self.device)
                d_sc = (2.0 ** k_sc) * d_min

                num_steps = (1.0 / d_sc).round()
                i = (torch.rand(num_sc, device=self.device) * num_steps).floor()
                t_sc = i * d_sc
            elif self.config.algo.use_continuous_shortcut:

                d_sc = self._sample_t_interval(num_sc, current_accumulation_step, t_min = 0.0, t_max = 1.0).clamp(min=1e-5, max=1.0)
                t_sc = torch.rand(num_sc, device=self.device) * (1.0 - d_sc)

                if self.config.algo.add_boundary:
                    p_boundary = 1.0 / (max_power)
                    is_boundary = torch.rand(num_sc, device=self.device) < p_boundary
                    
                    t_sc = torch.where(is_boundary, torch.tensor(0.0, device=self.device), t_sc)
                    d_sc = torch.where(is_boundary, torch.tensor(1.0, device=self.device), d_sc)
                
            else:
                t_sc = self._sample_t_interval(num_sc, current_accumulation_step, t_min = 0.0, t_max = 1.0)

                p_boundary = 1.0 / (max_power) 
                is_boundary = torch.rand(num_sc, device=self.device) < p_boundary
                
                remaining = (1.0 - t_sc) * K_max
                max_k_float = torch.log2(remaining.clamp(min=1.0))
                max_k = max_k_float.floor().clamp(min=1, max=max_power).long()
                
                no_valid_step = remaining < 2.0
                u = torch.rand(num_sc, device=self.device)
                k_sc = (u * max_k.float()).floor().long() + 1
                
                d_sc_temp = (2.0 ** k_sc) * d_min
                d_sc_temp = torch.where(no_valid_step, 1.0 - t_sc, d_sc_temp)
                
                t_sc = torch.where(is_boundary, torch.tensor(0.0, device=self.device), t_sc)
                d_sc = torch.where(is_boundary, torch.tensor(1.0, device=self.device), d_sc_temp)

            if self.config.algo.shortcut_on_alpha_t:
                c_t_sc = self._alpha_t_to_gamma(t_sc)
                x_t_sc, _ = self.corrupt_continuous(x0_sc, c_t_sc)

                with torch.no_grad():

                    c_t_sc = c_t_sc.view(-1, 1, 1)
                    c_t_end = self._alpha_t_to_gamma(t_sc + d_sc).view(-1, 1, 1)
                    c_d = c_t_end - c_t_sc
                    c_d_half = c_d / 2.0
                    d_half_1 = self._gamma_to_alphat(c_t_sc + c_d_half) - self._gamma_to_alphat(c_t_sc)
                    d_half_2 = self._gamma_to_alphat(c_t_end) - self._gamma_to_alphat(c_t_sc + c_d_half)
                    d_half_1 = d_half_1.squeeze()
                    d_half_2 = d_half_2.squeeze()
                    
                    
                    if getattr(self.config.algo, 'shortcut_mix_logit', False):
                        # Model mixes logits internally
                        pred_x1_s1 = (self.forward_with_ema if self.config.algo.bootstrap_ema else self.forward)(
                            x_t_sc * tau_inverse_fm,
                            t_sc, # condition on always alpha
                            d_half_1,
                            use_auxiliary_head=True,
                            c_d=c_d_half,
                            c_t=c_t_sc,
                        )
                        pred_x1_s1 = pred_x1_s1.detach().exp()
                    else:
                        # Mix after exp (original behavior)
                        pred_x1_s1_fm_head, pred_x1_s1_sc_head = (self.forward_with_ema if self.config.algo.bootstrap_ema else self.forward)(
                            x_t_sc * tau_inverse_fm,
                            t_sc, # condition on always alpha
                            d_half_1,
                            use_auxiliary_head=True,
                        )
                        pred_x1_s1_fm_head = pred_x1_s1_fm_head.detach()
                        
                        if self.config.algo.shortcut_mix_type == 'interpolate':
                            pred_x1_s1 = (1.0 - c_d_half) * pred_x1_s1_fm_head.exp() + c_d_half * pred_x1_s1_sc_head.exp()
                        elif self.config.algo.shortcut_mix_type == 'residual':
                            weight = 0.5 * c_d_half * (1.0 - c_t_sc)
                            pred_x1_s1 = pred_x1_s1_fm_head.exp() + weight * pred_x1_s1_sc_head.exp()
                    
                    v_1 = (pred_x1_s1 - x_t_sc) / (1.0 - c_t_sc + 1e-5)
                    x_mid = x_t_sc + v_1 * c_d_half

                    if getattr(self.config.algo, 'shortcut_mix_logit', False):
                        # Model mixes logits internally
                        pred_x1_s2 = (self.forward_with_ema if self.config.algo.bootstrap_ema else self.forward)(
                            x_mid * tau_inverse_fm,
                            t_sc + d_half_1,
                            d_half_2,
                            use_auxiliary_head=True,
                            c_d=c_d_half,
                            c_t=c_t_sc + c_d_half,
                        )
                        pred_x1_s2 = pred_x1_s2.exp()
                    else:
                        # Mix after exp (original behavior)
                        pred_x1_s2_fm_head, pred_x1_s2_sc_head = (self.forward_with_ema if self.config.algo.bootstrap_ema else self.forward)(
                            x_mid * tau_inverse_fm,
                            t_sc + d_half_1,
                            d_half_2,
                            use_auxiliary_head=True,
                        )

                        if self.config.algo.shortcut_mix_type == 'interpolate':
                            pred_x1_s2 = (1.0 - c_d_half) * pred_x1_s2_fm_head.exp() + c_d_half * pred_x1_s2_sc_head.exp()
                        elif self.config.algo.shortcut_mix_type == 'residual':
                            weight = 0.5 * c_d_half * (1.0 - (c_t_sc+c_d_half))
                            pred_x1_s2 = pred_x1_s2_fm_head.exp() + weight * pred_x1_s2_sc_head.exp()
                        
                    v_2 = (pred_x1_s2 - x_mid) / (1.0 - (c_t_sc + c_d_half) + 1e-5)

                    v_target = (v_1 + v_2) / 2.0
                    x_boot = x_t_sc + v_target * (1.0 - c_t_sc)
                    x_boot = x_boot.detach()
                    del pred_x1_s1, pred_x1_s2, x_mid, v_1, v_2, v_target
            # Store shortcut trajectories
            x_t_final[num_fm:] = x_t_sc * tau_inverse_sc
            target_final[num_fm:] = x_boot
            t_final[num_fm:] = t_sc
            d_final[num_fm:] = d_sc

        # if num_fm > 0:
        #     f_fm = self.forward(x_t_final[:num_fm], t_final[:num_fm], d_final[:num_fm], use_auxiliary_head=False).exp()
        
        # if num_sc > 0:
        #     if getattr(self.config.algo, 'shortcut_mix_logit', False):
        #         # Model mixes logits internally
        #         f_sc = self.forward(x_t_final[num_fm:], t_final[num_fm:], d_final[num_fm:], use_auxiliary_head=True, c_d=c_d, c_t=c_t_sc)
        #         f_sc = f_sc.exp()
        #     else:
        #         # Mix after exp (original behavior)
        #         f_sc_fm_head, f_sc_sc_head = self.forward(x_t_final[num_fm:], t_final[num_fm:], d_final[num_fm:], use_auxiliary_head=True)
        #         if self.config.algo.shortcut_mix_type == 'interpolate':
        #             f_sc = (1.0 - c_d) * f_sc_fm_head.exp() + c_d * f_sc_sc_head.exp()
        #         elif self.config.algo.shortcut_mix_type == 'residual':
        #             weight = 0.5 * c_d * (1.0 - c_t_sc)
        #             f_sc = f_sc_fm_head.exp() + weight * f_sc_sc_head.exp()
        
        # f = torch.cat([f_fm, f_sc], dim=0)
        
        # if self.trainer.global_step % 1000 == 0:
        #     if t_sc[0] == 0.0:
        #         print(t_sc[0], self.tokenizer.decode(f.argmax(dim=-1)[num_fm]))

        # if self.config.algo.shortcut_loss_type == 'mse':
        #     error = target_final - f
        #     loss = (error ** 2).mean(dim=-1) * self.vocab_size
        # else: # cross-entropy
        #     loss = -(target_final * f).sum(dim=-1)
        
        f_all_h1, f_all_h2 = self.forward(x_t_final, t_final, d_final, use_auxiliary_head=True) # log prob
        
        if num_fm > 0:
            f_fm = f_all_h1[:num_fm].exp()
        else:
            f_fm = torch.empty(0, L, V, device=self.device)
            
        if num_sc > 0:
            f_sc_h1 = f_all_h1[num_fm:].detach() 
            f_sc_h2 = f_all_h2[num_fm:]
            
            if getattr(self.config.algo, 'shortcut_mix_logit', False):
                return NotImplementedError
            else:
                f_sc_h1_prob = f_sc_h1.exp()
                f_sc_h2_prob = f_sc_h2.exp()
                if self.config.algo.shortcut_mix_type == 'interpolate':
                    f_sc = (1.0 - c_d) * f_sc_h1_prob + c_d * f_sc_h2_prob
                elif self.config.algo.shortcut_mix_type == 'residual':
                    weight = 0.5 * c_d * (1.0 - c_t_sc)
                    f_sc = f_sc_h1_prob + weight * f_sc_h2_prob
        else:
            f_sc = torch.empty(0, L, V, device=self.device)
            
        f = torch.cat([f_fm, f_sc], dim=0)
        
        if self.trainer.global_step % 1000 == 0:
            if t_sc[0] == 0.0:
                print(t_sc[0], self.tokenizer.decode(f.argmax(dim=-1)[num_fm]))

        if self.config.algo.shortcut_loss_type == 'mse':
            error = target_final - f
            loss = (error ** 2).mean(dim=-1) * self.vocab_size
        else: 
            loss = -(target_final * f).sum(dim=-1)
        
        c_final = self._alpha_t_to_gamma(t_final)
        c_d_final = self._alpha_t_to_gamma(t_final + d_final) - c_final
        idx = torch.arange(B, device=self.device)
        is_fm = idx < num_fm
        is_sc = idx >= num_fm

        if num_fm > 0:
            loss_fm = loss[is_fm].mean()
            self.log('tfm_loss', loss_fm, prog_bar=True, sync_dist=True)

        if num_sc > 0:
            c_sc = c_final[is_sc]
            if self.config.algo.scale_loss:
                w_sc = (1.0 / (1.0 - c_sc + 1e-5) ** 2).view(-1, 1)
            else:
                w_sc = torch.ones_like(c_sc).view(-1, 1)
            w_sc = w_sc.to(loss.dtype)
            # Apply weight to actual loss for training
            loss[is_sc] = loss[is_sc] * w_sc
            loss_sc = loss[is_sc].mean()
            self.log('shortcut_loss', loss_sc, prog_bar=True, sync_dist=True)

        if self.trainer.is_global_zero:

            num_bins = 10
            for b in range(num_bins):
                lo = b / num_bins
                hi = (b + 1) / num_bins

                if num_fm > 0:
                    mask_fm = is_fm & (c_final >= lo) & (c_final < hi)
                    if mask_fm.any():
                        bin_loss_fm = loss[mask_fm].mean()
                        self.log(
                            f'tfm_loss_gamma_bin_{b}',
                            bin_loss_fm,
                            prog_bar=False,
                            sync_dist=False,
                        )

                if num_sc > 0:
                    mask_sc = is_sc & (c_d_final >= lo) & (c_d_final < hi)
                    
                    if mask_sc.any():
                        bin_loss_sc = loss[mask_sc].mean()
                        self.log(
                            f'shortcut_loss_d_bin_{b}', 
                            bin_loss_sc,
                            prog_bar=False,
                            sync_dist=False,
                        )
        return loss

    def nll(self, input_tokens, output_tokens, current_accumulation_step=None, train_mode=False):
        raise NotImplementedError("DOS only supports meanflow loss")

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None,
                         eps=1e-5, sample_sample=None):
        """Generate samples from the model."""
        if num_steps is None:
            num_steps = self.config.sampling.steps

        print(f" sampling step {num_steps}")
        B = num_samples
        V = self.vocab_size
        L = self.num_tokens
        device = self.device

        if self.config.algo.use_curriculum == True:
            tau_inverse = 10 ** (-self.config.sampling.tau_log10)  # during training
        else:
            tau_inverse = 100
        
        if self.config.sampling.noise_removal == 'shortcut':
            t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  # (T+1,)
            z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
            prev_x1_one_hot = None
            prev_t_gamma = None
            
            for i in range(num_steps):
                t_curr = t_vals[i]
                t_next = t_vals[i + 1]

                t_in = t_curr.expand(B)                

                if self.config.algo.scale_input:
                    scale_t = 1.0 / (1.0 - t_curr.view(-1, 1, 1) + 1e-5)
                else:
                    scale_t = 1.0
                alpha_t_in = self._gamma_to_alphat(t_in)
                c_dt = t_next.expand(B) - t_in
                c_dt = c_dt.view(-1, 1, 1)
                dt_alpha = self._gamma_to_alphat(t_next.expand(B))-self._gamma_to_alphat(t_in)

    
                if getattr(self.config.algo, 'shortcut_mix_logit', False):
                    x_1_pred = self.forward(z * tau_inverse * scale_t, alpha_t_in, dt_alpha, use_auxiliary_head=True, c_d=c_dt, c_t=t_in.view(-1, 1, 1))
                    x_1_pred_probs = x_1_pred.exp()
                else:
                    x_1_pred_fm, x_1_pred_sc = self.forward(z * tau_inverse * scale_t, alpha_t_in, dt_alpha, use_auxiliary_head=True)

                    if self.config.algo.shortcut_mix_type == 'interpolate':
                        x_1_pred_probs = (1.0 - c_dt) * x_1_pred_fm.exp() + c_dt * x_1_pred_sc.exp()
                    elif self.config.algo.shortcut_mix_type == 'residual':
                        weight = 0.5 * c_dt * (1.0 - t_in.view(-1, 1, 1))
                        x_1_pred_probs = x_1_pred_fm.exp() + weight * x_1_pred_sc.exp()
                    

                if i == 0 and num_steps != 1 and getattr(self.config.sampling, 'hard_start', False):
                    sample_idx = torch.multinomial(x_1_pred_probs.view(-1, self.vocab_size), 1)
                    sample_idx = sample_idx.view(B, L)
                    x_1_pred_one_hot = F.one_hot(sample_idx, num_classes=self.vocab_size).to(self.device).float()
                else:
                    if self.config.sampling.argmax_correction:
                        x_1_pred_argmax = x_1_pred_probs.argmax(dim=-1)
                        x_1_pred_one_hot = F.one_hot(x_1_pred_argmax, self.vocab_size).to(self.device).float()
                    else:
                        x_1_pred_one_hot = x_1_pred_probs
                
                if i == num_steps - 1:
                    print(self.tokenizer.decode(x_1_pred_one_hot.argmax(dim=-1)[0].cpu().numpy()))
                    z = x_1_pred_one_hot
                    break
                
                if getattr(self.config.sampling, 'solver', 'euler') == 'DPMv2':
                    if i == 0:
                        v = (x_1_pred_one_hot - z) / (1.0 - t_curr + 1e-5)
                        z = z + v * (t_next - t_curr)
                    else:
                        h = (t_next - t_curr)
                        h_prev = (t_curr - prev_t_gamma)
                        r = h_prev / h
                        x1_hat = (1 + 1/(2*r)) * x_1_pred_one_hot - (1/(2*r)) * prev_x1_one_hot
                        
                        v_corrected = (x1_hat - z) / (1.0 - t_curr + 1e-5)
                        z = z + v_corrected * h
                        # print(self.tokenizer.decode(x1_hat.argmax(dim=-1)[0].cpu().numpy()))
                    
                    prev_x1_one_hot = x_1_pred_one_hot
                    prev_t_gamma = t_curr

                else: # euler solver
                    v = (x_1_pred_one_hot - z) / (1.0 - t_curr + 1e-5)
                    z = z + v * (t_next - t_curr)
                    
        elif self.config.sampling.noise_removal == 'shortcut_alpha':
            t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  # (T+1,)
            z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
                
            for i in range(num_steps):
                t_curr = t_vals[i]
                t_next = t_vals[i + 1]

                t_in = t_curr.expand(B)
                gamma_t_in = self._alpha_t_to_gamma(t_in)

                c_d_in = self._alpha_t_to_gamma(t_next.expand(B)) - self._alpha_t_to_gamma(t_in)

                if self.config.algo.scale_input:
                    scale_t = 1.0 / (1.0 - gamma_t_in.view(-1, 1, 1) + 1e-5)
                else:
                    scale_t = 1.0

                x_1_pred = self.forward(z * tau_inverse * scale_t, t_in, t_next.expand(B) - t_in, use_auxiliary_head=True)

                x_1_pred_argmax = x_1_pred.argmax(dim=-1)
                x_1_pred_one_hot = F.one_hot(x_1_pred_argmax, self.vocab_size).to(self.device).float()

                print(self.tokenizer.decode(x_1_pred_argmax[0].cpu().numpy()))
                
                if getattr(self.config.sampling, 'solver', 'euler') == 'DPMv2':
                    if i == num_steps - 1:
                        z = x_1_pred_one_hot
                    elif i == 0:
                        v = (x_1_pred_one_hot - z) / (1.0 - gamma_t_in.view(-1, 1, 1) + 1e-5)
                        z = z + v * c_d_in.view(-1, 1, 1)
                    else:
                        h = t_next - t_curr
                        h_prev = t_curr - prev_t_alpha
                        r = h_prev / h
                        
                        x1_hat = (1 + 1/(2*r)) * x_1_pred_one_hot - (1/(2*r)) * prev_x1_one_hot
                        
                        v_corrected = (x1_hat - z) / (1.0 - gamma_t_in.view(-1, 1, 1) + 1e-5)
                        z = z + v_corrected * c_d_in.view(-1, 1, 1)
                    
                    prev_x1_one_hot = x_1_pred_one_hot
                    prev_t_alpha = t_curr
                
                else:
                    if i < num_steps - 1:
                        v = (x_1_pred_one_hot - z) / (1.0 - gamma_t_in.view(-1, 1, 1) + 1e-5)
                        z = z + v * c_d_in.view(-1, 1, 1)
                    else:
                        z = x_1_pred_one_hot
                        break

        elif self.config.sampling.noise_removal == 'uniform':
            t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  # (T+1,)
            z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
            for i in range(num_steps):
                
                    
                t_curr = t_vals[i]
                t_next = t_vals[i + 1]
                dt = t_next - t_curr
                t_in = t_curr.expand(B)

                if self.config.algo.scale_input:
                    scale = 1.0 / (1.0 - t_curr + 1e-5)
                else:
                    scale = 1.0
                x_1_pred = self.forward(z * tau_inverse * scale, self._gamma_to_alphat(t_in))
                

                x_1_pred_argmax = x_1_pred.argmax(dim=-1)
                x_1_pred_one_hot = F.one_hot(x_1_pred_argmax, self.vocab_size).to(self.device).float()
                

                if i < num_steps - 1:
                    v = (x_1_pred_one_hot - z) / (1.0 - t_curr)
                    z = z + dt * v
                else:
                    print(self.tokenizer.decode(x_1_pred_one_hot.argmax(dim=-1)[0].cpu().numpy()))
                    z = x_1_pred_one_hot
                    break
        elif self.config.sampling.noise_removal == 'uniform_alpha':
            t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  # (T+1,)
            z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
            for i in range(num_steps):
                  
                t_curr = t_vals[i]
                t_next = t_vals[i + 1]
                dt = t_next - t_curr
                t_in = t_curr.expand(B)
                c_dt = self._alpha_t_to_gamma(t_next.expand(B)) - self._alpha_t_to_gamma(t_in)

                if self.config.algo.scale_input:
                    scale = 1.0 / (1.0 - self._alpha_t_to_gamma(t_in)[0] + 1e-5)
                else:
                    scale = 1.0
                x_1_pred = self.forward(z * tau_inverse * scale, t_in)
                

                x_1_pred_argmax = x_1_pred.argmax(dim=-1)
                x_1_pred_one_hot = F.one_hot(x_1_pred_argmax, self.vocab_size).to(self.device).float()
                
                print(self.tokenizer.decode(x_1_pred_argmax[0].cpu().numpy()))

                if i < num_steps - 1:
                    v = (x_1_pred_one_hot - z) / (1.0 - self._alpha_t_to_gamma(t_in)[0])
                    z = z + c_dt[0] * v
                else:
                    z = x_1_pred_one_hot
                    break

        elif self.config.sampling.noise_removal == 'consistency':
            t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  # (T+1,)
            z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
            for i in range(num_steps):
            
                t_curr = t_vals[i]
                t_next = t_vals[i + 1]
                dt = t_next - t_curr
                t_in = t_curr.expand(B)
                c_d = 1.0 - t_in

                x_1_pred = self.forward(z * tau_inverse, self._gamma_to_alphat(t_in), c_d)

                x_1_pred_argmax = x_1_pred.argmax(dim=-1)
                x_1_pred_one_hot = F.one_hot(x_1_pred_argmax, self.vocab_size).to(self.device).float()
                
                print(self.tokenizer.decode(x_1_pred_argmax[0].cpu().numpy()))

                if i < num_steps - 1:
                    v = (x_1_pred_one_hot - z) / c_d[0]
                    z = z + dt * v
                else:
                    z = x_1_pred_one_hot
                    break
        
        elif self.config.sampling.noise_removal == 'consistency_alpha':
            t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  # (T+1,)
            z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
            for i in range(num_steps):
            
                t_curr = t_vals[i]
                t_next = t_vals[i + 1]
                dt = t_next - t_curr
                t_in = t_curr.expand(B)
                c_d = self._alpha_t_to_gamma(torch.ones_like(t_in)) - self._alpha_t_to_gamma(t_in)

                if self.config.algo.scale_input:
                    scale = 1.0 / (1.0 - self._alpha_t_to_gamma(t_in)[0] + 1e-5)
                else:
                    scale = 1.0
                x_1_pred = self.forward(z * tau_inverse * scale, t_in, c_d)

                x_1_pred_argmax = x_1_pred.argmax(dim=-1)
                x_1_pred_one_hot = F.one_hot(x_1_pred_argmax, self.vocab_size).to(self.device).float()
                # import ipdb; ipdb.set_trace()
                print(self.tokenizer.decode(x_1_pred_argmax[0].cpu().numpy()))
                if i < num_steps - 1:
                    v = (x_1_pred_one_hot - z) / (1.0 - self._alpha_t_to_gamma(t_in)[0])
                    z = z + (self._alpha_t_to_gamma(t_next.expand(B))-self._alpha_t_to_gamma(t_in))[0] * v
                else:
                    z = x_1_pred_one_hot
                    break

        return z.argmax(dim=-1)
    
class DOS_distill(trainer_base.TrainerBase):
    def __init__(self, config, tokenizer):
        # Ensure n_separated_blocks is -1 (no separated head) for DOS_distill
        if hasattr(config.algo, 'n_separated_blocks'):
            original_n_separated_blocks = config.algo.n_separated_blocks
            config.algo.n_separated_blocks = -1
        else:
            original_n_separated_blocks = 0
            config.algo.n_separated_blocks = -1
        
        super().__init__(config, tokenizer)
        self._validate_configuration()
        self.flow_ratio = config.algo.flow_ratio
        self.jvp_api = config.algo.jvp_api
        self.gumbel_tau_log10_start = config.algo.gumbel_tau_log10_start
        self.gumbel_tau_log10_end = config.algo.gumbel_tau_log10_end
        self.curriculum_start = config.algo.curriculum_start
        self.curriculum_end = config.algo.curriculum_end
        self.sigma_min = config.algo.sigma_min
        self.t_min = config.algo.t_min
        self.t_max = config.algo.t_max
        self.use_curriculum = config.algo.use_curriculum
        self.log_flag = False

        assert self.jvp_api in [
            'funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if self.jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif self.jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True

        self.lut_a2g, self.lut_g2a = utils.build_luts(K=self.vocab_size)
        
        self.prev_distill_step = getattr(self.config.algo, 'distill_step', 1)    
        # Teacher will be initialized in on_load_checkpoint after student is loaded
        self.teacher_model = None
        if not (hasattr(config.training, 'finetune_path') and config.training.finetune_path != ''):
            raise ValueError("DOS_distill requires config.training.finetune_path to be set for teacher model")

    def _initialize_teacher_from_checkpoint(self, checkpoint_path):
        """Initialize teacher model by copying student (backbone)."""
        print(f"[DOS_distill] Initializing teacher model from student")
    
        self.teacher_model = copy.deepcopy(self.backbone)
    
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        print("[DOS_distill] Teacher model initialized successfully")

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        new_state_dict = collections.OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            new_key = k.replace('._orig_mod.', '.')
            new_state_dict[new_key] = v

        if self.config.mode != 'sample_eval':
            if self.config.algo.double_temb and self.backbone.sigma_map_prime is not None:
                if not any(k.startswith('backbone.sigma_map_prime') for k in new_state_dict.keys()):

                    print("[INFO] Adding sigma_map_prime to state_dict (Last-Layer Zero Init)")

                    for name, param in self.backbone.sigma_map_prime.named_parameters():
                        param_key = f'backbone.sigma_map_prime.{name}'

                        if 'mlp.2' in name:
                            print(name)
                            print("zero init mlp.2")
                            zero_tensor = torch.zeros_like(param.data)
                            new_state_dict[param_key] = zero_tensor
                            param.data.copy_(zero_tensor)
                        else:
                            new_state_dict[param_key] = param.data.clone()

        checkpoint['state_dict'] = new_state_dict
        super().on_load_checkpoint(checkpoint)
        
        # Initialize teacher after student weights are loaded
        # Copy student (backbone) and load teacher weights from finetune_path checkpoint
        if self.teacher_model is None:
            self._initialize_teacher_from_checkpoint(self.config.training.finetune_path)

    def _compute_gumbel_tau_inverse(self):
        if self.config.mode == 'sample_eval':
            tau = self.gumbel_tau_log10_end
            return 10 ** (-tau)
        start = self.gumbel_tau_log10_start
        end = self.gumbel_tau_log10_end
        delta = end - start
        if self.global_step < self.curriculum_start:
            tau = start
        elif self.global_step < self.curriculum_end:
            frac = (self.global_step - self.curriculum_start) / (
                self.curriculum_end - self.curriculum_start)
            tau = start + frac * delta
        else:
            tau = end
        return 10 ** (-tau)

    def training_step(self, batch, batch_idx):
        self.log(name='gumbel_tau_log10',
                 value=1 / self._compute_gumbel_tau_inverse(),
                 on_step=True,
                 on_epoch=False,
                 sync_dist=True)
        return super().training_step(batch, batch_idx)

    def _validate_configuration(self):
        pass

    def _process_sigma(self, sigma):
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1)
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.config.algo.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _process_model_output(self, model_output, xt, sigma, cap_value = 30.0):
        del xt, sigma
        model_output = cap_value * torch.tanh(model_output / cap_value)
        return model_output.log_softmax(dim=-1)


    def _sample_t_interval(self, n, accum_step, t_min=None, t_max=None):
        if t_min is None:
            t_min = self.t_min
        
        if t_max is None:
            t_max = self.t_max
        
        if accum_step is not None:
            # During training
            batch_dim = n
            n = self.config.loader.global_batch_size
        _eps_t = torch.rand(n, device=self.device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=self.device) / n
            _eps_t = (_eps_t / n + offset) % 1
            perm = torch.randperm(n, device=self.device)
            _eps_t = _eps_t[perm]

        t = (t_max - t_min) * _eps_t + t_min
        if accum_step is not None:
            t = t.chunk(self.trainer.num_nodes)[self.trainer.node_rank]
            t = t.chunk(self.trainer.num_devices)[self.trainer.local_rank]
            t = t.chunk(self.trainer.accumulate_grad_batches)[
                accum_step]
            # corner case for the last datapoint
            t = t[:batch_dim]
        return t
    # convert discrete time schedule alpha_t to continuous time schedule gamma_t
    def _alpha_t_to_gamma(self, alpha_t):
        return utils.alpha_to_gamma(alpha_t, self.lut_a2g)

    def _gamma_to_alphat(self, gamma_t):
        return utils.gamma_to_alpha(gamma_t, self.lut_g2a)

    def corrupt_continuous(self, x0, t):
        t = t.unsqueeze(-1).unsqueeze(-1)

        target_data = F.one_hot(x0, self.vocab_size).float()
        noise = torch.randn_like(target_data, dtype=torch.float32)
        x_t = (1 - t) * noise + t * target_data
        return x_t, target_data
    
    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)
    
    def teacher_forward(self, xt, sigma, sigma_prime=None, use_auxiliary_head=False):
        if self.teacher_model is None:
            raise RuntimeError("Teacher model is not initialized!")

        sigma = self._process_sigma(sigma)
        if sigma_prime is not None:
            sigma_prime = self._process_sigma(sigma_prime)
            
        with torch.no_grad(): 
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
                model_output = self.teacher_model(xt, sigma, sigma_prime, use_auxiliary_head=use_auxiliary_head)
        
        return self._process_model_output(model_output=model_output, xt=xt, sigma=sigma)
        
    def shortcut_loss(self, x0, output_tokens,
                      current_accumulation_step=None, train_mode=False, xT=None, given_t=None, not_sampling_t=False):
        del given_t, not_sampling_t
        del output_tokens
        B = x0.shape[0]
        L = x0.shape[1]
        V = self.vocab_size
        K_max = self.config.algo.shortcut_k_max
        d_min = 1.0 / K_max
        max_power = int(math.log2(K_max))

        if self.config.algo.use_curriculum == True:
            tau_inverse_fm = 10
            tau_inverse_sc = 10**(-self.config.algo.tau_log10_shortcut)
        else:
            tau_inverse_sc = 10

        initial_distill_step = getattr(self.config.algo, 'distill_step', 1)
        iter_per_distill_step = getattr(self.config.algo, 'iter_per_distill_step', 10000)
        
        distill_step = initial_distill_step + (self.global_step // iter_per_distill_step)
        distill_step = min(distill_step, max_power)
        

        if distill_step > self.prev_distill_step and self.teacher_model is not None:
            print(f"[DOS_distill] Step {self.global_step}: Distill step increased {self.prev_distill_step} -> {distill_step}. Updating teacher...")
            self.teacher_model.load_state_dict(self.backbone.state_dict())
            self.teacher_model.eval() 
            self.prev_distill_step = distill_step
        
        self._current_distill_step = distill_step
        
        d_limit = min(1.0, (2.0 ** distill_step) * d_min)
        
        if self.config.algo.sample_d_on_grid:
            k_sc = torch.randint(1, distill_step + 1, (B,), device=self.device)
            d_sc = (2.0 ** k_sc) * d_min
            d_sc = d_sc.clamp(max=1.0)

            num_steps = (1.0 / d_sc).round()
            i = (torch.rand(B, device=self.device) * num_steps).floor()
            t_sc = i * d_sc         
        elif self.config.algo.use_continuous_shortcut:
            d_limit = min(1.0, (2.0 ** distill_step) * (1.0 / K_max))
            d_sc = torch.rand(B, device=self.device) * d_limit
            t_sc = torch.rand(B, device=self.device) * (1.0 - d_sc)
            
        if self.config.algo.shortcut_on_alpha_t:
            with torch.no_grad():
                c_t_sc = self._alpha_t_to_gamma(t_sc)
                x_t_sc, _ = self.corrupt_continuous(x0, c_t_sc)


                d_half = d_sc / 2.0
                c_t_sc_view = c_t_sc.view(-1, 1, 1)

                c_t_mid = self._alpha_t_to_gamma(t_sc + d_half).view(-1, 1, 1)
                c_t_end = self._alpha_t_to_gamma(t_sc + d_sc).view(-1, 1, 1)
                c_d_half_1 = c_t_mid - c_t_sc_view
                c_d_half_2 = c_t_end - c_t_mid
                c_d_sc = c_t_end - c_t_sc_view
                
                pred_x1_s1 = self.teacher_forward(
                    x_t_sc * tau_inverse_fm,
                    t_sc,
                    d_half,
                ).exp()
                
                v_1 = (pred_x1_s1 - x_t_sc) / (1.0 - c_t_sc_view + 1e-5)
                x_mid = x_t_sc + v_1 * c_d_half_1

                pred_x1_s2 = self.teacher_forward(
                    x_mid * tau_inverse_fm,
                    t_sc + d_half,
                    d_half,
                ).exp()
                
                    
                v_2 = (pred_x1_s2 - x_mid) / (1.0 - c_t_mid + 1e-5)

                if (c_d_sc <= 0).any(): print(t_sc, d_sc, c_t_sc, c_t_end)
                v_target = (c_d_half_1*v_1 + c_d_half_2*v_2) / c_d_sc.clamp_min(1e-6)
                x_boot = x_t_sc + v_target * (1.0 - c_t_sc_view)
                x_boot = F.one_hot(x_boot.argmax(dim=-1), self.vocab_size).float()
                x_boot = x_boot.detach()

                del pred_x1_s1, pred_x1_s2, x_mid, v_1, v_2, v_target
        else:
            with torch.no_grad():
                c_t_sc = t_sc
                x_t_sc, _ = self.corrupt_continuous(x0, c_t_sc)
                t_sc = self._gamma_to_alphat(c_t_sc)

                c_d_half = d_sc / 2.0
                c_t_sc_view = c_t_sc.view(-1, 1, 1)

                c_t_mid = c_t_sc + c_d_half.view(-1, 1, 1)
                c_t_end = c_t_sc + d_sc.view(-1, 1, 1)
                c_d_half_1 = c_t_mid - c_t_sc_view
                d_half_1 = self._gamma_to_alphat(c_t_mid) - self._gamma_to_alphat(c_t_sc)
                c_d_half_2 = c_t_end - c_t_mid
                d_half_2 = self._gamma_to_alphat(c_t_end) - self._gamma_to_alphat(c_t_mid)
                c_d_sc = c_t_end - c_t_sc_view
                d_sc = self._gamma_to_alphat(c_t_end) - self._gamma_to_alphat(c_t_sc)
                
                pred_x1_s1 = self.teacher_forward(
                    x_t_sc * tau_inverse_fm,
                    t_sc,
                    d_half_1.flatten(),
                ).exp()
                
                v_1 = (pred_x1_s1 - x_t_sc) / (1.0 - c_t_sc_view + 1e-5)
                x_mid = x_t_sc + v_1 * c_d_half_1

                pred_x1_s2 = self.teacher_forward(
                    x_mid * tau_inverse_fm,
                    t_sc + d_half_1,
                    d_half_2.flatten(),
                ).exp()
                
                    
                v_2 = (pred_x1_s2 - x_mid) / (1.0 - c_t_mid + 1e-5)


                v_target = (c_d_half_1*v_1 + c_d_half_2*v_2) / c_d_sc.clamp_min(1e-6)
                x_boot = x_t_sc + v_target * (1.0 - c_t_sc_view)
                x_boot = F.one_hot(x_boot.argmax(dim=-1), self.vocab_size).float()
                x_boot = x_boot.detach()

                del pred_x1_s1, pred_x1_s2, x_mid, v_1, v_2, v_target
        import ipdb; ipdb.set_trace()
        f = self.forward(x_t_sc * tau_inverse_fm, t_sc, d_sc.flatten(), use_auxiliary_head=True)

        if self.trainer.global_step % 50 == 0:
            print(f"x_boot: {self.tokenizer.decode(x_boot.argmax(dim=-1)[0].cpu().numpy())}\n")
            print(f"f: {self.tokenizer.decode(f.argmax(dim=-1)[0].cpu().numpy())}\n")
            print(f"t_sc: {t_sc[0]}, d_sc: {d_sc[0]}, d_sc: {d_sc[0]}\n")

        if self.config.algo.shortcut_loss_type == 'mse':
            error = x_boot - f.exp()
            loss = (error ** 2).mean(dim=-1) * self.vocab_size
        else: # cross-entropy
            loss = -(x_boot * f).sum(dim=-1)

        c_d_final = self._alpha_t_to_gamma(t_sc + d_sc) - c_t_sc
        self.log('shortcut_loss', loss.mean(), prog_bar=True, sync_dist=True)

        if self.trainer.is_global_zero:
            num_bins = 10
            for b in range(num_bins):
                lo = b / num_bins
                hi = (b + 1) / num_bins
                mask_sc = (c_d_final >= lo) & (c_d_final < hi)
                if mask_sc.any():
                    bin_loss_sc = loss[mask_sc].mean()
                    self.log(
                        f'shortcut_loss_d_bin_{b}', 
                        bin_loss_sc,
                        prog_bar=False,
                        sync_dist=False,
                    )
        return loss

    def nll(self, input_tokens, output_tokens, current_accumulation_step=None, train_mode=False):
        raise NotImplementedError("DOS only supports meanflow loss")

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None,
                         eps=1e-5, sample_sample=None):
        """Generate samples from the model."""
        if num_steps is None:
            num_steps = self.config.sampling.steps

        print(f" sampling step {num_steps}")
        B = num_samples
        V = self.vocab_size
        L = self.num_tokens
        device = self.device

        if self.config.algo.use_curriculum == True:
            tau_inverse = 10 ** (-self.config.sampling.tau_log10)  # during training
        else:
            tau_inverse = 100
        
        if self.config.sampling.noise_removal == 'shortcut':
            t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  # (T+1,)
            z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
            prev_x1_one_hot = None
            prev_t_gamma = None
            
            for i in range(num_steps):
                t_curr = t_vals[i]
                t_next = t_vals[i + 1]

                t_in = t_curr.expand(B)                

                if self.config.algo.scale_input:
                    scale_t = 1.0 / (1.0 - t_curr.view(-1, 1, 1) + 1e-5)
                else:
                    scale_t = 1.0
                alpha_t_in = self._gamma_to_alphat(t_in)
                d_alpha = self._gamma_to_alphat(t_next.expand(B))-self._gamma_to_alphat(t_in)
                
                x_1_pred = self.forward(z * tau_inverse * scale_t, alpha_t_in, d_alpha, use_auxiliary_head=True)
                x_1_pred_probs = x_1_pred.exp()

                if num_steps == 1:
                    x_1_pred_argmax = x_1_pred.argmax(dim=-1)
                    z = F.one_hot(x_1_pred_argmax, self.vocab_size).to(self.device).float()
                    print(f"Single step output: {self.tokenizer.decode(x_1_pred_argmax[0].cpu().numpy())}")
                    break 
    
                if i == 0 and num_steps != 1 and getattr(self.config.sampling, 'hard_start', False):
                    sample_idx = torch.multinomial(x_1_pred_probs.view(-1, self.vocab_size), 1)
                    sample_idx = sample_idx.view(B, L)
                    x_1_pred_one_hot = F.one_hot(sample_idx, num_classes=self.vocab_size).to(self.device).float()
                else:
                    if self.config.sampling.argmax_correction:
                        x_1_pred_argmax = x_1_pred.argmax(dim=-1)
                        x_1_pred_one_hot = F.one_hot(x_1_pred_argmax, self.vocab_size).to(self.device).float()
                    else:
                        x_1_pred_one_hot = x_1_pred_probs
                
                if i == num_steps - 1:
                    print(self.tokenizer.decode(x_1_pred_one_hot.argmax(dim=-1)[0].cpu().numpy()))
                    z = x_1_pred_one_hot
                    break
                
                if getattr(self.config.sampling, 'solver', 'euler') == 'DPMv2':
                    if i == 0:
                        v = (x_1_pred_one_hot - z) / (1.0 - t_curr + 1e-5)
                        z = z + v * (t_next - t_curr)
                    else:
                        h = (t_next - t_curr)
                        h_prev = (t_curr - prev_t_gamma)
                        r = h_prev / h
                        x1_hat = (1 + 1/(2*r)) * x_1_pred_one_hot - (1/(2*r)) * prev_x1_one_hot
                        
                        v_corrected = (x1_hat - z) / (1.0 - t_curr + 1e-5)
                        z = z + v_corrected * h
                    
                    prev_x1_one_hot = x_1_pred_one_hot
                    prev_t_gamma = t_curr

                else: # euler solver
                    v = (x_1_pred_one_hot - z) / (1.0 - t_curr + 1e-5)
                    z = z + v * (t_next - t_curr)
                    
        elif self.config.sampling.noise_removal == 'shortcut_alpha':
            t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  # (T+1,)
            z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
                
            for i in range(num_steps):
                t_curr = t_vals[i]
                t_next = t_vals[i + 1]

                t_in = t_curr.expand(B)
                gamma_t_in = self._alpha_t_to_gamma(t_in)

                c_d_in = self._alpha_t_to_gamma(t_next.expand(B)) - self._alpha_t_to_gamma(t_in)

                if self.config.algo.scale_input:
                    scale_t = 1.0 / (1.0 - gamma_t_in.view(-1, 1, 1) + 1e-5)
                else:
                    scale_t = 1.0
                print(t_in[0], t_next-t_in)
                x_1_pred = self.forward(z * tau_inverse * scale_t, t_in, t_next.expand(B) - t_in, use_auxiliary_head=True)

                x_1_pred_argmax = x_1_pred.argmax(dim=-1)
                x_1_pred_one_hot = F.one_hot(x_1_pred_argmax, self.vocab_size).to(self.device).float()

                print(self.tokenizer.decode(x_1_pred_argmax[0].cpu().numpy()))
                
                if getattr(self.config.sampling, 'solver', 'euler') == 'DPMv2':
                    if i == num_steps - 1:
                        z = x_1_pred_one_hot
                    elif i == 0:
                        v = (x_1_pred_one_hot - z) / (1.0 - gamma_t_in.view(-1, 1, 1) + 1e-5)
                        z = z + v * c_d_in.view(-1, 1, 1)
                    else:
                        h = t_next - t_curr
                        h_prev = t_curr - prev_t_alpha
                        r = h_prev / h
                        
                        x1_hat = (1 + 1/(2*r)) * x_1_pred_one_hot - (1/(2*r)) * prev_x1_one_hot
                        
                        v_corrected = (x1_hat - z) / (1.0 - gamma_t_in.view(-1, 1, 1) + 1e-5)
                        z = z + v_corrected * c_d_in.view(-1, 1, 1)
                    
                    prev_x1_one_hot = x_1_pred_one_hot
                    prev_t_alpha = t_curr
                
                else:
                    if i < num_steps - 1:
                        v = (x_1_pred_one_hot - z) / (1.0 - gamma_t_in.view(-1, 1, 1) + 1e-5)
                        z = z + v * c_d_in.view(-1, 1, 1)
                    else:
                        z = x_1_pred_one_hot
                        break

        elif self.config.sampling.noise_removal == 'uniform':
            t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  # (T+1,)
            z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
            for i in range(num_steps):
                
                    
                t_curr = t_vals[i]
                t_next = t_vals[i + 1]
                dt = t_next - t_curr
                t_in = t_curr.expand(B)

                if self.config.algo.scale_input:
                    scale = 1.0 / (1.0 - t_curr + 1e-5)
                else:
                    scale = 1.0
                x_1_pred = self.forward(z * tau_inverse * scale, self._gamma_to_alphat(t_in))
                

                x_1_pred_argmax = x_1_pred.argmax(dim=-1)
                x_1_pred_one_hot = F.one_hot(x_1_pred_argmax, self.vocab_size).to(self.device).float()
                

                if i < num_steps - 1:
                    v = (x_1_pred_one_hot - z) / (1.0 - t_curr)
                    z = z + dt * v
                else:
                    z = x_1_pred_one_hot
                    break
        elif self.config.sampling.noise_removal == 'uniform_alpha':
            t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  
            z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
            for i in range(num_steps):
                  
                t_curr = t_vals[i]
                t_next = t_vals[i + 1]
                dt = t_next - t_curr
                t_in = t_curr.expand(B)
                c_dt = self._alpha_t_to_gamma(t_next.expand(B)) - self._alpha_t_to_gamma(t_in)

                if self.config.algo.scale_input:
                    scale = 1.0 / (1.0 - self._alpha_t_to_gamma(t_in)[0] + 1e-5)
                else:
                    scale = 1.0
                x_1_pred = self.forward(z * tau_inverse * scale, t_in)
                

                x_1_pred_argmax = x_1_pred.argmax(dim=-1)
                x_1_pred_one_hot = F.one_hot(x_1_pred_argmax, self.vocab_size).to(self.device).float()
                
                print(self.tokenizer.decode(x_1_pred_argmax[0].cpu().numpy()))

                if i < num_steps - 1:
                    v = (x_1_pred_one_hot - z) / (1.0 - self._alpha_t_to_gamma(t_in)[0])
                    z = z + c_dt[0] * v
                else:
                    z = x_1_pred_one_hot
                    break

        elif self.config.sampling.noise_removal == 'consistency':
            t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  # (T+1,)
            z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
            for i in range(num_steps):
            
                t_curr = t_vals[i]
                t_next = t_vals[i + 1]
                dt = t_next - t_curr
                t_in = t_curr.expand(B)
                c_d = 1.0 - t_in

                x_1_pred = self.forward(z * tau_inverse, self._gamma_to_alphat(t_in), c_d)

                x_1_pred_argmax = x_1_pred.argmax(dim=-1)
                x_1_pred_one_hot = F.one_hot(x_1_pred_argmax, self.vocab_size).to(self.device).float()
                
                print(self.tokenizer.decode(x_1_pred_argmax[0].cpu().numpy()))

                if i < num_steps - 1:
                    v = (x_1_pred_one_hot - z) / c_d[0]
                    z = z + dt * v
                else:
                    z = x_1_pred_one_hot
                    break
        
        elif self.config.sampling.noise_removal == 'consistency_alpha':
            t_vals = torch.linspace(0.0, 1.0, num_steps + 1, device=device)  # (T+1,)
            z = torch.randn((num_samples, L, V), device=device, dtype=self.dtype)
            for i in range(num_steps):
            
                t_curr = t_vals[i]
                t_next = t_vals[i + 1]
                dt = t_next - t_curr
                t_in = t_curr.expand(B)
                c_d = self._alpha_t_to_gamma(torch.ones_like(t_in)) - self._alpha_t_to_gamma(t_in)

                if self.config.algo.scale_input:
                    scale = 1.0 / (1.0 - self._alpha_t_to_gamma(t_in)[0] + 1e-5)
                else:
                    scale = 1.0
                x_1_pred = self.forward(z * tau_inverse * scale, t_in, c_d)

                x_1_pred_argmax = x_1_pred.argmax(dim=-1)
                x_1_pred_one_hot = F.one_hot(x_1_pred_argmax, self.vocab_size).to(self.device).float()
                # import ipdb; ipdb.set_trace()
                print(self.tokenizer.decode(x_1_pred_argmax[0].cpu().numpy()))
                if i < num_steps - 1:
                    v = (x_1_pred_one_hot - z) / (1.0 - self._alpha_t_to_gamma(t_in)[0])
                    z = z + (self._alpha_t_to_gamma(t_next.expand(B))-self._alpha_t_to_gamma(t_in))[0] * v
                else:
                    z = x_1_pred_one_hot
                    break

        return z.argmax(dim=-1)
