import os
import collections
import copy
import pickle

import fsspec
import numpy as np
import torch
import torch.nn.functional as F

import trainer_base
import utils


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
    checkpoint['state_dict'] = collections.OrderedDict(
      (k, v) for k, v in checkpoint['state_dict'].items()
      if not k.startswith('teacher'))
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
                    dalpha_t, low_var=False):
    assert alpha_t.ndim == 2
    assert x0.ndim == 2
    assert xt.ndim == 2
    assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
    x_reconst = log_x_theta.exp()
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
                                - 1 / xbar_theta_xt)
    
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
                   noise_removal_step=False):
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
    return None, trainer_base.sample_categorical(q_xs)


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
      print('max:{} {}'.format(gamma_t.max(), gamma_max))
      print('min:{} {}'.format(gamma_t.min(), gamma_min))
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
    self.loss_type = self.config.algo.loss_type
    self._validate_configuration()

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.integral_cache['pt'] = self.integral_cache[
      'pt'].to(*args, **kwargs)
    self.integral_cache['grad_pt'] = self.integral_cache[
      'grad_pt'].to(*args, **kwargs)
    return self

  def _compute_gumbel_tau_inverse(self):
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
      tau = -10
    return 10 ** (-tau)

  def training_step(self, batch, batch_idx):
    self.log(name='gumbel_tau_log10',
             value=1 / self._compute_gumbel_tau_inverse(),
             on_step=True,
             on_epoch=False,
             sync_dist=True)
    return super().training_step(batch, batch_idx)

  def _gamma_to_alphat(self, gamma_t):
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
                          device=self.device)
    return alpha_t * x + sigma_t * epsilon

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=False):
    use_true_nll = (self.global_step > self.curriculum_end
                    or not train_mode)
    if use_true_nll:
      return super().nll(x0, output_tokens,
                         current_accumulation_step)
    del output_tokens
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    gamma_t = self.gamma_min + t * (self.gamma_max
                                    - self.gamma_min)    
    gamma_t_prime = self.gamma_max - self.gamma_min
    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    T = 1000
    usdm_dalpha_t = gamma_t_prime * T * (
      self._gamma_to_alphat(gamma_t + 1 / T) - usdm_alpha_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    usdm_dalpha_t = usdm_dalpha_t.unsqueeze(-1)
    assert usdm_alpha_t.ndim == 2
    sigma = self._sigma_from_alphat(usdm_alpha_t)

    x0_one_hot = F.one_hot(x0, self.vocab_size)
    xt = self._q_xt_gaussian(x0_one_hot, gamma_t)
    xt = xt * self._compute_gumbel_tau_inverse()
    xt_usdm = xt.argmax(-1)
    log_x_theta = self.forward(xt, sigma=sigma)

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
        self.linear_growth_max -  self.linear_growth_min)
    n = self.global_step // self.update_teacher_every
    return 2 ** n / self.T

  def nll(self, x0, output_tokens,
          current_accumulation_step=None, train_mode=None):
    del output_tokens, train_mode
    t = self._sample_t(x0.shape[0], current_accumulation_step)
    dt = self._compute_dt()
    t = torch.clip(t + dt, 0, 1)

    gamma_t = self.gamma_min + t * (self.gamma_max
                                    - self.gamma_min)
    gamma_s = self.gamma_min + (
      t - dt) * (self.gamma_max - self.gamma_min)

    usdm_alpha_t = self._gamma_to_alphat(gamma_t)
    usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
    assert usdm_alpha_t.ndim == 2
    usdm_alpha_s = self._gamma_to_alphat(gamma_s)
    usdm_alpha_s = usdm_alpha_s.unsqueeze(-1)
    assert usdm_alpha_s.ndim == 2

    xt, xs = self._sample_trajectory(x0, gamma_t, gamma_s)
    xt_discrete = xt.argmax(-1)
    xs_discrete = xs.argmax(-1)
    log_x_theta_student = self.forward(
      xt_discrete, sigma=self._sigma_from_alphat(usdm_alpha_t))
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

class GaussianLogitDiffusion(trainer_base.TrainerBase):
  """
  Continuous diffusion LM over one-hot logit space.

  - Data: token sequences x0 of shape [B, L] with vocab_size V.
  - We embed them as one-hot matrices X0 in R^{B x L x V}.
  - Forward process: Gaussian diffusion in logit space:
      X_t = sqrt(alpha_bar_t) * X0 + sqrt(1 - alpha_bar_t) * eps
    where alpha_bar_t = sigmoid(-gamma_t).

  - Network: backbone(X_t, sigma) predicts epsilon_theta(X_t, t)
    (same shape as X_t).

  - Training loss (our "NLL"): standard DDPM ε-prediction loss:
      E_t,eps [ || eps - eps_theta(X_t, t) ||^2 ].
  """

  def __init__(self, config, tokenizer):
    # vocab_size comes from tokenizer directly
    super().__init__(config, tokenizer,
                     vocab_size=tokenizer.vocab_size)
    # Diffusion schedule in gamma-space (like Duo)
    self.gamma_min = self.config.algo.gamma_min
    self.gamma_max = self.config.algo.gamma_max
    # Number of diffusion steps to use for sampling
    # (can share with Duo's T if you like)
    self.T = getattr(self.config.algo, 'num_diffusion_steps',
                     getattr(self.config, 'T', 1000))

    self.save_hyperparameters()
    self._validate_configuration()

  def _sample_t(self, batch_size, current_accumulation_step=None):
    """Sample continuous time t ~ Uniform(eps, 1 - eps).

    This mimics the simple usage in Duo's nll functions without
    importance sampling. If you want identical behavior to DUO_BASE,
    you can copy its _sample_t implementation instead.
    """
    del current_accumulation_step
    eps = float(self.config.training.sampling_eps)
    # shape [batch_size], on correct device
    t = torch.rand(batch_size, device=self.device)
    t = t * (1.0 - 2.0 * eps) + eps
    return t

  def _validate_configuration(self):
    # For a continuous diffusion, time conditioning should be on.
    assert self.config.algo.time_conditioning, (
      "GaussianLogitDiffusion expects time conditioning to be enabled "
      "(config.algo.time_conditioning = True).")
    # No separate prior model
    assert self.config.prior.type == 'none'

  # -------------------------
  # Utility: schedule + noise
  # -------------------------

  def _gamma_from_t(self, t):
    """
    Map continuous t in [0, 1] to gamma_t linearly.
    t: shape [B]
    returns: gamma_t shape [B]
    """
    return self.gamma_min + t * (self.gamma_max - self.gamma_min)

  def _alpha_bar_from_gamma(self, gamma_t):
    """
    alpha_bar_t = sigmoid(-gamma_t)
    sqrt(alpha_bar_t) and sqrt(1 - alpha_bar_t) are used in q(x_t | x_0).
    """
    return torch.sigmoid(-gamma_t)

  def _q_xt_gaussian(self, x0_one_hot, gamma_t):
    """
    Forward Gaussian corruption:
      X_t = sqrt(alpha_bar_t) * X0 + sqrt(1 - alpha_bar_t) * eps
    x0_one_hot: [B, L, V]
    gamma_t: [B]
    returns: X_t [B, L, V]
    """
    assert gamma_t.ndim == 1
    assert x0_one_hot.ndim == 3

    alpha_bar_t = self._alpha_bar_from_gamma(gamma_t)  # [B]
    sqrt_ab = alpha_bar_t.sqrt().view(-1, 1, 1)
    sqrt_one_minus_ab = (1.0 - alpha_bar_t).sqrt().view(-1, 1, 1)

    eps = torch.randn_like(x0_one_hot)
    x_t = sqrt_ab * x0_one_hot + sqrt_one_minus_ab * eps
    return x_t, eps

  # -------------------------
  # TrainerBase plumbing
  # -------------------------

  def _process_model_input(self, x0, valid_tokens):
    """
    For this continuous diffusion, we just use x0 directly as both
    input_tokens and output_tokens for TrainerBase bookkeeping.
    """
    # input_tokens: used for building X0 in nll
    # output_tokens: not needed, but we keep shape consistent
    return x0, x0, valid_tokens

  def _process_sigma(self, sigma):
    """
    TrainerBase sometimes calls _process_sigma before passing it
    into the backbone. For this class, we treat `sigma` as our
    time-conditioning scalar (e.g., gamma_t or t itself), so we
    just return it unchanged.
    """
    return sigma

  # -------------------------
  # Core diffusion loss (ε prediction)
  # -------------------------

  def nll(self, input_tokens, output_tokens,
          current_accumulation_step=None, train_mode=True):
    """
    Continuous diffusion "NLL": DDPM-style ε-prediction loss.

    Args:
      input_tokens: [B, L] integer tokens x0
      output_tokens: ignored
      current_accumulation_step: for multi-GPU grad accumulation
      train_mode: unused (kept for API compatibility)

    Returns:
      loss_per_token: [B, L] (summing over vocab, averaging over eps dim)
    """
    del output_tokens, train_mode

    device = self.device
    x0 = input_tokens.to(device)                  # [B, L]
    B, L = x0.shape

    # Sample continuous time t ~ U(0,1)
    t = self._sample_t(B, current_accumulation_step)  # [B], defined in TrainerBase
    gamma_t = self._gamma_from_t(t)                  # [B]

    # One-hot encode tokens
    x0_one_hot = F.one_hot(x0, self.vocab_size).to(self.dtype)

    # Forward diffusion: get x_t and the true noise eps
    x_t, eps = self._q_xt_gaussian(x0_one_hot, gamma_t)  # [B,L,V] each

    # Time-conditioning variable (you can choose gamma_t or t; here gamma_t)
    sigma = gamma_t.unsqueeze(-1)  # [B,1]

    # Predict epsilon from x_t
    # backbone should return same shape as x_t: [B, L, V]
    eps_theta = self.backbone(x_t, self._process_sigma(sigma))

    if self.config.training.loss_precision == 'float64':
      eps = eps.to(torch.float64)
      eps_theta = eps_theta.to(torch.float64)

    # Standard DDPM ε-prediction loss (per token, summed over vocab dim)
    # We use 0.5 * ||eps - eps_theta||^2 to match typical continuous diffusion.
    diff = eps_theta - eps
    mse_per_vocab = 0.5 * diff.pow(2)                  # [B, L, V]
    loss_per_token = mse_per_vocab.sum(dim=-1)         # [B, L]

    return loss_per_token

  # -------------------------
  # Sampling (DDIM-style, argmax at the end)
  # -------------------------

  @torch.no_grad()
  def generate_samples(self, num_samples, num_steps=None, **kwargs):
    """
    DDIM-like sampler in logit space, followed by argmax at the end.

    - Start from Gaussian noise X_T ~ N(0, I).
    - Integrate backwards using epsilon prediction.
    - At final step, take argmax over vocab for each position.

    Args:
      num_samples: number of sequences to generate.
      num_steps: number of diffusion steps (default: self.T).

    Returns:
      tokens: [num_samples, L] int32 tensor of generated tokens.
    """
    del kwargs
    device = self.device
    if num_steps is None:
      num_steps = self.T

    # Sequence length = num_tokens used in TrainerBase
    L = self.num_tokens
    V = self.vocab_size

    # Initialize from standard Gaussian
    x = torch.randn(num_samples, L, V, device=device, dtype=self.dtype)

    # Build a fixed time schedule t_0=0, t_T=1
    # We'll step from i = num_steps ... 1
    t_values = torch.linspace(0., 1., steps=num_steps + 1, device=device)
    gamma_values = self._gamma_from_t(t_values)              # [num_steps+1]
    alpha_bar = self._alpha_bar_from_gamma(gamma_values)     # [num_steps+1]
    sqrt_alpha_bar = alpha_bar.sqrt()
    sqrt_one_minus_alpha_bar = (1. - alpha_bar).sqrt()

    for i in range(num_steps, 0, -1):
      # Current and previous time indices
      gamma_t = gamma_values[i]                # scalar
      ab_t = alpha_bar[i]
      sqrt_ab_t = sqrt_alpha_bar[i]
      sqrt_1m_t = sqrt_one_minus_alpha_bar[i]

      # Time conditioning: broadcast gamma_t over batch
      sigma = gamma_t.expand(num_samples)      # [B]
      sigma = sigma.unsqueeze(-1)             # [B, 1]

      # Predict epsilon at time t
      eps_theta = self.backbone(x, self._process_sigma(sigma))

      # Predict x0 from x_t and eps_theta:
      # x0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)
      sqrt_ab_t_b = sqrt_ab_t.view(1, 1, 1)
      sqrt_1m_t_b = sqrt_1m_t.view(1, 1, 1)
      x0_pred = (x - sqrt_1m_t_b * eps_theta) / (sqrt_ab_t_b + 1e-8)

      if i > 1:
        # DDIM update (eta = 0) to avoid additional noise:
        # x_{t-1} = sqrt(alpha_bar_{t-1}) * x0_pred
        #           + sqrt(1 - alpha_bar_{t-1}) * eps_theta
        ab_prev = alpha_bar[i - 1]
        sqrt_ab_prev = sqrt_alpha_bar[i - 1]
        sqrt_1m_prev = sqrt_one_minus_alpha_bar[i - 1]

        sqrt_ab_prev_b = sqrt_ab_prev.view(1, 1, 1)
        sqrt_1m_prev_b = sqrt_1m_prev.view(1, 1, 1)

        x = sqrt_ab_prev_b * x0_pred + sqrt_1m_prev_b * eps_theta
      else:
        # Final step: just take predicted x0
        x = x0_pred

    # Decode by argmax over vocab dimension
    tokens = x.argmax(dim=-1)   # [num_samples, L]
    return tokens