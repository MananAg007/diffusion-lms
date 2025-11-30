import hydra
import tqdm
import torch

from gidd.utils import parse_dtype, sample_categorical
from gidd.checkpoints import load_checkpoint
from gidd.sampling import get_sampler


def generate_with_intermediates(sampler, num_samples, num_denoising_steps, max_length, device, dtype, tokenizer):
    """Generate samples and decode all intermediate denoising steps."""
    # Get the noise schedule and model
    noise_schedule = sampler.noise_schedule
    model = sampler.model
    t_eps = sampler.t_eps
    
    # Initialize time steps
    ts = torch.linspace(0, 1, num_denoising_steps + 1, device=device).unsqueeze(-1)
    ts = (1 - 2 * t_eps) * ts + t_eps
    
    # Sample from prior
    z_t = noise_schedule.sample_prior((num_samples, max_length)).to(device, non_blocking=True)
    
    # Store all intermediate states
    all_intermediates = []
    
    # Decode initial state (pure noise) - keep special tokens to see masks
    initial_decoded = tokenizer.batch_decode(z_t.cpu(), skip_special_tokens=False)
    all_intermediates.append({
        'step': 0,
        'time': ts[num_denoising_steps - 1].item(),
        'tokens': z_t.cpu().clone(),
        'decoded_texts': initial_decoded
    })
    
    # Denoising loop
    for i in tqdm.trange(num_denoising_steps - 1, -1, -1, desc="Denoising", dynamic_ncols=True):
        with torch.autocast(device.type, dtype=dtype):
            z_t = sampler.sampling_step(z_t, ts[i], ts[max(0, i-1)]).clone()
        
        # Decode current state - keep special tokens to see masks
        decoded_texts = tokenizer.batch_decode(z_t.cpu(), skip_special_tokens=False)
        
        # Store intermediate result
        all_intermediates.append({
            'step': num_denoising_steps - i,
            'time': ts[max(0, i-1)].item(),
            'tokens': z_t.cpu().clone(),
            'decoded_texts': decoded_texts
        })
    
    return all_intermediates


@hydra.main(config_path="../configs", config_name="generate", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    print(f"Generating {args.num_samples} samples from {args.path}")

    ckpt_path = hydra.utils.to_absolute_path(args.path)

    model, noise_schedule, tokenizer, config = load_checkpoint(ckpt_path, device=device)
    model.eval()
    config.training.eval_batch_size = args.batch_size
    dtype = parse_dtype(config.training.dtype)

    sampler = get_sampler(config, model, tokenizer, noise_schedule, min_p=args.min_p)
    model.eval()

    # Check if we should decode intermediates
    decode_intermediates = getattr(args, 'decode_intermediates', False)
    
    if decode_intermediates:
        print(f"Generating with intermediate decoding (batch_size={args.batch_size})")
        all_batches = []
        
        max_length = config.model.max_seq_len
        with tqdm.tqdm(total=args.num_samples, desc="Sampling batches", dynamic_ncols=True) as pbar:
            with torch.no_grad():
                for i in range(0, args.num_samples, args.batch_size):
                    bs = min(args.batch_size, args.num_samples - i)
                    intermediates = generate_with_intermediates(
                        sampler, bs, args.num_denoising_steps, max_length, device, dtype, tokenizer
                    )
                    all_batches.append(intermediates)
                    pbar.update(bs)
        
        # Save all intermediates
        output_data = {
            'num_samples': args.num_samples,
            'num_denoising_steps': args.num_denoising_steps,
            'batches': all_batches
        }
        torch.save(output_data, hydra.utils.to_absolute_path(args.samples_path))
        print(f"Saved samples with {args.num_denoising_steps + 1} intermediate steps per sample")
    else:
        # Original behavior: just save final tokens
        samples = []
        max_length = config.model.max_seq_len
        with tqdm.tqdm(total=args.num_samples, desc="Sampling", dynamic_ncols=True) as pbar:
            with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
                for i in range(0, args.num_samples, args.batch_size):
                    bs = min(args.batch_size, args.num_samples - i)
                    z_t = sampler.generate(bs, args.num_denoising_steps, max_length=max_length, decode=False, show_progress=False)
                    samples.append(z_t)
                    pbar.update(bs)
        samples = torch.cat(samples, dim=0).cpu()
        torch.save(samples, hydra.utils.to_absolute_path(args.samples_path))


if __name__ == "__main__":
    main()
