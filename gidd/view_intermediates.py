#!/usr/bin/env python3
"""
Simple script to view intermediate denoising steps from generated samples.
"""

import torch
import random
import sys


# ANSI color codes
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def count_masks(tokens, mask_token_id=50258):
    """Count the number of mask tokens in a sequence."""
    return (tokens == mask_token_id).sum().item()


def colorize_text(text):
    """Add color coding to special tokens in the text."""
    # Color MASK tokens (appears as [MASK] in decoded text)
    text = text.replace('[MASK]', f'{RED}[MASK]{RESET}')
    # Also handle <|endoftext|> format just in case
    text = text.replace('<|endoftext|>', f'{RED}<|endoftext|>{RESET}')
    # Color PAD tokens if present
    text = text.replace('[PAD]', f'{YELLOW}[PAD]{RESET}')
    text = text.replace('<|pad|>', f'{YELLOW}<|pad|>{RESET}')
    return text


def main():
    # Load the samples file
    samples_path = "/home/mananaga/experiments/mugidd-0.5/outputs/generations/samples_with_intermediates.pt"
    
    if len(sys.argv) > 1:
        samples_path = sys.argv[1]
    
    print(f"Loading samples from: {samples_path}")
    data = torch.load(samples_path, map_location='cpu')
    
    print(f"\nDataset info:")
    print(f"  Total samples: {data['num_samples']}")
    print(f"  Denoising steps: {data['num_denoising_steps']}")
    print(f"  Number of batches: {len(data['batches'])}")
    
    # Randomly select a batch and sample within that batch
    batch_idx = random.randint(0, len(data['batches']) - 1)
    batch = data['batches'][batch_idx]
    
    # Each batch has intermediates, get first sample from first step to determine batch size
    batch_size = len(batch[0]['decoded_texts'])
    sample_idx = random.randint(0, batch_size - 1)
    
    print(f"\nRandomly selected: Batch {batch_idx}, Sample {sample_idx}")
    print(f"=" * 80)
    
    # Print all intermediate steps for this sample
    for step_data in batch:
        step_num = step_data['step']
        time_val = step_data['time']
        tokens = step_data['tokens'][sample_idx]
        decoded_text = step_data['decoded_texts'][sample_idx]
        
        # Count masks
        num_masks = count_masks(tokens)
        mask_pct = 100 * num_masks / len(tokens)
        
        print(f"\nStep {step_num:3d} (t={time_val:.4f}) - Masks: {num_masks}/{len(tokens)} ({mask_pct:.1f}%):")
        # Colorize the text
        colored_text = colorize_text(decoded_text)
        print(f"  Text: {colored_text}")
        
        # For very first and last few steps, also show tokens
        if step_num == 0 or step_num >= data['num_denoising_steps'] - 5:
            print(f"  Tokens (first 30): {tokens[:30].tolist()}")
    
    print(f"\n" + "=" * 80)
    print(f"\nFinal generated text:")
    final_text = batch[-1]['decoded_texts'][sample_idx]
    final_tokens = batch[-1]['tokens'][sample_idx]
    print(f"Length: {len(final_tokens)} tokens")
    print(f"Remaining masks: {count_masks(final_tokens)}")
    print(f"\n{colorize_text(final_text)}")


if __name__ == "__main__":
    main()

