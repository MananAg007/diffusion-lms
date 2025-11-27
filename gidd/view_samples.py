#!/usr/bin/env python3
"""Simple script to view generated samples from samples.pt file."""

import torch
from transformers import AutoTokenizer

# Paths
SAMPLES_PATH = "/project/flame/mananaga/gidd/outputs/samples.pt"
CHECKPOINT_PATH = "/project/flame/mananaga/gidd/outputs/latest"

# Load samples
print(f"Loading samples from {SAMPLES_PATH}...")
samples = torch.load(SAMPLES_PATH, map_location="cpu", weights_only=True)
print(f"Loaded {samples.shape[0]} samples with shape {samples.shape}")
print()

# Load tokenizer
print(f"Loading tokenizer from {CHECKPOINT_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
print()

# Decode and display samples
num_to_show = min(10, len(samples))
print(f"Showing first {num_to_show} samples:")
print("=" * 80)

for i in range(num_to_show):
    text = tokenizer.decode(samples[i], skip_special_tokens=True)
    print(f"\nSample {i+1}:")
    print("-" * 80)
    print(text)
    print("-" * 80)

print()
print(f"Total samples generated: {len(samples)}")

