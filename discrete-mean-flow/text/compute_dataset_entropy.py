"""
Script to compute entropy of a dataset.
Computes token-level (unigram) entropy and optionally n-gram entropy.
"""
import os
import argparse
import numpy as np
import torch
from collections import Counter, defaultdict
from tqdm import tqdm
import hydra
import omegaconf
from omegaconf import DictConfig
import dataloader
import utils

omegaconf.OmegaConf.register_new_resolver(
    'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
    'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)


def compute_entropy(probs):
    """Compute entropy from probability distribution."""
    probs = np.array(probs)
    probs = probs[probs > 0]  # Remove zeros
    return -np.sum(probs * np.log2(probs))


def compute_sample_entropy_like_metrics(dataset, tokenizer, max_samples=None):
    """
    Compute entropy in the same way as metrics.record_entropy():
    For each sample, compute entropy of token distribution within that sample,
    then average across all samples.
    This matches the 'Sample entropy' metric used for generated samples.
    """
    sample_entropies = []
    total_samples = 0
    
    print("Computing sample entropy (like metrics.record_entropy)...")
    try:
        total = min(len(dataset), max_samples) if max_samples else len(dataset)
    except (TypeError, AttributeError):
        total = max_samples if max_samples else None
    
    for i, example in enumerate(tqdm(dataset, total=total)):
        if max_samples is not None and i >= max_samples:
            break
        
        input_ids = example['input_ids']
        if isinstance(input_ids, torch.Tensor):
            sample_tensor = input_ids.flatten()
        elif isinstance(input_ids, list):
            sample_tensor = torch.tensor(input_ids).flatten()
        else:
            sample_tensor = torch.tensor(input_ids).flatten()
        
        # Compute entropy for this sample (same as metrics.record_entropy)
        _, counts = torch.unique(sample_tensor, return_counts=True, sorted=False)
        entropy = torch.special.entr(counts.float() / counts.sum()).sum().item()
        sample_entropies.append(entropy)
        total_samples += 1
    
    avg_entropy = np.mean(sample_entropies) if sample_entropies else 0.0
    return avg_entropy, sample_entropies, total_samples


def compute_unigram_entropy(dataset, tokenizer, max_samples=None):
    """Compute unigram (token-level) entropy over entire dataset."""
    token_counts = Counter()
    total_tokens = 0
    
    print("Computing unigram entropy (over entire dataset)...")
    try:
        total = min(len(dataset), max_samples) if max_samples else len(dataset)
    except (TypeError, AttributeError):
        total = max_samples if max_samples else None
    for i, example in enumerate(tqdm(dataset, total=total)):
        if max_samples is not None and i >= max_samples:
            break
        
        input_ids = example['input_ids']
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy()
        elif isinstance(input_ids, list):
            input_ids = np.array(input_ids)
        
        # Flatten and count tokens
        tokens = input_ids.flatten()
        token_counts.update(tokens.tolist())
        total_tokens += len(tokens)
    
    # Compute probabilities and entropy
    probs = np.array([count / total_tokens for count in token_counts.values()])
    entropy = compute_entropy(probs)
    
    return entropy, token_counts, total_tokens


def compute_ngram_entropy(dataset, tokenizer, n=2, max_samples=None):
    """Compute n-gram entropy."""
    ngram_counts = Counter()
    total_ngrams = 0
    
    print(f"Computing {n}-gram entropy...")
    try:
        total = min(len(dataset), max_samples) if max_samples else len(dataset)
    except (TypeError, AttributeError):
        total = max_samples if max_samples else None
    for i, example in enumerate(tqdm(dataset, total=total)):
        if max_samples is not None and i >= max_samples:
            break
        
        input_ids = example['input_ids']
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy()
        elif isinstance(input_ids, list):
            input_ids = np.array(input_ids)
        
        # Extract n-grams
        tokens = input_ids.flatten().tolist()
        for j in range(len(tokens) - n + 1):
            ngram = tuple(tokens[j:j+n])
            ngram_counts[ngram] += 1
            total_ngrams += 1
    
    # Compute probabilities and entropy
    probs = np.array([count / total_ngrams for count in ngram_counts.values()])
    entropy = compute_entropy(probs)
    
    return entropy, ngram_counts, total_ngrams


def compute_conditional_entropy(dataset, tokenizer, max_samples=None):
    """Compute conditional entropy H(X_t | X_{t-1})."""
    bigram_counts = defaultdict(Counter)
    unigram_counts = Counter()
    total_bigrams = 0
    
    print("Computing conditional entropy H(X_t | X_{t-1})...")
    try:
        total = min(len(dataset), max_samples) if max_samples else len(dataset)
    except (TypeError, AttributeError):
        total = max_samples if max_samples else None
    for i, example in enumerate(tqdm(dataset, total=total)):
        if max_samples is not None and i >= max_samples:
            break
        
        input_ids = example['input_ids']
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy()
        elif isinstance(input_ids, list):
            input_ids = np.array(input_ids)
        
        tokens = input_ids.flatten().tolist()
        for j in range(len(tokens) - 1):
            prev_token = tokens[j]
            curr_token = tokens[j + 1]
            bigram_counts[prev_token][curr_token] += 1
            unigram_counts[prev_token] += 1
            total_bigrams += 1
    
    # Compute conditional entropy: H(X|Y) = sum_y P(y) * H(X|Y=y)
    conditional_entropy = 0.0
    for prev_token, count in unigram_counts.items():
        p_prev = count / total_bigrams
        next_token_counts = bigram_counts[prev_token]
        probs = np.array([c / count for c in next_token_counts.values()])
        h_cond = compute_entropy(probs)
        conditional_entropy += p_prev * h_cond
    
    return conditional_entropy, bigram_counts, unigram_counts, total_bigrams


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    # Get tokenizer
    tokenizer = dataloader.get_tokenizer(config)
    vocab_size = len(tokenizer)
    
    print(f"Tokenizer: {config.data.tokenizer_name_or_path}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Cache directory: {config.data.cache_dir}")
    
    # Use existing get_dataloaders function (same as main.py)
    # This will use config.data.train from the yaml file (e.g., 'lm1b' from lm1b-wrap.yaml)
    train_loader, _ = dataloader.get_dataloaders(
        config, tokenizer, skip_valid=True
    )
    dataset = train_loader.dataset
    
    print(f"Dataset size: {len(dataset)}")
    
    # Use 10% of the dataset by default
    dataset_size = len(dataset)
    max_samples = getattr(config, 'max_samples', None)
    if max_samples is None:
        # Default to 10% of the dataset
        max_samples = max(1, int(dataset_size * 0.1))
        print(f"Using 10% of dataset: {max_samples} samples (out of {dataset_size} total)")
    else:
        print(f"Computing entropy on first {max_samples} samples (out of {dataset_size} total)")
    
    # Compute sample entropy (same way as metrics.record_entropy for generated samples)
    sample_entropy, sample_entropies, total_samples = compute_sample_entropy_like_metrics(
        dataset, tokenizer, max_samples=max_samples
    )
    
    # Compute unigram entropy (over entire dataset)
    unigram_entropy, token_counts, total_tokens = compute_unigram_entropy(
        dataset, tokenizer, max_samples=max_samples
    )
    
    print("\n" + "="*60)
    print("ENTROPY RESULTS")
    print("="*60)
    print(f"\n[Sample Entropy - same as generated samples]")
    print(f"  Average sample entropy: {sample_entropy:.4f} bits")
    print(f"  Number of samples: {total_samples:,}")
    print(f"  Std of sample entropies: {np.std(sample_entropies):.4f} bits")
    
    print(f"\n[Dataset-wide Unigram Entropy]")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Unique tokens: {len(token_counts):,}")
    print(f"  Unigram entropy: {unigram_entropy:.4f} bits")
    print(f"  Max possible entropy (uniform): {np.log2(vocab_size):.4f} bits")
    print(f"  Entropy ratio: {unigram_entropy / np.log2(vocab_size):.4f}")
    
    # Compute bigram entropy
    bigram_entropy, bigram_counts, total_bigrams = compute_ngram_entropy(
        dataset, tokenizer, n=2, max_samples=max_samples
    )
    print(f"\nBigram entropy: {bigram_entropy:.4f} bits")
    
    # Compute conditional entropy
    cond_entropy, _, _, _ = compute_conditional_entropy(
        dataset, tokenizer, max_samples=max_samples
    )
    print(f"Conditional entropy H(X_t | X_{{t-1}}): {cond_entropy:.4f} bits")
    
    # Additional statistics
    print("\n" + "="*60)
    print("ADDITIONAL STATISTICS")
    print("="*60)
    
    # Most common tokens
    print("\nTop 10 most frequent tokens:")
    for token_id, count in token_counts.most_common(10):
        token_str = tokenizer.decode([token_id]) if token_id < vocab_size else f"<{token_id}>"
        prob = count / total_tokens
        print(f"  {token_str:20s} (id={token_id:5d}): {count:10,} ({prob*100:.4f}%)")
    
    # Token distribution statistics
    probs = np.array([count / total_tokens for count in token_counts.values()])
    print(f"\nToken probability statistics:")
    print(f"  Mean: {np.mean(probs):.6f}")
    print(f"  Std:  {np.std(probs):.6f}")
    print(f"  Min:  {np.min(probs):.6f}")
    print(f"  Max:  {np.max(probs):.6f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()

