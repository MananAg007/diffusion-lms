#!/usr/bin/env python3
"""
Plot generative PPL metrics across training steps.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Hardcoded metrics directory
METRICS_DIR = Path.home() / "experiments/gidd-baseline/outputs/generations/metrics"

# Maximum PPL value for plot y-axis
MAX_PPL = 5000

def load_metrics(metrics_dir):
    """Load all metrics JSON files from directory."""
    metrics_files = sorted(metrics_dir.glob("samples_step_*_metrics.json"))
    
    data = []
    for metrics_file in metrics_files:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            
            # Extract step number from filename
            step = int(metrics_file.stem.replace("samples_step_", "").replace("_metrics", ""))
            metrics['step'] = step
            data.append(metrics)
    
    # Sort by step
    data.sort(key=lambda x: x['step'])
    return data

def plot_metrics(data):
    """Create subplots for different metrics."""
    steps = [d['step'] for d in data]
    
    # Check if corpus PPL is available
    has_corpus_ppl = 'total_ppl' in data[0]
    
    if has_corpus_ppl:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Generative Perplexity over Training Steps', fontsize=16, fontweight='bold')
        
        # Plot: Corpus-level PPL
        total_ppl = [d['total_ppl'] for d in data]
        ax.plot(steps, total_ppl, marker='o', linewidth=2, markersize=8, 
                linestyle='-', color='tab:blue')
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Perplexity', fontsize=12)
        ax.set_title('Corpus-Level Perplexity', fontsize=14, fontweight='bold')
        
        # Auto-scale y-axis with margin
        min_ppl = min(total_ppl)
        max_ppl = max(total_ppl)
        margin = (max_ppl - min_ppl) * 0.1  # 10% margin
        ax.set_ylim(max(0, min_ppl - margin), max_ppl + margin)
        ax.grid(True, alpha=0.3)
    else:
        # Fallback for backwards compatibility
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Generative Perplexity over Training Steps', fontsize=16, fontweight='bold')
        
        ppl_mean = [d.get('ppl_mean', d.get('ppl', 0)) for d in data]
        
        ax.plot(steps, ppl_mean, marker='o', linewidth=2, markersize=8, linestyle='-')
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Perplexity', fontsize=12)
        ax.set_title('Perplexity', fontsize=14, fontweight='bold')
        
        # Auto-scale y-axis with margin
        min_ppl = min(ppl_mean)
        max_ppl = max(ppl_mean)
        margin = (max_ppl - min_ppl) * 0.1  # 10% margin
        ax.set_ylim(max(0, min_ppl - margin), max_ppl + margin)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = METRICS_DIR / "metrics_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()

def print_summary(data):
    """Print summary statistics."""
    print("\n=== Summary Statistics ===")
    print(f"Number of checkpoints evaluated: {len(data)}")
    print(f"Step range: {data[0]['step']} to {data[-1]['step']}")
    
    # Use ppl_mean if available, otherwise fall back to ppl
    ppl_key = 'ppl_mean' if 'ppl_mean' in data[0] else 'ppl'
    
    # Find best checkpoint
    best_ppl_idx = min(range(len(data)), key=lambda i: data[i][ppl_key])
    best_checkpoint = data[best_ppl_idx]
    
    print(f"\nBest checkpoint (lowest PPL):")
    print(f"  Step: {best_checkpoint['step']}")
    if 'ppl_mean' in best_checkpoint:
        print(f"  PPL (mean): {best_checkpoint['ppl_mean']:.2f} ± {best_checkpoint['ppl_std']:.2f}")
        print(f"  PPL (corpus): {best_checkpoint['total_ppl']:.2f}")
    else:
        print(f"  PPL: {best_checkpoint['ppl']:.2f}")
    print(f"  Avg NLL: {best_checkpoint['avg_nll']:.4f}")
    print(f"  Accuracy: {best_checkpoint['acc']:.4f}")
    
    # Find worst checkpoint
    worst_ppl_idx = max(range(len(data)), key=lambda i: data[i][ppl_key])
    worst_checkpoint = data[worst_ppl_idx]
    
    print(f"\nWorst checkpoint (highest PPL):")
    print(f"  Step: {worst_checkpoint['step']}")
    if 'ppl_mean' in worst_checkpoint:
        print(f"  PPL (mean): {worst_checkpoint['ppl_mean']:.2f} ± {worst_checkpoint['ppl_std']:.2f}")
        print(f"  PPL (corpus): {worst_checkpoint['total_ppl']:.2f}")
    else:
        print(f"  PPL: {worst_checkpoint['ppl']:.2f}")
    print(f"  Avg NLL: {worst_checkpoint['avg_nll']:.4f}")
    print(f"  Accuracy: {worst_checkpoint['acc']:.4f}")

def main():
    print(f"Loading metrics from: {METRICS_DIR}")
    
    if not METRICS_DIR.exists():
        print(f"Error: Directory not found: {METRICS_DIR}")
        return
    
    data = load_metrics(METRICS_DIR)
    
    if not data:
        print("No metrics files found!")
        return
    
    print(f"Loaded {len(data)} checkpoints")
    
    print_summary(data)
    plot_metrics(data)

if __name__ == "__main__":
    main()

