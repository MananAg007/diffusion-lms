#!/usr/bin/env python3
"""
Plot KL divergence by position for different horizons (compared to horizon 1).

For each position and each sample:
- Compare horizon 2 vs horizon 1
- Compare horizon 4 vs horizon 1  
- Compare horizon 8 vs horizon 1
Then average across samples.

Usage: python plot_kl_by_horizon.py configs/baseline_list/00000.json.gz
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Import our helper functions
from utils.load_saved_logits import load_logits_data, calculate_kl_divergence


def compute_kl_by_position_and_horizon(logits_data, max_position=64):
    """
    Compute KL divergence for each position and horizon compared to horizon 1.
    
    Returns:
        dict: {horizon: {position: [kl_values_per_sample]}}
    """
    # Organize data by sample, position, and horizon
    data_map = defaultdict(lambda: defaultdict(dict))
    
    for entry in logits_data:
        sample_idx = entry['sample_idx']
        position = entry['relative_position']
        horizon = entry['horizon']
        logits = entry['logits']
        
        data_map[sample_idx][position][horizon] = logits
    
    # Compute KL divergence for each horizon vs horizon 1
    results = {2: {}, 4: {}, 8: {}}
    
    for position in range(max_position):
        for horizon in [2, 4, 8]:
            kl_values = []
            
            for sample_idx in data_map.keys():
                sample_data = data_map[sample_idx]
                
                if position not in sample_data:
                    continue
                
                pos_data = sample_data[position]
                
                # Need both horizon X and horizon 1 for this position
                if horizon in pos_data and 1 in pos_data:
                    logits_horizon = pos_data[horizon]
                    logits_horizon1 = pos_data[1]
                    
                    kl = calculate_kl_divergence(logits_horizon, logits_horizon1)
                    kl_values.append(kl)
            
            if kl_values:
                results[horizon][position] = kl_values
    
    return results


def plot_kl_divergence(results, output_file=None):
    """
    Plot KL divergence by position for different horizons.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {2: '#2ecc71', 4: '#3498db', 8: '#e74c3c'}
    labels = {2: 'Horizon 2 vs 1', 4: 'Horizon 4 vs 1', 8: 'Horizon 8 vs 1'}
    
    for horizon in [2, 4, 8]:
        positions = sorted(results[horizon].keys())
        
        if not positions:
            continue
        
        means = []
        stds = []
        
        for pos in positions:
            kl_values = results[horizon][pos]
            means.append(np.mean(kl_values))
            stds.append(np.std(kl_values))
        
        means = np.array(means)
        stds = np.array(stds)
        
        # Plot mean line
        ax.plot(positions, means, label=labels[horizon], 
                color=colors[horizon], linewidth=2, marker='o', markersize=3)
        
        # Plot confidence band (mean ± std)
        ax.fill_between(positions, means - stds, means + stds, 
                        alpha=0.2, color=colors[horizon])
    
    ax.set_xlabel('Position in Generation', fontsize=12)
    ax.set_ylabel('KL Divergence (compared to Horizon 1)', fontsize=12)
    ax.set_title('Prediction Divergence by Position and Horizon\n(How predictions change as we get closer to the position)', 
                 fontsize=14, pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(max(results[h].keys()) for h in [2,4,8] if results[h]))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_file}")
    else:
        plt.show()
    
    return fig, ax


def print_statistics(results):
    """Print statistics about the KL divergence."""
    print("\n" + "="*70)
    print("KL DIVERGENCE STATISTICS BY HORIZON")
    print("="*70)
    
    for horizon in [2, 4, 8]:
        print(f"\nHorizon {horizon} vs Horizon 1:")
        print("-" * 70)
        
        if not results[horizon]:
            print("  No data available")
            continue
        
        positions = sorted(results[horizon].keys())
        print(f"  Positions analyzed: {min(positions)} to {max(positions)} ({len(positions)} total)")
        
        # Compute overall statistics
        all_kl_values = []
        for pos in positions:
            all_kl_values.extend(results[horizon][pos])
        
        if all_kl_values:
            print(f"  Overall mean KL: {np.mean(all_kl_values):.4f}")
            print(f"  Overall std KL:  {np.std(all_kl_values):.4f}")
            print(f"  Overall median KL: {np.median(all_kl_values):.4f}")
            print(f"  Overall max KL: {np.max(all_kl_values):.4f}")
            print(f"  Total comparisons: {len(all_kl_values)}")
        
        # Show a few example positions
        print(f"\n  Example positions:")
        for pos in positions[:5]:
            kl_values = results[horizon][pos]
            print(f"    Position {pos:2d}: mean={np.mean(kl_values):.4f}, "
                  f"std={np.std(kl_values):.4f}, n={len(kl_values)}")
    
    print("="*70 + "\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_kl_by_horizon.py <results_file.json.gz> [output.png]")
        print("\nExample:")
        print("  python plot_kl_by_horizon.py configs/baseline_list/00000.json.gz")
        print("  python plot_kl_by_horizon.py configs/baseline_list/00000.json.gz kl_plot.png")
        sys.exit(1)
    
    results_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"Loading logits data from: {results_file}")
    logits_data = load_logits_data(results_file)
    
    if not logits_data:
        print("No logits data found!")
        sys.exit(1)
    
    # Get number of samples
    num_samples = len(set(entry['sample_idx'] for entry in logits_data))
    print(f"Found {num_samples} samples")
    print(f"Total logits entries: {len(logits_data)}")
    
    print("\nComputing KL divergence by position and horizon...")
    results = compute_kl_by_position_and_horizon(logits_data)
    
    # Print statistics
    print_statistics(results)
    
    # Plot
    print("Generating plot...")
    plot_kl_divergence(results, output_file)
    
    if not output_file:
        print("\nPlot displayed. Close the window to exit.")


if __name__ == "__main__":
    main()

