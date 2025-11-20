#!/usr/bin/env python3
"""
Plot ParallelBench evaluation results showing speed vs quality tradeoffs.
"""

import json
import gzip
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURATION: List experiments to compare
# ============================================================================
EXPERIMENTS = ['baseline']

# Metrics to plot (choose from: 'score', 'bertscore', 'inv_bleu', 'grammar')
# Comment out or remove metrics you don't want to display

METRICS_TO_PLOT = [
    'grammar'
]

# METRICS_TO_PLOT = [
#     'score',      # Overall Score
#     'bertscore',  # BERTScore (F1)
#     'inv_bleu',   # Inverse BLEU Score
#     'grammar',    # Grammar Score
# ]

# Color scheme for experiments (consistent across all subplots)
COLORS = ['#2E86AB', '#E63946', '#06A77D', '#F18F01', '#A23B72', '#5E2B5E']

# ============================================================================
# Load results for all experiments
# ============================================================================
all_experiment_results = {}

for exp_name in EXPERIMENTS:
    config_file = f"{exp_name}_list.yaml"
    results_dir = Path(exp_name + "_list")
    
    if not Path(config_file).exists():
        print(f"Warning: {config_file} not found, skipping experiment '{exp_name}'...")
        continue
    
    # Load configurations
    with open(config_file, 'r') as f:
        configs = yaml.safe_load(f)
    
    # Extract configuration parameters and load results
    results = []
    for idx, cfg in enumerate(configs):
        result_file = results_dir / f"{idx:05d}.json.gz"
        
        if not result_file.exists():
            print(f"Warning: {result_file} not found, skipping...")
            continue
        
        with gzip.open(result_file, 'rt') as f:
            data = json.load(f)
        
        # Extract key info
        block_length = cfg['generation']['block_length']
        steps = cfg['generation']['steps']
        metrics = data['metrics']
        
        results.append({
            'config_idx': idx,
            'block_length': block_length,
            'steps': steps,
            'nfe': metrics.get('nfe_mean', 0),
            'score': metrics.get('score', 0),
            'bertscore': metrics.get('bertscore_score', 0),
            'inv_bleu': metrics.get('inv_bleu_score', 0),
            'grammar': metrics.get('grammar_score', 0),
        })
    
    # Sort by block_length for proper plotting
    results = sorted(results, key=lambda x: x['block_length'])
    all_experiment_results[exp_name] = results
    print(f"✓ Loaded {len(results)} results for experiment '{exp_name}'")

# ============================================================================
# Define metric configurations
# ============================================================================
METRIC_CONFIG = {
    'score': {
        'title': 'Overall Score vs Parallelism',
        'ylabel': 'Overall Score',
        'data_key': 'score'
    },
    'bertscore': {
        'title': 'BERTScore vs Parallelism',
        'ylabel': 'BERTScore (F1)',
        'data_key': 'bertscore'
    },
    'inv_bleu': {
        'title': 'Inv-BLEU vs Parallelism',
        'ylabel': 'Inverse BLEU Score',
        'data_key': 'inv_bleu'
    },
    'grammar': {
        'title': 'Grammar Score vs Parallelism',
        'ylabel': 'Grammar Score',
        'data_key': 'grammar'
    }
}

# ============================================================================
# Create figure with subplots
# ============================================================================
num_metrics = len(METRICS_TO_PLOT)
if num_metrics == 0:
    print("Error: No metrics selected to plot!")
    exit(1)

# Calculate subplot layout (prefer 2 columns)
ncols = min(2, num_metrics)
nrows = (num_metrics + ncols - 1) // ncols

# Make plots square: use same size for width and height per subplot
subplot_size = 7
fig, axes = plt.subplots(nrows, ncols, figsize=(subplot_size * ncols, subplot_size * nrows))
fig.suptitle('LLaDA-1.0 Parallel Decoding: Comparison Across Remasking Strategies', 
             fontsize=16, fontweight='bold')

# Make axes iterable even if there's only one subplot
if num_metrics == 1:
    axes = [axes]
else:
    axes = axes.flatten()

# Plot all experiments on each selected metric subplot
for metric_idx, metric_name in enumerate(METRICS_TO_PLOT):
    if metric_name not in METRIC_CONFIG:
        print(f"Warning: Unknown metric '{metric_name}', skipping...")
        continue
    
    metric_cfg = METRIC_CONFIG[metric_name]
    ax = axes[metric_idx]
    
    for exp_idx, (exp_name, results) in enumerate(all_experiment_results.items()):
        color = COLORS[exp_idx % len(COLORS)]
        
        # Extract data for this experiment and metric
        block_lengths = [r['block_length'] for r in results]
        metric_values = [r[metric_cfg['data_key']] for r in results]
        
        # Plot the data
        ax.plot(block_lengths, metric_values, 'o-', linewidth=2, markersize=8, 
                color=color, label=exp_name, alpha=0.8)
    
    # Configure the subplot
    ax.set_xlabel('Block Length (Parallelism)', fontsize=11)
    ax.set_ylabel(metric_cfg['ylabel'], fontsize=11)
    ax.set_title(metric_cfg['title'], fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.legend(fontsize=9, loc='best')

# Hide any unused subplots
for idx in range(num_metrics, nrows * ncols):
    axes[idx].set_visible(False)

plt.tight_layout()

# Save figure
output_file = "results_plot.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Plot saved to: {output_file}")

# ============================================================================
# Print summary tables for all experiments
# ============================================================================
for exp_name, results in all_experiment_results.items():
    print("\n" + "="*90)
    print(f"EXPERIMENT: {exp_name.upper()}")
    print("="*90)
    print(f"{'Block':>6} {'Steps':>6} {'NFE':>6} {'Speedup':>8} {'Score':>8} {'BERT':>8} {'Inv-BLEU':>10} {'Grammar':>8}")
    print("-"*90)
    
    baseline_nfe = results[0]['nfe']
    for r in results:
        speedup = baseline_nfe / r['nfe'] if r['nfe'] > 0 else float('inf')
        print(f"{r['block_length']:>6} {r['steps']:>6} {r['nfe']:>6.0f} {speedup:>7.1f}× "
              f"{r['score']:>8.2f} {r['bertscore']:>8.2f} {r['inv_bleu']:>10.2f} {r['grammar']:>8.2f}")
    
    print("-"*90)
    print(f"  • Baseline (BL=1): NFE={baseline_nfe:.0f}, Score={results[0]['score']:.2f}")
    max_speedup = baseline_nfe/results[-1]['nfe'] if results[-1]['nfe'] > 0 else float('inf')
    print(f"  • Max speedup (BL={results[-1]['block_length']}): {max_speedup:.1f}×, Score={results[-1]['score']:.2f}")
    print(f"  • Score degradation: {results[0]['score'] - results[-1]['score']:.2f} points")

print("\n" + "="*90)

plt.show()

