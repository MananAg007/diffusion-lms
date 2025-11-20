#!/usr/bin/env python3
"""
Plot metrics for experiments comparing different configurations.

This script reads results from multiple experiments and creates visualizations
showing how metrics vary with decoding rate (block length) for each task.

Usage:
    python plot_metrics.py --experiments baseline modified --results_dir results --output_dir visualizations

Example:
    python plot_metrics.py --experiments baseline modified
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict


def load_experiment_results(experiment_name: str, results_dir: Path) -> pd.DataFrame:
    """
    Load results for a single experiment from JSONL file.

    Args:
        experiment_name: Name of the experiment (e.g., 'baseline')
        results_dir: Directory containing result files

    Returns:
        DataFrame with experiment results
    """
    results_file = results_dir / f"{experiment_name}.jsonl"

    if not results_file.exists():
        raise ValueError(f"Results file not found: {results_file}")

    # Read JSONL file
    results = []
    with open(results_file, "r") as f:
        for line in f:
            results.append(json.loads(line))

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Expand metrics column into separate columns
    if "metrics" in df.columns:
        metrics_df = pd.json_normalize(df["metrics"])
        df = pd.concat([df.drop("metrics", axis=1), metrics_df], axis=1)

    # Add experiment name
    df["experiment"] = experiment_name

    return df


def prepare_plot_data(experiments_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Prepare data for plotting by grouping by task and block_length.

    Args:
        experiments_data: Dictionary mapping experiment names to DataFrames

    Returns:
        Dictionary mapping task names to combined DataFrames
    """
    # Combine all experiments
    all_data = pd.concat(experiments_data.values(), ignore_index=True)

    # Group by task
    task_data = {}
    for task in all_data["task"].unique():
        task_df = all_data[all_data["task"] == task].copy()

        # Sort by block_length for proper plotting
        task_df = task_df.sort_values(["experiment", "block_length"])

        task_data[task] = task_df

    return task_data


def plot_task_metrics(
    task_name: str,
    task_df: pd.DataFrame,
    output_dir: Path,
    metric_columns: List[str] = None
):
    """
    Create subplot figure for a single task showing all metrics.

    Args:
        task_name: Name of the task
        task_df: DataFrame with task results
        output_dir: Directory to save plots
        metric_columns: List of metric column names to plot (None = auto-detect)
    """
    # Set style
    sns.set_style("whitegrid")
    # Use primary colors for different experiments
    primary_colors = ['#0066CC', '#DC3545', '#28A745', '#FFC107', '#6F42C1', '#FD7E14']
    sns.set_palette(primary_colors)

    # Auto-detect metrics for this specific task if not provided
    if metric_columns is None:
        metric_columns = [
            col for col in task_df.columns
            if any(pattern in col.lower() for pattern in [
                "score", "accuracy", "f1", "precision", "recall",
                "rouge", "bleu", "bertscore", "grammar", "inv_bleu"
            ]) and col not in ["metrics_computed"]
        ]
        metric_columns = sorted(metric_columns)

    # Filter to only metrics that exist and have non-null values for this task
    available_metrics = []
    for col in metric_columns:
        if col in task_df.columns and task_df[col].notna().any():
            available_metrics.append(col)

    if not available_metrics:
        print(f"  ⚠ Warning: No metrics found for task {task_name}")
        return

    # Create subplots
    n_metrics = len(available_metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # Make axes indexable even for single subplot
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Get unique experiments and block lengths
    experiments = sorted(task_df["experiment"].unique())
    block_lengths = sorted(task_df["block_length"].unique())

    # Create mapping from block_length to decoding rate label (2^x)
    # Assuming block_length follows powers of 2: 1, 2, 4, 8, 16, 32, 64
    decoding_rate_labels = {bl: f"$2^{{{int(np.log2(bl))}}}$" for bl in block_lengths}

    # Plot each metric
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]

        # Plot each experiment
        for exp_name in experiments:
            exp_data = task_df[task_df["experiment"] == exp_name]

            # Group by block_length and compute mean
            grouped = exp_data.groupby("block_length")[metric].mean().reset_index()

            # Plot with markers
            ax.plot(
                grouped["block_length"],
                grouped[metric],
                'o-',
                linewidth=2,
                markersize=8,
                label=exp_name,
                alpha=0.8
            )

        # Customize subplot
        ax.set_xlabel("Decoding Rate", fontsize=11)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
        ax.set_title(f"{metric.replace('_', ' ').title()}", fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

        # Set x-axis to log scale and use custom labels
        ax.set_xscale('log', base=2)
        ax.set_xticks(block_lengths)
        ax.set_xticklabels([decoding_rate_labels[bl] for bl in block_lengths])

        # Set y-axis limits dynamically with 5% margin
        all_values = []
        for exp_name_inner in experiments:
            exp_data_inner = task_df[task_df["experiment"] == exp_name_inner]
            grouped_inner = exp_data_inner.groupby("block_length")[metric].mean()
            all_values.extend(grouped_inner.values)

        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            y_range = y_max - y_min
            margin = y_range * 0.1 if y_range > 0 else 5
            ax.set_ylim(y_min - margin, y_max + margin)

    # Remove extra subplots
    for idx in range(n_metrics, len(axes)):
        fig.delaxes(axes[idx])

    # Add main title
    fig.suptitle(
        f"Task: {task_name}",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean task name for filename
    safe_task_name = task_name.replace("/", "_").replace(" ", "_")
    output_file = output_dir / f"{safe_task_name}.png"

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {output_file}")


def plot_combined_metric(
    metric_name: str,
    task_data: Dict[str, pd.DataFrame],
    output_dir: Path
):
    """
    Create a combined plot showing one metric across all tasks.

    Args:
        metric_name: Name of the metric to plot (e.g., 'score', 'grammar_score')
        task_data: Dictionary mapping task names to DataFrames
        output_dir: Directory to save plots
    """
    # Filter tasks that have this metric
    tasks_with_metric = {
        task: df for task, df in task_data.items()
        if metric_name in df.columns and df[metric_name].notna().any()
    }

    if not tasks_with_metric:
        print(f"  ⚠ Warning: Metric '{metric_name}' not found in any task")
        return

    # Set style
    sns.set_style("whitegrid")
    # Use primary colors for different experiments
    primary_colors = ['#0066CC', '#DC3545', '#28A745', '#FFC107', '#6F42C1', '#FD7E14']
    sns.set_palette(primary_colors)

    # Create subplots - one per task
    n_tasks = len(tasks_with_metric)
    n_cols = min(3, n_tasks)
    n_rows = (n_tasks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # Make axes indexable even for single subplot
    if n_tasks == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each task
    for idx, (task_name, task_df) in enumerate(sorted(tasks_with_metric.items())):
        ax = axes[idx]

        # Get unique experiments and block lengths
        experiments = sorted(task_df["experiment"].unique())
        block_lengths = sorted(task_df["block_length"].unique())

        # Create mapping from block_length to decoding rate label (2^x)
        decoding_rate_labels = {bl: f"$2^{{{int(np.log2(bl))}}}$" for bl in block_lengths}

        # Plot each experiment
        for exp_name in experiments:
            exp_data = task_df[task_df["experiment"] == exp_name]

            # Group by block_length and compute mean
            grouped = exp_data.groupby("block_length")[metric_name].mean().reset_index()

            # Plot with markers
            ax.plot(
                grouped["block_length"],
                grouped[metric_name],
                'o-',
                linewidth=2,
                markersize=8,
                label=exp_name,
                alpha=0.8
            )

        # Customize subplot
        ax.set_xlabel("Decoding Rate", fontsize=11)
        ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=11)

        # Clean task name for display
        display_task_name = task_name.replace("_", " ").replace("/", " / ").title()
        ax.set_title(display_task_name, fontweight='bold')

        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)

        # Set x-axis to log scale and use custom labels
        ax.set_xscale('log', base=2)
        ax.set_xticks(block_lengths)
        ax.set_xticklabels([decoding_rate_labels[bl] for bl in block_lengths])

        # Set y-axis limits dynamically with 10% margin
        all_values = []
        for exp_name_inner in experiments:
            exp_data_inner = task_df[task_df["experiment"] == exp_name_inner]
            grouped_inner = exp_data_inner.groupby("block_length")[metric_name].mean()
            all_values.extend(grouped_inner.values)

        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            y_range = y_max - y_min
            margin = y_range * 0.1 if y_range > 0 else 5
            ax.set_ylim(y_min - margin, y_max + margin)

    # Remove extra subplots
    for idx in range(n_tasks, len(axes)):
        fig.delaxes(axes[idx])

    # Add main title
    fig.suptitle(
        f"Metric: {metric_name.replace('_', ' ').title()} Across All Tasks",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"combined_{metric_name}.png"

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved combined plot: {output_file}")


def main():
    """Main function to plot metrics from experiments."""
    parser = argparse.ArgumentParser(
        description="Plot metrics for experiments comparing different configurations."
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=["baseline"],
        help="List of experiment names (e.g., baseline modified)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory containing result JSONL files (default: results)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory to save plots (default: visualizations)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Specific metrics to plot (default: all available metrics)"
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Skip generating combined plots for common metrics"
    )

    args = parser.parse_args()

    # Convert to Path objects
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    # Load all experiments
    print("Loading experiment results...")
    experiments_data = {}
    for exp_name in args.experiments:
        print(f"  Loading {exp_name}...")
        try:
            experiments_data[exp_name] = load_experiment_results(exp_name, results_dir)
            print(f"    ✓ Loaded {len(experiments_data[exp_name])} results")
        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue

    if not experiments_data:
        print("Error: No experiment data loaded!")
        return

    # Prepare data grouped by task
    print("\nPreparing plot data...")
    task_data = prepare_plot_data(experiments_data)
    print(f"Found {len(task_data)} tasks")

    # Determine which metrics to plot (if specified)
    metric_columns = args.metrics if args.metrics else None

    if metric_columns:
        print(f"Plotting specified metrics: {metric_columns}")
    else:
        print("Auto-detecting metrics for each task")
    print()

    # Plot each task
    print("Generating individual task plots...")
    for task_name, task_df in task_data.items():
        print(f"  Plotting {task_name}...")
        try:
            plot_task_metrics(task_name, task_df, output_dir, metric_columns)
        except Exception as e:
            print(f"    ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate combined plots for common metrics
    if not args.skip_combined:
        print("\nGenerating combined plots for common metrics...")
        common_metrics = ["score", "grammar_score"]

        for metric in common_metrics:
            print(f"  Creating combined plot for '{metric}'...")
            try:
                plot_combined_metric(metric, task_data, output_dir)
            except Exception as e:
                print(f"    ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "=" * 80)
    print(f"All plots saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()