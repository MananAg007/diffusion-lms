#!/usr/bin/env python3
"""
Compute metrics for experiments in an output directory.

This script processes experiment results stored in compressed JSON files,
computes metrics for each experiment, and saves aggregated results.

Usage:
    python compute_metrics.py --output_dir <output_directory>

Example:
    python compute_metrics.py --output_dir baseline_list
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm
import pandas as pd

from dataset.parallel_bench import ParallelBench


def load_experiment_results(output_dir: Path) -> List[Dict[str, Any]]:
    """
    Load all experiment results from gzipped JSON files in the output directory.

    Args:
        output_dir: Path to the directory containing experiment results

    Returns:
        List of experiment result dictionaries
    """
    experiment_files = sorted(output_dir.glob("*.json.gz"))

    if not experiment_files:
        raise ValueError(f"No .json.gz files found in {output_dir}")

    experiments = []
    for exp_file in tqdm(experiment_files, desc="Loading experiment files"):
        try:
            with gzip.open(exp_file, "rt") as f:
                experiment = json.load(f)
                experiment["experiment_file"] = str(exp_file.name)
                experiments.append(experiment)
        except Exception as e:
            print(f"Error loading {exp_file}: {e}")
            continue

    return experiments


def extract_config_details(cfg_file: str) -> Dict[str, Any]:
    """
    Extract configuration details from the cfg_file path.

    Args:
        cfg_file: Path to the configuration file

    Returns:
        Dictionary containing configuration details
    """
    # Example: "temp/baseline_list/00000.yaml"
    cfg_path = Path(cfg_file)
    return {
        "cfg_file": cfg_file,
        "experiment_name": cfg_path.parent.name,
        "config_id": cfg_path.stem,
    }


def compute_metrics_for_experiment(experiment: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute metrics for a single experiment if not already computed.

    Args:
        experiment: Dictionary containing experiment configuration and outputs

    Returns:
        Dictionary containing experiment details and computed metrics
    """
    # Extract configuration details
    config_details = extract_config_details(experiment["cfg_file"])

    # Check if metrics are already computed
    existing_metrics = experiment.get("metrics", {})

    # If metrics exist and are non-empty, use them
    if existing_metrics and len(existing_metrics) > 0:
        result = {
            **config_details,
            "metrics_computed": False,
            "metrics": existing_metrics,
            "num_samples": len(experiment.get("outputs", []))
        }
        return result

    # Extract outputs and labels
    outputs_data = experiment.get("outputs", [])
    if not outputs_data:
        print(f"Warning: No outputs found in {config_details['config_id']}")
        return {
            **config_details,
            "metrics_computed": False,
            "metrics": {},
            "num_samples": 0,
            "error": "No outputs found"
        }

    # Try to determine task from first output
    try:
        # Load the original config to get task information
        import yaml
        cfg_file_path = Path(experiment["cfg_file"])

        if cfg_file_path.exists():
            with open(cfg_file_path, "r") as f:
                cfg = yaml.safe_load(f)
            task = cfg.get("dataset", {}).get("task")

            if task:
                # Load the dataset to get the metric function
                dataset = ParallelBench(task)

                # Extract predictions and references
                predictions = [output["output"] for output in outputs_data]
                references = [output["label"] for output in outputs_data if output.get("label") is not None]

                if len(predictions) != len(references):
                    print(f"Warning: Mismatch in predictions ({len(predictions)}) and references ({len(references)})")
                    references = [output.get("label") for output in outputs_data]

                # Compute metrics
                metrics = dataset.compute_metrics(predictions, references)

                # Extract generation parameters
                generation_params = cfg.get("generation", {})

                result = {
                    **config_details,
                    "task": task,
                    "model_name": cfg.get("model", {}).get("model_name"),
                    "block_length": generation_params.get("block_length"),
                    "steps": generation_params.get("steps"),
                    "max_tokens": generation_params.get("max_tokens"),
                    "temperature": generation_params.get("temperature"),
                    "remasking": generation_params.get("remasking"),
                    "fast_dllm_threshold": generation_params.get("fast_dllm_threshold"),
                    "fast_dllm_factor": generation_params.get("fast_dllm_factor"),
                    "metrics_computed": True,
                    "metrics": metrics,
                    "num_samples": len(predictions)
                }

                return result

        # Fallback: metrics already exist or cannot be computed
        return {
            **config_details,
            "metrics_computed": False,
            "metrics": existing_metrics,
            "num_samples": len(outputs_data),
            "error": "Could not load task configuration"
        }

    except Exception as e:
        print(f"Error computing metrics for {config_details['config_id']}: {e}")
        import traceback
        traceback.print_exc()
        return {
            **config_details,
            "metrics_computed": False,
            "metrics": existing_metrics,
            "num_samples": len(outputs_data),
            "error": str(e)
        }


def print_summary_statistics(results: List[Dict[str, Any]]):
    """
    Print summary statistics for the computed results.

    Args:
        results: List of result dictionaries
    """
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    total_experiments = len(results)
    computed = sum(1 for r in results if r.get("metrics_computed", False))
    existing = total_experiments - computed

    print(f"Total experiments processed: {total_experiments}")
    print(f"Newly computed metrics: {computed}")
    print(f"Used existing metrics: {existing}")

    # Group by task and show average scores
    df = pd.DataFrame(results)

    if "task" in df.columns:
        print("\n" + "-" * 80)
        print("METRICS BY TASK")
        print("-" * 80)

        # Expand metrics column
        metrics_df = pd.json_normalize(df['metrics'])
        df_expanded = pd.concat([df.drop('metrics', axis=1), metrics_df], axis=1)

        # Group by task and show statistics
        if "score" in metrics_df.columns:
            grouped = df_expanded.groupby("task")["score"].agg(["mean", "std", "min", "max", "count"])
            print(grouped.to_string())

        # Group by task and block_length if available
        if "block_length" in df.columns and "score" in metrics_df.columns:
            print("\n" + "-" * 80)
            print("METRICS BY TASK AND BLOCK LENGTH")
            print("-" * 80)
            pivot = df_expanded.pivot_table(
                values="score",
                index="task",
                columns="block_length",
                aggfunc="mean"
            )
            print(pivot.to_string())


def main():
    """Main function to process experiments and compute metrics."""
    parser = argparse.ArgumentParser(
        description="Compute metrics for experiments in an output directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="configs/grammar-beam-8_list",
        help="Path to the output directory containing experiment results (e.g., outputs/baseline_list)"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to save aggregated results (default: results)"
    )

    args = parser.parse_args()

    # Kill any lingering LanguageTool Java servers from previous runs
    print("Cleaning up old LanguageTool servers...")
    import subprocess
    try:
        subprocess.run(["pkill", "-9", "-f", "languagetool-server.jar"],
                      stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        print("Old servers cleaned up.\n")
    except Exception as e:
        print(f"Note: Could not clean up old servers: {e}\n")

    # Pre-initialize LanguageTool to avoid multiple downloads
    print("Initializing LanguageTool (one-time setup)...")
    from utils.grammar_check import get_language_tool
    get_language_tool()
    print("LanguageTool ready!\n")

    # Convert to Path objects
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)

    # Validate output directory
    if not output_dir.exists():
        raise ValueError(f"Output directory does not exist: {output_dir}")

    if not output_dir.is_dir():
        raise ValueError(f"Output path is not a directory: {output_dir}")

    print(f"Processing experiments from: {output_dir}")
    print(f"Results will be saved to: {results_dir}")
    print()

    # Load all experiment results
    experiments = load_experiment_results(output_dir)
    print(f"\nLoaded {len(experiments)} experiments")

    # Prepare output file
    results_dir.mkdir(parents=True, exist_ok=True)
    output_name = output_dir.name
    if output_name.endswith("_list"):
        output_name = output_name[:-5]  # Remove "_list" suffix
    output_file = results_dir / f"{output_name}.jsonl"

    # Delete existing results file if it exists
    if output_file.exists():
        print(f"Deleting existing results file: {output_file}")
        output_file.unlink()

    # Open file for writing (will append as we go)
    print(f"Results will be written to: {output_file}")
    print()

    # Compute metrics for each experiment and write immediately
    results = []
    with open(output_file, "w") as f:
        for experiment in tqdm(experiments, desc="Computing metrics"):
            result = compute_metrics_for_experiment(experiment)
            results.append(result)

            # Write result immediately to file
            f.write(json.dumps(result) + "\n")
            f.flush()  # Ensure it's written to disk immediately

    # Print summary statistics
    print(f"\nAll results saved to: {output_file}")
    print_summary_statistics(results)

    print("\n" + "=" * 80)
    print("Processing complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()