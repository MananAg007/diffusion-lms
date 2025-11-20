"""
Helper utilities for loading and analyzing saved logits data.

Usage:
    from utils.load_saved_logits import load_logits_data, analyze_divergence
    
    # Load logits data
    logits_data = load_logits_data('results/experiment/config.json.gz')
    
    # Analyze prediction divergence
    divergence_stats = analyze_divergence(logits_data, sample_idx=0)
"""

import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np

try:
    from safetensors.torch import load_file as safetensors_load
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available. Install with: pip install safetensors")


def load_logits_data(results_file: str) -> List[Dict]:
    """
    Load saved logits data from results file.
    
    Args:
        results_file: Path to results .json.gz file
    
    Returns:
        List of dictionaries containing logits data with tensors loaded
    """
    results_file = Path(results_file)
    
    # Load main results to find logits file
    with gzip.open(results_file, 'rt') as f:
        results = json.load(f)
    
    # Check if any outputs have saved_logits_metadata
    has_logits = any('saved_logits_metadata' in output for output in results['outputs'])
    
    if not has_logits:
        print("No saved logits data found in results file.")
        return []
    
    # Find the logits file
    logits_file = None
    for output in results['outputs']:
        if 'saved_logits_metadata' in output:
            metadata = output['saved_logits_metadata']
            logits_file = results_file.parent / metadata['logits_file']
            break
    
    if logits_file is None:
        print("Could not find logits file reference.")
        return []
    
    # Load based on file extension
    if logits_file.suffix == '.safetensors':
        return _load_from_safetensors(logits_file)
    elif logits_file.suffix == '.pt':
        return _load_from_torch(logits_file)
    else:
        print(f"Unknown logits file format: {logits_file.suffix}")
        return []


def _load_from_safetensors(logits_file: Path) -> List[Dict]:
    """Load logits data from safetensors format."""
    if not SAFETENSORS_AVAILABLE:
        raise ImportError("safetensors not available. Install with: pip install safetensors")
    
    # Load tensors
    tensors_dict = safetensors_load(str(logits_file))
    
    # Load metadata
    metadata_file = logits_file.with_suffix('.metadata.json')
    with open(metadata_file, 'r') as f:
        metadata_list = json.load(f)
    
    # Combine tensors with metadata
    logits_data = []
    for metadata in metadata_list:
        key = metadata['key']
        entry = {
            'sample_idx': metadata['sample_idx'],
            'question_id': metadata['question_id'],
            'timestep': metadata['timestep'],
            'horizon': metadata['horizon'],
            'absolute_position': metadata['absolute_position'],
            'relative_position': metadata['relative_position'],
            'last_unmasked_position': metadata['last_unmasked_position'],
            'logits': tensors_dict[key]
        }
        logits_data.append(entry)
    
    print(f"Loaded {len(logits_data)} logits entries from {logits_file}")
    return logits_data


def _load_from_torch(logits_file: Path) -> List[Dict]:
    """Load logits data from torch.save format."""
    logits_data = torch.load(logits_file)
    print(f"Loaded {len(logits_data)} logits entries from {logits_file}")
    return logits_data


def get_predictions_for_position(logits_data: List[Dict], 
                                 sample_idx: int, 
                                 relative_position: int) -> List[Dict]:
    """
    Get all predictions made for a specific position across timesteps.
    
    Args:
        logits_data: Loaded logits data
        sample_idx: Sample index
        relative_position: Position within generation region
    
    Returns:
        List of entries for that position, sorted by timestep
    """
    predictions = [
        entry for entry in logits_data
        if entry['sample_idx'] == sample_idx and 
           entry['relative_position'] == relative_position
    ]
    predictions.sort(key=lambda x: x['timestep'])
    return predictions


def calculate_kl_divergence(logits1: torch.Tensor, logits2: torch.Tensor) -> float:
    """Calculate KL divergence between two logit distributions."""
    p = torch.softmax(logits1, dim=-1)
    q = torch.softmax(logits2, dim=-1)
    
    # KL(P || Q)
    kl = torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10)))
    return kl.item()


def calculate_js_divergence(logits1: torch.Tensor, logits2: torch.Tensor) -> float:
    """Calculate Jensen-Shannon divergence between two logit distributions."""
    p = torch.softmax(logits1, dim=-1)
    q = torch.softmax(logits2, dim=-1)
    m = 0.5 * (p + q)
    
    kl_pm = torch.sum(p * (torch.log(p + 1e-10) - torch.log(m + 1e-10)))
    kl_qm = torch.sum(q * (torch.log(q + 1e-10) - torch.log(m + 1e-10)))
    
    js = 0.5 * (kl_pm + kl_qm)
    return js.item()


def calculate_top_k_agreement(logits1: torch.Tensor, logits2: torch.Tensor, k: int = 5) -> float:
    """Calculate agreement in top-k predictions."""
    top_k1 = set(torch.topk(logits1, k).indices.tolist())
    top_k2 = set(torch.topk(logits2, k).indices.tolist())
    
    intersection = len(top_k1 & top_k2)
    return intersection / k


def analyze_divergence(logits_data: List[Dict], 
                       sample_idx: int = 0,
                       metrics: List[str] = ['kl', 'js', 'top5'],
                       verbose: bool = False) -> Dict:
    """
    Analyze prediction divergence across horizons and timesteps.
    
    Args:
        logits_data: Loaded logits data
        sample_idx: Sample index to analyze
        metrics: List of metrics to compute ('kl', 'js', 'top5', 'top1')
        verbose: If True, print detailed information about what's being compared
    
    Returns:
        Dictionary with divergence statistics
    """
    sample_data = [entry for entry in logits_data if entry['sample_idx'] == sample_idx]
    
    if not sample_data:
        return {}
    
    # Group by position
    positions = sorted(set(entry['relative_position'] for entry in sample_data))
    
    results = {
        'by_position': {},
        'by_horizon': {1: [], 2: [], 4: [], 8: []},
        'summary': {},
        'skipped_positions': []
    }
    
    if verbose:
        print("\n" + "="*70)
        print("DETAILED ANALYSIS OF PREDICTION DIVERGENCE")
        print("="*70)
        print(f"\nSample {sample_idx}: Analyzing {len(positions)} positions")
        print(f"Position range: {min(positions)} to {max(positions)}")
    
    for pos in positions:
        predictions = get_predictions_for_position(logits_data, sample_idx, pos)
        
        if len(predictions) < 2:
            results['skipped_positions'].append(pos)
            if verbose:
                print(f"\nPosition {pos}: SKIPPED (only {len(predictions)} prediction)")
            continue
        
        # Get horizons available for this position
        horizons_available = [p['horizon'] for p in predictions]
        
        # Compare earliest vs latest prediction
        early = predictions[0]
        late = predictions[-1]
        early_logits = early['logits']
        late_logits = late['logits']
        
        pos_stats = {
            'num_predictions': len(predictions),
            'horizons_available': horizons_available,
            'earliest_horizon': early['horizon'],
            'earliest_timestep': early['timestep'],
            'latest_horizon': late['horizon'],
            'latest_timestep': late['timestep'],
        }
        
        if verbose:
            print(f"\nPosition {pos}:")
            print(f"  Available horizons: {horizons_available}")
            print(f"  Comparing:")
            print(f"    EARLY: timestep={early['timestep']}, horizon={early['horizon']} (token {early_logits.argmax().item()})")
            print(f"    LATE:  timestep={late['timestep']}, horizon={late['horizon']} (token {late_logits.argmax().item()})")
        
        if 'kl' in metrics:
            pos_stats['kl_divergence'] = calculate_kl_divergence(early_logits, late_logits)
            if verbose:
                print(f"  KL divergence: {pos_stats['kl_divergence']:.4f}")
        
        if 'js' in metrics:
            pos_stats['js_divergence'] = calculate_js_divergence(early_logits, late_logits)
        
        if 'top5' in metrics:
            pos_stats['top5_agreement'] = calculate_top_k_agreement(early_logits, late_logits, k=5)
        
        if 'top1' in metrics:
            early_top1 = early_logits.argmax().item()
            late_top1 = late_logits.argmax().item()
            pos_stats['top1_changed'] = (early_top1 != late_top1)
            pos_stats['early_top1'] = early_top1
            pos_stats['late_top1'] = late_top1
        
        results['by_position'][pos] = pos_stats
        
        # Group by earliest prediction's horizon
        horizon = predictions[0]['horizon']
        if 'kl' in metrics:
            results['by_horizon'][horizon].append(pos_stats['kl_divergence'])
            if verbose:
                print(f"  → Added to Horizon {horizon} bucket")
    
    # Compute summary statistics
    if 'kl' in metrics:
        for horizon in [1, 2, 4, 8]:
            if results['by_horizon'][horizon]:
                results['summary'][f'horizon_{horizon}_mean_kl'] = np.mean(results['by_horizon'][horizon])
                results['summary'][f'horizon_{horizon}_std_kl'] = np.std(results['by_horizon'][horizon])
                results['summary'][f'horizon_{horizon}_count'] = len(results['by_horizon'][horizon])
    
    if verbose:
        print("\n" + "="*70)
    
    return results


def print_divergence_summary(divergence_stats: Dict):
    """Print a human-readable summary of divergence statistics."""
    print("\n" + "="*60)
    print("PREDICTION DIVERGENCE ANALYSIS SUMMARY")
    print("="*60)
    
    # Show skipped positions
    if 'skipped_positions' in divergence_stats and divergence_stats['skipped_positions']:
        print(f"\nSkipped positions (only 1 prediction): {divergence_stats['skipped_positions']}")
    
    if 'by_position' in divergence_stats:
        print(f"\nAnalyzed positions: {len(divergence_stats['by_position'])}")
        if divergence_stats['by_position']:
            analyzed_positions = sorted(divergence_stats['by_position'].keys())
            print(f"Position range: {min(analyzed_positions)} to {max(analyzed_positions)}")
    
    if 'summary' in divergence_stats:
        print("\nSummary by Horizon (earliest prediction's horizon):")
        print("-" * 60)
        for horizon in [1, 2, 4, 8]:
            mean_key = f'horizon_{horizon}_mean_kl'
            std_key = f'horizon_{horizon}_std_kl'
            count_key = f'horizon_{horizon}_count'
            if mean_key in divergence_stats['summary']:
                mean_kl = divergence_stats['summary'][mean_key]
                std_kl = divergence_stats['summary'][std_key]
                count = divergence_stats['summary'][count_key]
                print(f"  Horizon {horizon}: KL = {mean_kl:.4f} ± {std_kl:.4f} (n={count} positions)")
                print(f"    → Comparing positions where earliest prediction was {horizon} steps ahead")
        
        # Show a few examples
        print("\nExample positions (first 5):")
        print("-" * 60)
        for pos, stats in list(divergence_stats['by_position'].items())[:5]:
            print(f"  Position {pos}:")
            print(f"    Available horizons: {stats.get('horizons_available', 'N/A')}")
            print(f"    Comparing: timestep {stats.get('earliest_timestep', '?')} (horizon {stats.get('earliest_horizon', '?')}) "
                  f"vs timestep {stats.get('latest_timestep', '?')} (horizon {stats.get('latest_horizon', '?')})")
            if 'kl_divergence' in stats:
                print(f"    KL divergence: {stats['kl_divergence']:.4f}")
            if 'top1_changed' in stats:
                print(f"    Top-1 changed: {stats['top1_changed']} ({stats.get('early_top1', '?')} → {stats.get('late_top1', '?')})")
    
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_saved_logits.py <results_file.json.gz> [--verbose]")
        print("  --verbose: Show detailed comparison for each position")
        sys.exit(1)
    
    results_file = sys.argv[1]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    print(f"Loading logits data from: {results_file}")
    logits_data = load_logits_data(results_file)
    
    if logits_data:
        print(f"\nAnalyzing divergence for first sample...")
        divergence_stats = analyze_divergence(logits_data, sample_idx=0, verbose=verbose)
        print_divergence_summary(divergence_stats)

