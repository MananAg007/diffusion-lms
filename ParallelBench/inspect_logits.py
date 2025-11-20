#!/usr/bin/env python3
"""
Simple script to inspect the saved logits data format.
Usage: python inspect_logits.py configs/baseline_list/00000.json.gz
"""

import json
import gzip
import sys
from pathlib import Path
import torch
from safetensors.torch import load_file as safetensors_load

def inspect_logits(results_file):
    """Inspect the format of saved logits data."""
    results_file = Path(results_file)
    
    print("="*70)
    print("INSPECTING SAVED LOGITS DATA")
    print("="*70)
    
    # 1. Check main results file
    print("\n1. MAIN RESULTS FILE:")
    print(f"   File: {results_file}")
    with gzip.open(results_file, 'rt') as f:
        results = json.load(f)
    
    print(f"   Total outputs: {len(results['outputs'])}")
    
    # Check first output for saved_logits_metadata
    first_output = results['outputs'][0]
    if 'saved_logits_metadata' in first_output:
        metadata = first_output['saved_logits_metadata']
        print(f"\n   Saved Logits Metadata:")
        print(f"     - Number of entries: {metadata['num_entries']}")
        print(f"     - Logits file: {metadata['logits_file']}")
        print(f"     - Horizons: {metadata['horizons']}")
    
    # 2. Load safetensors file
    logits_file = results_file.parent / metadata['logits_file']
    print(f"\n2. SAFETENSORS FILE:")
    print(f"   File: {logits_file}")
    print(f"   Size: {logits_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    tensors_dict = safetensors_load(str(logits_file))
    print(f"   Number of tensors: {len(tensors_dict)}")
    
    # Show first few keys
    print(f"\n   First 5 tensor keys:")
    for i, key in enumerate(list(tensors_dict.keys())[:5]):
        tensor = tensors_dict[key]
        print(f"     {key}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    # 3. Load metadata file
    metadata_file = logits_file.with_suffix('.metadata.json')
    print(f"\n3. METADATA FILE:")
    print(f"   File: {metadata_file}")
    print(f"   Size: {metadata_file.stat().st_size / 1024:.2f} KB")
    
    with open(metadata_file, 'r') as f:
        metadata_list = json.load(f)
    
    print(f"   Number of entries: {len(metadata_list)}")
    
    # Show first entry
    print(f"\n   First entry:")
    first_entry = metadata_list[0]
    for key, value in first_entry.items():
        print(f"     {key}: {value}")
    
    # 4. Show a complete example (metadata + tensor)
    print(f"\n4. COMPLETE EXAMPLE (Entry 0):")
    print(f"   Metadata:")
    for key, value in first_entry.items():
        if key != 'key':
            print(f"     {key}: {value}")
    
    # Get corresponding tensor
    tensor_key = first_entry['key']
    tensor = tensors_dict[tensor_key]
    
    print(f"\n   Tensor:")
    print(f"     Shape: {tensor.shape}")
    print(f"     Dtype: {tensor.dtype}")
    print(f"     Device: {tensor.device}")
    print(f"     Min value: {tensor.min().item():.4f}")
    print(f"     Max value: {tensor.max().item():.4f}")
    print(f"     Mean value: {tensor.mean().item():.4f}")
    
    # Show top-5 predictions
    probs = torch.softmax(tensor, dim=-1)
    top5_probs, top5_indices = torch.topk(probs, 5)
    
    print(f"\n   Top-5 predictions:")
    for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs)):
        print(f"     {i+1}. Token {idx.item():6d}: probability = {prob.item():.6f}")
    
    # 5. Statistics by horizon
    print(f"\n5. STATISTICS BY HORIZON:")
    by_horizon = {1: [], 2: [], 4: [], 8: []}
    
    for entry in metadata_list:
        if entry['sample_idx'] == 0:  # Just first sample
            by_horizon[entry['horizon']].append(entry)
    
    for horizon in [1, 2, 4, 8]:
        entries = by_horizon[horizon]
        print(f"   Horizon {horizon}: {len(entries)} entries")
    
    # 6. Show example of tracking one position
    print(f"\n6. TRACKING A SINGLE POSITION:")
    target_position = 5
    predictions_for_pos = [e for e in metadata_list 
                          if e['sample_idx'] == 0 and e['relative_position'] == target_position]
    predictions_for_pos.sort(key=lambda x: x['timestep'])
    
    print(f"   Position {target_position}: {len(predictions_for_pos)} predictions over time")
    
    for pred in predictions_for_pos:
        # Get tensor and top prediction
        tensor = tensors_dict[pred['key']]
        top_token = tensor.argmax().item()
        top_prob = torch.softmax(tensor, dim=-1).max().item()
        
        print(f"     Timestep {pred['timestep']:2d}, horizon {pred['horizon']}: "
              f"predicted token {top_token:6d} (conf={top_prob:.4f})")
    
    # 7. Size comparison
    print(f"\n7. STORAGE EFFICIENCY:")
    total_size = (logits_file.stat().st_size + 
                 metadata_file.stat().st_size + 
                 results_file.stat().st_size)
    print(f"   Total size: {total_size / 1024 / 1024:.2f} MB")
    print(f"   - Safetensors: {logits_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   - Metadata JSON: {metadata_file.stat().st_size / 1024:.2f} KB")
    print(f"   - Results JSON: {results_file.stat().st_size / 1024:.2f} KB")
    
    # Estimate what JSON would have been
    vocab_size = tensor.shape[0]
    num_entries = len(metadata_list)
    # Each float as string ~8 bytes, plus JSON overhead
    estimated_json_size = num_entries * vocab_size * 8 / 1024 / 1024
    print(f"\n   Estimated size if stored as JSON: ~{estimated_json_size:.2f} MB")
    print(f"   Space saved: ~{(1 - total_size/1024/1024/estimated_json_size)*100:.1f}%")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_logits.py <results_file.json.gz>")
        print("Example: python inspect_logits.py configs/baseline_list/00000.json.gz")
        sys.exit(1)
    
    results_file = sys.argv[1]
    inspect_logits(results_file)

