#!/usr/bin/env python3
"""
Build configuration list from a base config with varying block_length and steps.

Usage:
    python build_config_list.py base_config.yaml [output_list.yaml] [--block-lengths 1 2 4 8 16 32 64] [--all-tasks]
    
If output file is not specified, it will be auto-generated as <base>_list.yaml
For example: configs/ngram-debug.yaml -> configs/ngram-debug_list.yaml

Use --all-tasks to generate configs for all 5 tasks instead of just the one in base config.
"""

import yaml
import argparse
import sys
from pathlib import Path
from copy import deepcopy


def build_config_list(base_config, block_lengths=None, all_tasks=False):
    """
    Generate a list of configs with different block_length values.
    
    Args:
        base_config: Dictionary containing the base configuration
        block_lengths: List of block lengths to generate configs for.
                      If None, uses default powers of 2 up to max_tokens.
        all_tasks: If True, generate configs for all tasks instead of just the one
                  in base_config. Default: False.
    
    Returns:
        List of config dictionaries
    """
    # Define all tasks to use when all_tasks is True
    ALL_TASKS = [
        'paraphrase_summarize/chatgpt-paraphrases',
        'paraphrase_summarize/samsum',
        'words_to_sentence/easy',
        'words_to_sentence/medium',
        'words_to_sentence/hard',
    ]
    
    # Get max_tokens from the base config
    max_tokens = base_config.get('generation', {}).get('max_tokens')
    if max_tokens is None:
        raise ValueError("base_config must have 'generation.max_tokens' specified")
    
    # Check if save_logits is enabled (requires block_length=1)
    save_logits = base_config.get('generation', {}).get('save_logits', False)
    
    # If block_lengths not specified, use powers of 2 up to max_tokens
    if block_lengths is None:
        if save_logits:
            # save_logits only works with block_length=1
            block_lengths = [1]
            print("Note: save_logits=True detected, using only block_length=1")
        else:
            block_lengths = []
            bl = 1
            while bl <= max_tokens:
                block_lengths.append(bl)
                bl *= 2
    
    # Validate block_lengths
    for bl in block_lengths:
        if max_tokens % bl != 0:
            raise ValueError(f"max_tokens ({max_tokens}) must be divisible by block_length ({bl})")
        if save_logits and bl != 1:
            raise ValueError(f"save_logits=True requires block_length=1, but got block_length={bl}")
    
    # Determine which tasks to use
    if all_tasks:
        tasks = ALL_TASKS
    else:
        # Use the task from base_config
        tasks = [base_config.get('dataset', {}).get('task')]
        if tasks[0] is None:
            raise ValueError("base_config must have 'dataset.task' specified")
    
    # Generate config list
    config_list = []
    for task in tasks:
        for block_length in block_lengths:
            # Deep copy the base config
            config = deepcopy(base_config)
            
            # Set the task
            config['dataset']['task'] = task
            
            # Set block_length and calculate steps
            config['generation']['block_length'] = block_length
            config['generation']['steps'] = max_tokens // block_length
            
            config_list.append(config)
    
    return config_list


def load_base_config(config_path):
    """Load base configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config_list(config_list, output_path):
    """Save configuration list to YAML file."""
    with open(output_path, 'w') as f:
        yaml.dump(config_list, f, default_flow_style=False, sort_keys=False)
    print(f"✓ Generated {len(config_list)} configs in {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Build configuration list from base config',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate configs with default block lengths (auto-output to ngram-debug_list.yaml)
  python build_config_list.py configs/ngram-debug.yaml
  
  # Generate configs with custom block lengths
  python build_config_list.py configs/ngram-debug.yaml --block-lengths 1 2 4 8
  
  # Generate configs for all tasks
  python build_config_list.py configs/perplexity-beam.yaml --all-tasks
  
  # Generate configs for all tasks with custom block lengths
  python build_config_list.py configs/base.yaml --all-tasks --block-lengths 1 2 4 8 16 32 64
  
  # Specify custom output file
  python build_config_list.py configs/base.yaml configs/custom_output.yaml
        """
    )
    
    parser.add_argument(
        'base_config',
        type=str,
        help='Path to base configuration YAML file (e.g., configs/ngram-debug.yaml)'
    )
    
    parser.add_argument(
        'output_file',
        type=str,
        nargs='?',
        default=None,
        help='Path to output configuration list YAML file (default: auto-generated as <base>_list.yaml)'
    )
    
    parser.add_argument(
        '--block-lengths',
        type=int,
        nargs='+',
        default=None,
        help='List of block lengths to generate configs for (default: powers of 2 up to max_tokens)'
    )
    
    parser.add_argument(
        '--all-tasks',
        action='store_true',
        help='Generate configs for all tasks instead of just the one in base config'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.base_config).exists():
        print(f"Error: Base config file not found: {args.base_config}", file=sys.stderr)
        sys.exit(1)
    
    # Auto-generate output filename if not provided
    if args.output_file is None:
        base_path = Path(args.base_config)
        # Replace .yaml with _list.yaml
        if base_path.suffix == '.yaml':
            output_file = str(base_path.with_suffix('')) + '_list.yaml'
        else:
            output_file = str(base_path) + '_list.yaml'
        args.output_file = output_file
        print(f"Auto-generated output file: {args.output_file}")
    
    try:
        # Load base config
        print(f"Loading base config from {args.base_config}...")
        base_config = load_base_config(args.base_config)
        
        # Get max_tokens for display
        max_tokens = base_config.get('generation', {}).get('max_tokens', '?')
        print(f"  max_tokens: {max_tokens}")
        
        # Check if save_logits is enabled
        save_logits = base_config.get('generation', {}).get('save_logits', False)
        
        # Determine block lengths
        if args.block_lengths:
            block_lengths = args.block_lengths
            print(f"  block_lengths (custom): {block_lengths}")
        else:
            # Let build_config_list handle defaults (especially for save_logits)
            block_lengths = None
            if save_logits:
                print(f"  block_lengths (auto): [1] (restricted due to save_logits=True)")
            else:
                # Calculate default for display
                bl_list = []
                bl = 1
                while bl <= max_tokens:
                    bl_list.append(bl)
                    bl *= 2
                print(f"  block_lengths (auto): {bl_list}")
        
        # Build config list
        print(f"Generating configs...")
        if args.all_tasks:
            print(f"  Mode: all tasks")
        else:
            print(f"  Mode: single task")
        config_list = build_config_list(base_config, block_lengths, all_tasks=args.all_tasks)
        
        # Display preview
        print(f"\nGenerated configs:")
        for i, cfg in enumerate(config_list, 1):
            task = cfg['dataset']['task']
            bl = cfg['generation']['block_length']
            steps = cfg['generation']['steps']
            print(f"  {i:3d}. task={task:45s} block_length={bl:3d}, steps={steps:3d}")
        
        # Save to output file
        print(f"\nSaving to {args.output_file}...")
        save_config_list(config_list, args.output_file)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

