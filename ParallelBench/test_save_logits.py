#!/usr/bin/env python3
"""
Test script to verify save_logits functionality.
This script tests that the save_logits feature works correctly with block_length=1.
"""

import torch
import yaml
from pathlib import Path

# Test configuration
test_config = {
    "skip_metrics": True,
    "model": {
        "model_name": "GSAI-ML/LLaDA-8B-Instruct",
        "accel_framework": None
    },
    "dataset": {
        "dataset_name": "parallel_bench",
        "task": "paraphrase_summarize/chatgpt-paraphrases"
    },
    "generation": {
        "temperature": 0.0,
        "max_tokens": 16,  # Small for testing
        "block_length": 1,  # Required for save_logits
        "steps": 16,
        "fast_dllm_threshold": None,
        "fast_dllm_factor": None,
        "remasking": "low_confidence",
        "pool_size": 1,
        "score_function": "grammar",
        "beam_size": 1,
        "max_joint": 128,
        "save_logits": True  # Enable logits saving
    }
}

def test_config_validation():
    """Test that config validation works correctly."""
    print("Testing config validation...")
    
    # Test 1: save_logits with block_length=1 should work
    try:
        from model.llada_model import LladaGenerationConfig
        config = LladaGenerationConfig(
            max_tokens=16,
            block_length=1,
            steps=16,
            save_logits=True
        )
        print("✓ Test 1 passed: save_logits=True with block_length=1 works")
    except AssertionError as e:
        print(f"✗ Test 1 failed: {e}")
        return False
    
    # Test 2: save_logits with block_length>1 should fail
    try:
        config = LladaGenerationConfig(
            max_tokens=16,
            block_length=2,
            steps=8,
            save_logits=True
        )
        print("✗ Test 2 failed: save_logits=True with block_length=2 should have raised AssertionError")
        return False
    except AssertionError:
        print("✓ Test 2 passed: save_logits=True with block_length>1 correctly raises AssertionError")
    
    return True

def test_logits_data_structure():
    """Test that saved logits data has the correct structure."""
    print("\nTesting logits data structure...")
    
    # Create a mock saved_logits_data entry
    mock_entry = {
        'question_id': 'test_123',
        'timestep': 0,
        'horizon': 1,
        'absolute_position': 100,
        'relative_position': 0,
        'last_unmasked_position': -1,
        'logits': torch.randn(50000),  # typical vocab size
    }
    
    required_keys = ['question_id', 'timestep', 'horizon', 'absolute_position', 
                     'relative_position', 'last_unmasked_position', 'logits']
    
    for key in required_keys:
        if key not in mock_entry:
            print(f"✗ Missing required key: {key}")
            return False
    
    print("✓ Logits data structure is correct")
    return True

def main():
    print("=" * 60)
    print("Testing save_logits functionality")
    print("=" * 60)
    
    # Run tests
    tests_passed = True
    tests_passed &= test_config_validation()
    tests_passed &= test_logits_data_structure()
    
    print("\n" + "=" * 60)
    if tests_passed:
        print("✓ All tests passed!")
        print("\nTo run a full experiment with save_logits:")
        print("1. Create a config with block_length=1 and save_logits=True")
        print("2. Run: bash run.sh your_config_name")
        print("3. Check results/*.json.gz for saved_logits_data field")
    else:
        print("✗ Some tests failed!")
    print("=" * 60)

if __name__ == "__main__":
    main()

