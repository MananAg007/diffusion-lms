# TESS Inference Guide

This guide explains how to set up the conda environment and run inference with TESS (Text-to-text Self-conditioned Simplex Diffusion).

## 1. Setting Up Conda Environment

### Step 1: Create the conda environment

The codebase uses a conda environment defined in `environment.yaml`. You have two options:

**Option A: Create environment in a specific location (recommended)**
```bash
# Set LOCAL_DIR to your preferred directory for the environment
export LOCAL_DIR=/home/mananaga  # or your preferred path

# Create the conda environment
conda env create -f environment.yaml --prefix ${LOCAL_DIR}/conda/envs/sdlm

# Activate the environment
conda activate ${LOCAL_DIR}/conda/envs/sdlm
```

**Option B: Create environment with default name**
```bash
# Create the environment (will be named 'sdlm')
conda env create -f environment.yaml

# Activate the environment
conda activate sdlm
```

### Step 2: Install the package in development mode

From the `tess-diffusion` directory:
```bash
cd /home/mananaga/diffusion-lms/tess-diffusion
pip install -e .
```

### Step 3: Update environment (if needed)

If you need to update dependencies later:
```bash
conda env update --file environment.yaml --prune
```

## 2. Understanding Inference

TESS inference is done through task-specific scripts:
- **`run_mlm.py`** - For masked language modeling / unconditional generation
- **`run_summarization.py`** - For summarization tasks
- **`run_translation.py`** - For translation tasks  
- **`run_simplification.py`** - For text simplification
- **`run_glue.py`** - For GLUE classification tasks

All scripts use the `--do_eval` flag to run inference (without training).

## 3. Running Inference

### Basic Inference Command

The general format for inference is:

```bash
python run_mlm.py \
    --model_name_or_path <path_to_trained_model_or_checkpoint> \
    --do_eval \
    --output_dir <output_directory> \
    --config_file configs/<config_file>.json \
    [additional arguments]
```

### Key Inference Parameters

- `--model_name_or_path`: Path to trained model checkpoint or HuggingFace model name
- `--do_eval`: Enable evaluation/inference mode (required for inference)
- `--output_dir`: Directory to save inference results
- `--max_eval_samples`: Number of samples to evaluate (default: all)
- `--num_inference_diffusion_steps`: Number of diffusion steps during inference (can be less than training steps)
- `--top_p`: Nucleus sampling parameter (e.g., 0.95, 0.99)
- `--temperature`: Temperature for sampling (default: 1.0)
- `--per_device_eval_batch_size`: Batch size for evaluation
- `--truncation_length`: Length to truncate sequences
- `--eval_context_size`: Context size for conditional generation (e.g., 25)
- `--load_states_in_eval_from_model_path`: Load checkpoint states from model path
- `--skip_special_tokens`: Whether to skip special tokens in output (True/False)

### Example 1: Unconditional Generation (MLM)

```bash
python run_mlm.py \
    --model_name_or_path <path_to_checkpoint> \
    --do_eval \
    --output_dir ./outputs/inference_results \
    --max_eval_samples 100 \
    --max_seq_length 256 \
    --truncation_length 56 \
    --num_inference_diffusion_steps 1000 \
    --top_p 0.95 \
    --temperature 1.0 \
    --per_device_eval_batch_size 25 \
    --eval_context_size 25 \
    --simplex_value 5 \
    --load_states_in_eval_from_model_path \
    --skip_special_tokens True \
    --tokenized_data_path processed_data/openwebtext_256_split_gpt_eval/
```

### Example 2: Using a Config File

Create or use an existing config file (e.g., `configs/simple_data_test.json`):

```bash
python run_mlm.py configs/simple_data_test.json \
    --do_eval \
    --model_name_or_path <path_to_checkpoint> \
    --output_dir ./outputs/inference_results \
    --max_eval_samples 100
```

### Example 3: Summarization Task

```bash
python run_summarization.py \
    --model_name_or_path <path_to_checkpoint> \
    --do_eval \
    --dataset_name xsum \
    --output_dir ./outputs/summarization_results \
    --max_eval_samples 96 \
    --max_source_length 384 \
    --max_target_length 128 \
    --max_seq_length 512 \
    --num_inference_diffusion_steps 2500 \
    --conditional_generation seq2seq \
    --simplex_value 5 \
    --top_p 0.99
```

### Example 4: With Self-Conditioning

If your model was trained with self-conditioning:

```bash
python run_mlm.py \
    --model_name_or_path <path_to_checkpoint> \
    --do_eval \
    --output_dir ./outputs/inference_results \
    --self_condition logits_addition \
    --max_eval_samples 1000 \
    --num_inference_diffusion_steps 1000 \
    --top_p 0.95 \
    --temperature 1.0 \
    --max_seq_length 256 \
    --truncation_length 206 \
    --eval_context_size 25 \
    --tokenized_data_path processed_data/openwebtext_256_split_gpt_eval/
```

## 4. Using Pre-trained Models

If you have a pre-trained model checkpoint, point `--model_name_or_path` to:
- The checkpoint directory (e.g., `checkpoint-15000`)
- Or a path like `ul2/checkpoint-15000` if organized in subdirectories

The checkpoint should contain:
- `config.json` - Model configuration
- `pytorch_model.bin` or model files - Model weights
- `tokenizer files` - Tokenizer configuration

## 5. Understanding Output

After inference, the results will be saved in the `output_dir`:
- Generated texts (in various formats: from simplex, from logits, etc.)
- Evaluation metrics (perplexity, distinct-n, repetition, etc.)
- Token predictions

The outputs include:
- `pred_texts_from_simplex`: Texts generated from simplex representation
- `pred_texts_from_logits`: Texts generated from logits
- `pred_texts_from_simplex_masked`: Masked portions (for conditional generation)
- Various metrics in JSON format

## 6. Common Issues

1. **CUDA/GPU**: Ensure CUDA is available if using GPU:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python run_mlm.py ...
   ```

2. **Multi-GPU**: Use accelerate for multi-GPU inference:
   ```bash
   accelerate launch --config_file configs/accelerate_1_gpu.yaml run_mlm.py ...
   ```

3. **Data Path**: Ensure the tokenized data path exists or provide raw data files with `--train_file` and `--validation_file`

4. **Memory Issues**: Reduce `--per_device_eval_batch_size` if running out of memory

## 7. Additional Tools

- **`compute_mlm_metrics.py`**: Compute additional metrics after inference
- **`gpt2_eval.py`**: Evaluate with GPT-2 as reference model
- **`gold_text_eval.py`**: Compare against gold standard texts

Example:
```bash
python compute_mlm_metrics.py \
    --model_name_or_path <checkpoint_path> \
    --output_dir <inference_output_dir> \
    --eval_for_all_metrics \
    [same arguments as inference]
```

