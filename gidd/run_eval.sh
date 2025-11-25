#!/bin/bash
#SBATCH --partition=flame
#SBATCH --qos=flame-8gpu-b_qos
#SBATCH --account=aditirag
#SBATCH --job-name=gidd-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out
mkdir -p /home/mananaga/logs/%j

echo "Working directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job submitted from: $SLURM_SUBMIT_DIR"
echo "Running on node: $SLURMD_NODENAME"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Activate conda environment
source /project/flame/mananaga/miniconda3/etc/profile.d/conda.sh && conda activate gidd

cd /home/mananaga/diffusion-lms/gidd

# Add current directory to PYTHONPATH so Python can find the gidd module
export PYTHONPATH=/home/mananaga/diffusion-lms/gidd:${PYTHONPATH:-}

export HF_HOME=/project/flame/mananaga/.hf_cache
export HF_HUB_CACHE=/project/flame/mananaga/.hf_cache/hub
export HF_DATASETS_CACHE=/project/flame/mananaga/.hf_cache/datasets
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# Default checkpoint path - change this or pass as argument
CHECKPOINT_PATH=${1:-/project/flame/mananaga/gidd/outputs/latest}
SAMPLES_PATH=/project/flame/mananaga/gidd/outputs/samples.pt
METRICS_PATH=/project/flame/mananaga/gidd/outputs/gen_ppl_metrics.json

mkdir -p /project/flame/mananaga/gidd/outputs

echo "Using checkpoint: ${CHECKPOINT_PATH}"
echo "Samples will be saved to: ${SAMPLES_PATH}"
echo "Metrics will be saved to: ${METRICS_PATH}"

# Step 1: Generate samples from the checkpoint
echo "=== Step 1: Generating samples ==="
python gidd/eval/generate_samples.py \
    path="${CHECKPOINT_PATH}" \
    samples_path="${SAMPLES_PATH}" \
    num_samples=1000 \
    num_denoising_steps=128 \
    batch_size=16

# Step 2: Compute generative perplexity
echo "=== Step 2: Computing generative perplexity ==="
python gidd/eval/generative_ppl.py \
    samples_path="${SAMPLES_PATH}" \
    model_tokenizer=gpt2 \
    pretrained_model=google/gemma-2-9b \
    batch_size=4 \
    metrics_path="${METRICS_PATH}"

echo "Evaluation complete! Results saved to: ${METRICS_PATH}"

