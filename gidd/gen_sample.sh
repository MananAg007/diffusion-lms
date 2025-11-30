#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --account=aditirag
#SBATCH --job-name=gen-samples
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

# Checkpoint path
CHECKPOINT_PATH="/home/mananaga/experiments/mugidd-0.5/outputs/latest"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint directory $CHECKPOINT_PATH does not exist"
    exit 1
fi

# Output directory for generated samples
OUTPUT_DIR="/home/mananaga/experiments/mugidd-0.5/outputs/generations"
mkdir -p "$OUTPUT_DIR"

# Activate conda environment
source /project/flame/mananaga/miniconda3/etc/profile.d/conda.sh && conda activate gidd

cd /home/mananaga/diffusion-lms/gidd

# Add current directory to PYTHONPATH
export PYTHONPATH=/home/mananaga/diffusion-lms/gidd:${PYTHONPATH:-}

export HF_HOME=/tmp/mananaga/.cache
export HF_HUB_CACHE=/tmp/mananaga/.cache/hub
export HF_DATASETS_CACHE=/tmp/mananaga/.cache/datasets
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

# Create cache directories if they don't exist
mkdir -p /tmp/mananaga/.cache/hub
mkdir -p /tmp/mananaga/.cache/datasets

echo "Generating samples from checkpoint: $CHECKPOINT_PATH"
echo "Saving samples to: $OUTPUT_DIR"
echo "================================"

# Run the sample generation with intermediate decoding
python gidd/eval/generate_samples.py \
    path="$CHECKPOINT_PATH" \
    num_samples=16 \
    batch_size=16 \
    num_denoising_steps=128 \
    min_p=0.0 \
    samples_path="${OUTPUT_DIR}/samples_with_intermediates.pt" \
    decode_intermediates=true

echo ""
echo "================================"
echo "Sample generation complete!"
echo "Samples saved in: $OUTPUT_DIR"

