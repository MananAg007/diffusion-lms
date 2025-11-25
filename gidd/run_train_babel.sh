#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --job-name=gidd-train
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out
mkdir -p /home/mananaga/logs/%j

echo "Working directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job submitted from: $SLURM_SUBMIT_DIR"
echo "Running on node: $SLURMD_NODENAME"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Activate conda environment
source /home/mananaga/miniconda/etc/profile.d/conda.sh && conda activate gidd

cd /home/mananaga/diffusion-lms/gidd

# Add current directory to PYTHONPATH so Python can find the gidd module
export PYTHONPATH=/home/mananaga/diffusion-lms/gidd:${PYTHONPATH:-}

# Set HuggingFace cache directory to avoid corrupted cache issues
# HF_HOME is the main cache directory that transformers and datasets will use
export HF_HOME=/data/user_data/mananaga/.hf_cache
export HF_HUB_CACHE=/data/user_data/mananaga/.hf_cache/hub
export HF_DATASETS_CACHE=/data/user_data/mananaga/.hf_cache/datasets
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

mkdir -p /data/user_data/mananaga/gidd/outputs

torchrun --nnodes 1 --nproc_per_node 4 gidd/train.py \
    --config-name gidd \
    logging.run_name="'small-gidd+-owt-pu=0.0'" \
    logging.save_dir="/data/user_data/mananaga/gidd/outputs"

