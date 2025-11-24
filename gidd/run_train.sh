#!/bin/bash
#SBATCH --partition=flame
#SBATCH --qos=flame-8gpu-b_qos
#SBATCH --account=aditirag
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

# Change to the project directory
cd /home/mananaga/gidd

# Create output directory for checkpoints
mkdir -p ./outputs

# (Optional) Login to wandb for experiment tracking
# Uncomment the line below if you want to use Weights & Biases
# wandb login

# Or disable wandb if you don't want to use it
wandb disabled

# Run training with GIDD+
# Default configuration: p_u = 0.0 (mask-only baseline)
# Adjust p_uniform parameter for hybrid noise (e.g., 0.1, 0.2)
torchrun --nnodes 1 --nproc_per_node 4 gidd/train.py \
    --config-name gidd \
    logging.run_name="'small-gidd+-owt-pu=0.0'"

# Alternative configurations (uncomment to use):

# GIDD+ with p_u = 0.1 (hybrid: 10% uniform noise)
# torchrun --nnodes 1 --nproc_per_node 4 gidd/train.py \
#     --config-name gidd \
#     model.p_uniform=0.1 \
#     logging.run_name="'small-gidd+-owt-pu=0.1'"

# GIDD+ with p_u = 0.2 (hybrid: 20% uniform noise)
# torchrun --nnodes 1 --nproc_per_node 4 gidd/train.py \
#     --config-name gidd \
#     model.p_uniform=0.2 \
#     logging.run_name="'small-gidd+-owt-pu=0.2'"

# MDLM baseline
# torchrun --nnodes 1 --nproc_per_node 4 gidd/train.py \
#     --config-name mdlm \
#     logging.run_name="'small-mdlm-owt'"

# AR baseline
# torchrun --nnodes 1 --nproc_per_node 4 gidd/train.py \
#     --config-name ar \
#     logging.run_name="'small-ar-owt'"

