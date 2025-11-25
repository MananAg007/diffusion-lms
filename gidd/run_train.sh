#!/bin/bash
#SBATCH --partition=flame
#SBATCH --qos=flame-8gpu-b_qos
#SBATCH --account=aditirag
#SBATCH --job-name=gidd-train
#SBATCH --gres=gpu:8
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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Activate conda environment
source /project/flame/mananaga/miniconda3/etc/profile.d/conda.sh && conda activate gidd

cd /home/mananaga/diffusion-lms/gidd

mkdir -p /project/flame/mananaga/gidd/outputs

torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py \
    --config-name gidd \
    logging.run_name="'small-gidd+-owt-pu=0.0'" \
    logging.save_dir="/project/flame/mananaga/gidd/outputs"

