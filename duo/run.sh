#!/bin/bash
#SBATCH --partition=flame
#SBATCH --qos=flame-8gpu-b_qos
#SBATCH --account=aditirag
#SBATCH --job-name=duo
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=50
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out
mkdir -p logs

echo "Working directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job submitted from: $SLURM_SUBMIT_DIR"
echo "Running on node: $SLURMD_NODENAME"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Activate conda environment
source ~/miniconda/etc/profile.d/conda.sh && conda activate duo

# Change to the project directory where main.py is located
cd /home/mananaga/diffusion-lms/duo

# Create cache directory if it doesn't exist
mkdir -p /project/flame/mananaga/duo_data
mkdir -p /project/flame/mananaga/duo_checkpoints

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.

python -u -m main \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  loader.num_workers=8 \
  data=openwebtext-split \
  data.cache_dir=/project/flame/mananaga/duo_data \
  checkpointing.save_dir=/project/flame/mananaga/duo_checkpoints \
  callbacks.checkpoint_every_n_steps.save_top_k=0 \
  callbacks.checkpoint_every_n_steps.save_last=false \
  wandb.name=duo-owt \
  model=small \
  algo=duo \
  model.length=1024 \
  algo.gumbel_tau_log10_start=-3.0 \
  algo.gumbel_tau_log10_end=-3.0 \
  algo.gamma_min=-3.55 \
  algo.gamma_max=-1.85 \
  algo.curriculum_start=0 \
  algo.curriculum_end=500000
