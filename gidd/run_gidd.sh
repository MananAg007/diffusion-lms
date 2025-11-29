#!/bin/bash
#SBATCH --partition=flame
#SBATCH --qos=flame-8gpu-b_qos
#SBATCH --account=aditirag
#SBATCH --job-name=gidd-train
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
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

# Add current directory to PYTHONPATH so Python can find the gidd module
export PYTHONPATH=/home/mananaga/diffusion-lms/gidd:${PYTHONPATH:-}

export HF_HOME=/project/flame/mananaga/.hf_cache
export HF_HUB_CACHE=/project/flame/mananaga/.hf_cache/hub
export HF_DATASETS_CACHE=/project/flame/mananaga/.hf_cache/datasets
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_TOKEN=$(cat ~/.cache/huggingface/token)

mkdir -p /project/flame/mananaga/gidd/outputs

torchrun --nnodes 1 --nproc_per_node 8 gidd/train.py \
    --config-name gidd \
    logging.run_name="'small-gidd+-owt-pu=0.0'" \
    logging.save_dir="/project/flame/mananaga/gidd/outputs" \
    logging.wandb_entity="diffusion-lms" \
    logging.gen_ppl_enabled=true \
    logging.gen_ppl_freq=10000 \
    logging.gen_ppl_num_samples=800 \
    logging.gen_ppl_num_denoising_steps=128 \
    logging.gen_ppl_batch_size=16 \
    logging.gen_ppl_reference_model="google/gemma-2-9b" \
    data.num_workers=2

