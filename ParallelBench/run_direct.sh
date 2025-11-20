#!/bin/bash
#SBATCH --job-name=llada_pipeline
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out
#SBATCH --time=48:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -c 4
#SBATCH --hint=nomultithread

export HF_HOME=/data/user_data/mananaga/.hf_cache
export HF_HUB_CACHE=/data/user_data/mananaga/.hf_cache/hub
export HF_DATASETS_CACHE=/data/user_data/mananaga/.hf_cache/datasets
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/.bashrc
conda activate d1

SCRIPT_DIR=/home/mananaga/diverse-data-synthesis/ParallelBench
cd "$SCRIPT_DIR"

GPU_IDS="0"

CONFIG_LIST="configs/${1}_list.yaml"
ALL_TASKS=0

# Run experiments
python run_all.py eval.py \
    --device $GPU_IDS \
    --cfg "$CONFIG_LIST" \
    --logger none \
    --skip_metrics


