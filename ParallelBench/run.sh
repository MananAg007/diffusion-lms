#!/bin/bash
#SBATCH --job-name=llada_pipeline
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
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
conda activate parallelbench

SCRIPT_DIR=/home/mananaga/diverse-data-synthesis/ParallelBench
cd "$SCRIPT_DIR"

GPU_IDS="0"

#Clean up old LanguageTool Java servers from previous runs
echo "Cleaning up old LanguageTool servers..."
pkill -9 -f "languagetool-server.jar" 2>/dev/null || true
echo "Cleanup complete."
echo ""

BASE_CONFIG="configs/${1}.yaml"
CONFIG_LIST="configs/${1}_list.yaml"
ALL_TASKS=0


# Check if output directory already exists
if [ -d "configs/${1}_list" ]; then
    echo "configs/${1}_list exists; exiting."
    exit 0
fi

# Build config list from base config
echo "Building config list from $BASE_CONFIG..."
if [ "$ALL_TASKS" = "1" ]; then
    echo "Using --all-tasks flag"
    python build_config_list.py "$BASE_CONFIG" --all-tasks
else
    python build_config_list.py "$BASE_CONFIG"
fi

# Run experiments
python run_all.py eval.py \
    --device $GPU_IDS \
    --cfg "$CONFIG_LIST" \
    --logger none \
    --skip_metrics

rm -f "$CONFIG_LIST"


