#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --account=aditirag
#SBATCH --job-name=eval-ppl
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

# Hardcoded path to generations directory
GENERATIONS_DIR="/home/mananaga/experiments/gidd-baseline/outputs/generations"

if [ ! -d "$GENERATIONS_DIR" ]; then
    echo "Error: Directory $GENERATIONS_DIR does not exist"
    exit 1
fi

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

# Create metrics directory
METRICS_DIR="${GENERATIONS_DIR}/metrics"
mkdir -p "$METRICS_DIR"

echo "Evaluating all .pt files in: $GENERATIONS_DIR"
echo "Saving metrics to: $METRICS_DIR"
echo "================================"

# Loop through all .pt files in the directory
for sample_file in "$GENERATIONS_DIR"/*.pt; do
    if [ -f "$sample_file" ]; then
        filename=$(basename "$sample_file" .pt)
        metrics_file="${METRICS_DIR}/${filename}_metrics.json"
        
        echo ""
        echo "Processing: $filename"
        echo "Output: $metrics_file"
        
        python gidd/eval/generative_ppl.py \
            samples_path="$sample_file" \
            model_tokenizer=gpt2 \
            pretrained_model=google/gemma-2-9b \
            batch_size=8 \
            metrics_path="$metrics_file" \
            torch_compile=false
        
        echo "✓ Completed: $filename"
    fi
done

echo ""
echo "================================"
echo "All evaluations complete!"
echo "Metrics saved in: $METRICS_DIR"

# Create a summary CSV file
SUMMARY_FILE="${METRICS_DIR}/summary.csv"
echo "file,pretrained_model,median_nll,avg_nll,total_ppl,ppl_mean,ppl_std,ppl_median,ppl_min,ppl_max,acc,tokens" > "$SUMMARY_FILE"

for metrics_file in "$METRICS_DIR"/*_metrics.json; do
    if [ -f "$metrics_file" ]; then
        python -c "
import json
import sys
with open('$metrics_file', 'r') as f:
    m = json.load(f)
    print(f\"{m['file']},{m['pretrained_model']},{m['median_nll']},{m['avg_nll']},{m['total_ppl']},{m['ppl_mean']},{m['ppl_std']},{m['ppl_median']},{m['ppl_min']},{m['ppl_max']},{m['acc']},{m['tokens']}\")
" >> "$SUMMARY_FILE"
    fi
done

echo ""
echo "Summary CSV created: $SUMMARY_FILE"
cat "$SUMMARY_FILE"

