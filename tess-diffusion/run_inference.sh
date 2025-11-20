#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --job-name=tess-inference
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/mananaga/logs/%j/.out
#SBATCH --error=/home/mananaga/logs/%j/.out
mkdir -p logs

echo "Working directory: $(pwd)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job submitted from: $SLURM_SUBMIT_DIR"
echo "Running on node: $SLURMD_NODENAME"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

module purge                           # clear any default Intel modules
# re-load only what you really need, e.g.:
module load cuda/12.1   

# Activate conda environment
source /home/mananaga/miniconda/etc/profile.d/conda.sh && conda activate sdlm

# Change to the project directory
cd /home/mananaga/diffusion-lms/tess-diffusion

# Create output directory if it doesn't exist
mkdir -p /data/user_data/mananaga/tess_inference_outputs

# Run inference with MLM (masked language modeling)
# Update model_name_or_path to point to your trained checkpoint
# Using accelerate launch for cleaner inference setup
accelerate launch --config_file configs/accelerate_1_gpu.yaml run_mlm.py \
    --do_eval \
    --output_dir /data/user_data/mananaga/tess_inference_outputs \
    --max_eval_samples 100 \
    --max_seq_length 256 \
    --truncation_length 56 \
    --num_inference_diffusion_steps 1000 \
    --top_p 0.95 \
    --temperature 1.0 \
    --per_device_eval_batch_size 25 \
    --eval_context_size 25 \
    --simplex_value 5 \
    --load_states_in_eval_from_model_path \
    --skip_special_tokens True

