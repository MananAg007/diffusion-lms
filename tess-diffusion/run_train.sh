#!/bin/bash
#SBATCH --partition=flame
#SBATCH --qos=flame-8gpu-b_qos
#SBATCH --account=aditirag
#SBATCH --job-name=tess-train
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
source /project/flame/mananaga/miniconda3/etc/profile.d/conda.sh && conda activate sdlm

# Change to the project directory
cd /home/mananaga/diffusion-lms/tess-diffusion

# Create directories if they don't exist
mkdir -p /project/flame/mananaga/tess_checkpoints
mkdir -p /project/flame/mananaga/tess_data

# Run training with TESS
# Adjust parameters as needed for your training setup
python -m torch.distributed.launch --nproc_per_node 8 run_mlm.py \
    --model_name_or_path roberta-large \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 6 \
    --do_train \
    --do_eval \
    --output_dir /project/flame/mananaga/tess_checkpoints/opentext_ul2_lr_1e-4_length_256 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --max_seq_length 256 \
    --max_eval_samples 96 \
    --simplex_value 5 \
    --num_diffusion_steps 5000 \
    --num_inference_diffusion_steps 2500 \
    --lr_scheduler_type cosine \
    --learning_rate 1e-4 \
    --pad_to_max_length \
    --beta_schedule squaredcos_improved_ddpm \
    --weight_decay 0.01 \
    --tokenized_data_path processed_data/openwebtext_256_split/ \
    --top_p 0.99 \
    --max_steps 2000000 \
    --gradient_accumulation_steps 8 \
    --warmup_steps 2000 \
    --logging_steps 50 \
    --save_steps 1000 \
    --conditional_generation "ul2"

