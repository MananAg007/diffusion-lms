#!/bin/bash
#SBATCH -J duo-lm1b                   # Job name
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=64000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=anonymous          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --constraint="gpu-mid|gpu-high"
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon pre-emption

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.


run_name="${1:-owt_1k_flow_scratch_mse_log_softmax_output_cd3a94cf}"
python -u -m main \
    algo=duo_finetune \
    seed=42 \
    wandb.project=dos_debug \
    wandb.name=$run_name \
    data=openwebtext-1k \
    eval.compute_generative_perplexity=True \
    loader.batch_size=16 \
    loader.eval_batch_size=16 \
    trainer.max_steps=1000000 \
    trainer.log_every_n_steps=10 \
    trainer.val_check_interval=null \
    training.use_torch_compile=False
    +trainer.check_val_every_n_epoch=2000 \
    model=small \
    model.length=1024 \
    optim.lr=3e-4 \
    callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
    callbacks.checkpoint_every_n_steps.save_top_k=-1 \
    training.loss_type=flow \
    sampling.noise_removal=flow \
    training.pred_type=x0 \
    algo.curriculum_start=0 \
    algo.curriculum_end=50000 \
    algo.flow_warmup_steps=500000 \
    algo.flow_ratio=0.75 \
    algo.sigma_min=0.00001 \
    algo.pred_log_interval=2000 \
    algo.gamma_min=-3.5 \
    algo.gamma_max=-1.75 \
    algo.t_max=0.45 \
    algo.t_min=0.15 \
    algo.gumbel_tau_log10_start=-1.0 \
    algo.gumbel_tau_log10_end=-5.0 \
    algo.use_curriculum=True \
    checkpointing.resume_from_ckpt=False \
    +wandb.offline=False

