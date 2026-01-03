#!/bin/bash
#SBATCH --job-name=duo-lm1b                   
#SBATCH --output=watch_folder/%x_%j.out       
#SBATCH --account=torch_pr_375_cilvr           
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:2                          
#SBATCH --cpus-per-task=4                     
#SBATCH --mem=32GB                            
#SBATCH --time=48:00:00                       
#SBATCH --requeue               # Requeue upon pre-emption

# To enable preemption re-loading, set `hydra.run.dir` or 
# `checkpointing.save_dir` explicitly.
export APPTAINER_CACHEDIR=/scratch/$USER/.apptainer_cache
export APPTAINER_TMPDIR=/scratch/$USER/.apptainer_tmp
export WANDB_API_KEY=63ad8696f69589a33665a9f6134acd8d489e6cca
export WANDB_MODE=online
export WANDB_CACHE_DIR=/scratch/$USER/.wandb_cache
export WANDB_INSECURE_DISABLE_SSL=true

export CURL_CA_BUNDLE=""
export PYTHONHTTPSVERIFY=0
mkdir -p $WANDB_CACHE_DIR

export TRITON_CACHE_DIR=/scratch/$USER/.triton_cache
# rm -rf $TRITON_CACHE_DIR
# mkdir -p $TRITON_CACHE_DIR
export APPTAINERENV_TRITON_CACHE_DIR=$TRITON_CACHE_DIR

export APPTAINERENV_LC_ALL=C
export APPTAINERENV_PATH="/opt/conda/bin:/usr/local/bin:/usr/bin:/bin"

export APPTAINERENV_LD_LIBRARY_PATH="/opt/conda/lib:/usr/local/lib:/usr/lib:$LD_LIBRARY_PATH"

export APPTAINERENV_LIBRARY_PATH=""
export APPTAINERENV_LD_PRELOAD=""



WANDB_NAME="scratch_no_grid_no_boundary_decouple_4_blocks_sample_0.1_prob_mix_residual"
OUTPUT_FILE="watch_folder/${WANDB_NAME}_${SLURM_JOB_ID}.out"
mkdir -p watch_folder
exec > >(tee "$OUTPUT_FILE") 2>&1

/share/apps/apptainer/bin/singularity exec --nv \
  --bind /lib64/libcuda.so.1:/lib/x86_64-linux-gnu/libcuda.so.1 \
  --bind /lib64/libcuda.so.1:/usr/lib64/libcuda.so.1 \
  /scratch/$USER/chan_dos.sif \
  DIT_USE_COMPILE=TRUE \
  python -u -m main \
  loader.batch_size=256 \
  loader.eval_batch_size=256 \
  data=lm1b-wrap \
  data.cache_dir=/scratch/$USER/datasets \
  wandb.project=lm1b_full \
  wandb.name=${WANDB_NAME} \
  model=small \
  training.loss_type=shortcut \
  trainer.max_steps=1500000 \
  trainer.precision=bf16 \
  trainer.val_check_interval=10000 \
  model.length=128 \
  optim.lr=3e-4 \
  algo=dos_shortcut \
  algo.flow_ratio=0.75 \
  algo.shortcut_loss_type=mse \
  algo.flow_warmup=False \
  algo.double_temb=True \
  algo.use_discrete_schedule=True \
  algo.sample_d_on_grid=False \
  algo.use_continuous_shortcut=True \
  algo.add_boundary=False \
  algo.tau_log10_fm=-2.0 \
  algo.tau_log10_shortcut=-2.0 \
  sampling.tau_log10=-1.0 \
  sampling.solver=euler \
  sampling.noise_removal=shortcut \
  algo.scale_loss=False \
  algo.scale_input=False \
  algo.bootstrap_ema=True \
  algo.bootstrap_argmax=False \
  algo.shortcut_k_max=128 \
  algo.n_separated_blocks=4 \
  algo.shortcut_mix_type=residual \
  algo.shortcut_mix_logit=False \
  checkpointing.resume_from_ckpt=False \