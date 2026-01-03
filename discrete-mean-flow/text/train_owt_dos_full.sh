DIT_USE_COMPILE=TRUE

python main.py \
  loader.global_batch_size=512 \
  loader.batch_size=32 \
  loader.eval_batch_size=16 \
  data=openwebtext-split \
  wandb.name=owt_full_flow_cond_alpha_t_sample_alpha_t_1-t_scale_cont \
  model=small \
  algo=dos \
  model.length=1024 \
  trainer.max_steps=1500000 \
  trainer.precision=bf16 \
  optim.lr=3e-4 \
  trainer.val_check_interval=30000 \
  algo.double_temb=False \
  algo.use_discrete_schedule=True \
  +wandb.offline=False \
  algo.gumbel_tau_log10_start=-1.0 \
  algo.gumbel_tau_log10_end=-2.0 \
  algo.curriculum_end=25000 \
  sampling.noise_removal=uniform \
  algo.time_condition=alpha_t \
  training.use_torch_compile=True \
  algo.scale_input=False \
  algo.scale_loss=False \
  checkpointing.resume_from_ckpt=False \
  