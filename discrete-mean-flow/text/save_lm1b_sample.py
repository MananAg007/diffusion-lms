#!/usr/bin/env python3
"""Save a single lm1b sample to npy file for debugging.

This mimics exactly how `main.py` loads the config via Hydra and how
`train_lm1b_dos.sh` overrides it on the command line.
"""

import os
import numpy as np
import torch
import dataloader
import hydra
import omegaconf


# Register the same OmegaConf resolvers that main.py uses
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)


def build_config() -> omegaconf.DictConfig:
    """Compose the config exactly like:

    python -m main \
      loader.global_batch_size=512 \
      loader.batch_size=64 \
      loader.eval_batch_size=64 \
      data=lm1b-wrap \
      wandb.project=lm1b_full \
      wandb.name=lm1b_full_flow \
      model=small \
      algo=dos \
      trainer.max_steps=1500000 \
      trainer.precision=bf16 \
      trainer.val_check_interval=20000 \
      model.length=128 \
      optim.lr=3e-4 \
      algo.double_temb=False \
      algo.use_discrete_schedule=True \
      algo.gumbel_tau_log10_start=-1.0 \
      algo.gumbel_tau_log10_end=-2.0 \
      algo.curriculum_start=0 \
      algo.curriculum_end=25000 \
      sampling.noise_removal=uniform \
      algo.time_condition=alpha_t \
      algo.scale_input=False \
      algo.scale_loss=False \
      checkpointing.resume_from_ckpt=False
    """

    overrides = [
        "loader.global_batch_size=512",
        "loader.batch_size=64",
        "loader.eval_batch_size=64",
        "data=lm1b-wrap",
        "wandb.project=lm1b_full",
        "wandb.name=lm1b_full_flow",
        "model=small",
        "algo=dos",
        "trainer.max_steps=1500000",
        "trainer.precision=bf16",
        "trainer.val_check_interval=20000",
        "model.length=128",
        "optim.lr=3e-4",
        "algo.double_temb=False",
        "algo.use_discrete_schedule=True",
        "algo.gumbel_tau_log10_start=-1.0",
        "algo.gumbel_tau_log10_end=-2.0",
        "algo.curriculum_start=0",
        "algo.curriculum_end=25000",
        "sampling.noise_removal=uniform",
        "algo.time_condition=alpha_t",
        "algo.scale_input=False",
        "algo.scale_loss=False",
        "checkpointing.resume_from_ckpt=False",
    ]

    # Compose the config in the same way as @hydra.main(config_path="configs", config_name="config")
    with hydra.initialize(version_base=None, config_path="configs"):
        cfg = hydra.compose(config_name="config", overrides=overrides)
    return cfg


def main() -> None:
    # 1) Build config exactly like training
    config = build_config()

    # 2) Get tokenizer & train dataloader exactly like main.py/_train
    tokenizer = dataloader.get_tokenizer(config)
    train_ds, _ = dataloader.get_dataloaders(config, tokenizer, skip_valid=True)

    # 3) Take a single batch from train loader and grab one sequence
    print("Fetching one batch from lm1b train dataloader...")
    batch = next(iter(train_ds))
    x0 = batch["input_ids"][0]  # shape: [seq_len]

    if isinstance(x0, torch.Tensor):
        x0 = x0.cpu().numpy()
    else:
        x0 = np.array(x0)

    # Ensure shape is (1, seq_len) like owt_1sample_data/x0.npy
    if x0.ndim == 1:
        x0 = x0.reshape(1, -1)
    elif x0.ndim == 2 and x0.shape[0] != 1:
        x0 = x0[:1]

    print(f"Sample shape: {x0.shape}")
    print(f"Sample dtype: {x0.dtype}")
    print(f"Sample preview: {x0[0, :20]}")
    import ipdb;ipdb.set_trace()
    print(tokenizer.decode(x0[0]))
    # 4) Save to npy file under lm1b_1sample_data/x0.npy
    out_dir = "lm1b_1sample_data"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "x0.npy")
    np.save(out_path, x0)
    print(f"Saved lm1b sample to: {out_path}")

    # Simple verification
    loaded = np.load(out_path)
    print(f"Verified reload: shape={loaded.shape}, dtype={loaded.dtype}")


if __name__ == "__main__":
    main()
