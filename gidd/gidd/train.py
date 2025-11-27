import datetime
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import hydra
import tqdm
import wandb
from omegaconf import OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

from gidd.models.dit import DIT
from gidd.checkpoints import (
    save_checkpoint,
    load_checkpoint_for_training,
    TrainingState,
    save_rng_state,
    load_rng_state,
)
from gidd.diffusion_process import get_noise_schedule
from gidd.modeling import get_tokenizer, get_model
from gidd.data import get_dataloaders
from gidd.loss import get_loss
from gidd.trainer import get_trainer
from gidd.optimizer import get_optimizer
from gidd.sampling import get_sampler
from gidd.utils import (
    get_lr,
    parse_dtype,
    calculate_flops_per_batch,
)


class Logger:
    def __init__(self, is_main_process):
        self.is_main_process = is_main_process

    def init(self, *args, **kwargs):
        if self.is_main_process:
            wandb.init(*args, **kwargs)

    def log(self, *args, **kwargs):
        if self.is_main_process:
            wandb.log(*args, **kwargs)


def compute_generative_ppl(sampler, model_tokenizer, reference_model, reference_tokenizer, config, dtype, device, is_main_process):
    """Generate samples and compute generative perplexity."""
    if not is_main_process:
        return {}
    
    try:
        print(f"[Gen PPL] Generating {config.logging.gen_ppl_num_samples} samples...")
        samples = []
        max_length = config.model.max_seq_len
        
        with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
            for i in range(0, config.logging.gen_ppl_num_samples, config.logging.gen_ppl_batch_size):
                bs = min(config.logging.gen_ppl_batch_size, config.logging.gen_ppl_num_samples - i)
                z_t = sampler.generate(
                    bs, 
                    config.logging.gen_ppl_num_denoising_steps, 
                    max_length=max_length, 
                    decode=False, 
                    show_progress=False
                )
                samples.append(z_t)
        
        samples = torch.cat(samples, dim=0)
        texts = model_tokenizer.batch_decode(samples, skip_special_tokens=True)
        
        print(f"[Gen PPL] Computing perplexity using {config.logging.gen_ppl_reference_model}...")
        total_nll = 0
        total_tokens = 0
        total_acc = 0
        all_nlls = []
        
        with torch.no_grad():
            for i in range(0, len(texts), config.logging.gen_ppl_batch_size):
                xs = texts[i:i + config.logging.gen_ppl_batch_size]
                
                batch = reference_tokenizer(
                    xs, 
                    padding=True, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512
                ).to(device)
                attn_mask = batch["attention_mask"]
                
                logits = reference_model(
                    input_ids=batch["input_ids"], 
                    attention_mask=attn_mask, 
                    use_cache=False
                ).logits[:, :-1]
                
                labels = batch["input_ids"][:, 1:]
                loss_mask = attn_mask[:, :-1]
                
                nll = F.cross_entropy(
                    logits.flatten(0, 1), 
                    labels.flatten(0, 1), 
                    reduction='none'
                ).view_as(labels)
                
                all_nlls.extend(nll[loss_mask == 1].cpu().numpy().tolist())
                total_nll += (nll * loss_mask).sum().item()
                
                acc = (logits.argmax(-1) == labels).float()
                total_acc += (acc * loss_mask).sum().item()
                
                total_tokens += loss_mask.sum().item()
        
        nll = total_nll / total_tokens
        ppl = np.exp(nll)
        acc = total_acc / total_tokens
        
        metrics = {
            "gen_ppl/nll": nll,
            "gen_ppl/ppl": ppl,
            "gen_ppl/acc": acc,
            "gen_ppl/tokens": total_tokens,
            "gen_ppl/median_nll": np.median(all_nlls),
        }
        
        print(f"[Gen PPL] Results: PPL={ppl:.2f}, NLL={nll:.4f}, Acc={acc:.4f}")
        return metrics
        
    except Exception as e:
        print(f"[Gen PPL] Error during generative ppl computation: {e}")
        import traceback
        traceback.print_exc()
        return {}


@contextmanager
def main_process_first():
    if dist.is_initialized():
        if dist.get_rank() == 0:
            yield
            dist.barrier()
        else:
            dist.barrier()
            yield
    else:
        yield


@hydra.main(config_path="configs", config_name="gidd", version_base="1.1")
def main(config):
    try:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(minutes=30),
            device_id=torch.device("cuda", local_rank),
        )
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()  # only a single group, don't have to worry about local vs. global rank
        is_main_process = (global_rank == 0)
    except RuntimeError:
        print("Distributed training not available, running on single device.")
        world_size = 1
        local_rank = 0
        global_rank = 0
        is_main_process = True
    with open_dict(config):
        config.training.world_size = world_size

    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    seed = config.training.seed + global_rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.backends.cuda.enable_flash_sdp(enabled=True)

    dtype = parse_dtype(config.training.dtype)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device=} and {dtype=}")

    if config.training.resume is None:
        tokenizer = get_tokenizer(config)

        model = get_model(config, tokenizer, dtype=dtype)
        noise_schedule = get_noise_schedule(config, tokenizer)
        loss_fn = get_loss(config, tokenizer, noise_schedule)
        trainer = get_trainer(config, model, tokenizer, noise_schedule, loss_fn, dtype)
        trainer = trainer.to(device)

        optimizer = get_optimizer(config, trainer)

        state = TrainingState(
            epoch=0,
            epoch_start_step=0,
            step=0,
        )
    else:
        (
            model,
            noise_schedule,
            tokenizer,
            old_config,
            trainer,
            optimizer,
            state
        ) = load_checkpoint_for_training(config.training.resume, device=device, dtype=dtype)

    with main_process_first():
        train_dl, test_dl = get_dataloaders(config, tokenizer)

    max_lr = config.optimizer.lr

    logger = Logger(is_main_process)
    logger.init(
        name=config.logging.run_name,
        entity=config.logging.wandb_entity,
        project=config.logging.wandb_project,
        config=OmegaConf.to_container(config, resolve=True),
    )

    if is_main_process:
        pwd = Path(".").resolve()
        wandb.config.update({"pwd": pwd})
        print(f"Working directory: {pwd}")

    # Initialize generative PPL evaluation if enabled
    reference_model = None
    reference_tokenizer = None
    sampler = None
    if config.logging.gen_ppl_enabled and is_main_process:
        print(f"[Gen PPL] Loading reference model: {config.logging.gen_ppl_reference_model}")
        try:
            reference_tokenizer = AutoTokenizer.from_pretrained(config.logging.gen_ppl_reference_model)
            if reference_tokenizer.pad_token_id is None:
                reference_tokenizer.pad_token = reference_tokenizer.eos_token
            reference_model = AutoModelForCausalLM.from_pretrained(
                config.logging.gen_ppl_reference_model, 
                device_map="auto"
            )
            reference_model.eval()
            print(f"[Gen PPL] Reference model loaded successfully")
            
            # Create sampler for generation
            sampler = get_sampler(config, model, tokenizer, noise_schedule, compile_step=False, min_p=0.0)
            print(f"[Gen PPL] Sampler initialized")
        except Exception as e:
            print(f"[Gen PPL] Failed to load reference model: {e}")
            print("[Gen PPL] Disabling generative PPL evaluation")
            with open_dict(config):
                config.logging.gen_ppl_enabled = False

    if isinstance(model, DIT):
        non_emb_params = sum(p.numel() for p in model.blocks.parameters())
    else:  # Llama
        non_emb_params = sum(p.numel() for p in model.model.layers.parameters())

    flops_per_batch = calculate_flops_per_batch(config, model, len(tokenizer), non_emb_params, method="hoffmann")

    trainable_params = sum(p.numel() for p in trainer.parameters() if p.requires_grad)

    if config.training.compile_model:
        opt_trainer = torch.compile(trainer)
    else:
        opt_trainer = trainer

    if is_distributed:
        ddp_trainer = DDP(opt_trainer, device_ids=[device.index])
    else:
        ddp_trainer = opt_trainer

    if is_main_process:
        non_emb_params_str = f"{non_emb_params / 1e6:.1f}M" if non_emb_params < 500 * 1e6 else f"{non_emb_params / 1e9:.1f}B"
        trainable_params_str = f"{trainable_params / 1e6:.1f}M" if trainable_params < 500 * 1e6 else f"{trainable_params / 1e9:.1f}B"
        print(f"*** Starting training ***")
        print(f"* World size: {world_size}")
        print(f"* FLOPS per batch: {flops_per_batch:.3g}")
        print(f"* Per-device batch size: {config.training.train_batch_size}")
        print(f"* Total batch size: {config.training.train_batch_size * world_size}")
        print(f"* Non-embedding parameters: {non_emb_params_str}")
        print(f"* Trainable parameters: {trainable_params_str}")
        print(f"* Model dtype: {next(iter(model.parameters())).dtype}")
        print(f"*************************")

    if is_distributed and hasattr(train_dl.sampler, "set_epoch"):
        train_dl.sampler.set_epoch(state.epoch)
    batch_iterator = iter(train_dl)

    # initialize eval dataloader to prevent new processes getting started during training
    # (without this crashes can occur if the code changes before the first eval step)
    _ = next(iter(test_dl))

    # for resuming training, skip the batches that were already trained on
    if state.step - state.epoch_start_step > 0:
        for _ in tqdm.trange(state.step - state.epoch_start_step, desc="Skipping batches", dynamic_ncols=True, disable=not is_main_process):
            next(batch_iterator)

    curr_time = time.time()
    # adjust start time in case we're resuming training
    trained_time = 0 if config.training.resume is None else (state.start_time - state.curr_time)
    state.start_time = curr_time - trained_time
    state.curr_time = curr_time
    prev_time = curr_time

    log_buffer = []

    if config.training.resume is not None:
        load_rng_state(config.training.resume, global_rank)

    with tqdm.tqdm(total=config.training.num_train_steps, initial=state.step, desc="Training", dynamic_ncols=True, disable=not is_main_process) as pbar:
        for step in range(state.step, config.training.num_train_steps):
                
            ### TRAIN ###

            try:
                batch = next(batch_iterator)
            except StopIteration:
                state.epoch += 1
                state.epoch_start_step = step
                if is_distributed and hasattr(train_dl.sampler, "set_epoch"):
                    train_dl.sampler.set_epoch(state.epoch)
                batch_iterator = iter(train_dl)
                batch = next(batch_iterator)

            curr_lr = get_lr(config, max_lr, step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = curr_lr

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss, metrics = ddp_trainer(batch)

            (loss * config.loss.loss_scale).backward()

            if config.optimizer.grad_clip_norm and config.optimizer.grad_clip_norm > 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip_norm)
            else:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e6)

            optimizer.step()
            optimizer.zero_grad()

            batch_tokens = batch["attention_mask"].sum().item() * config.training.world_size
            batch_flops = flops_per_batch * config.training.world_size
            total_batch_size = batch["input_ids"].size(0) * config.training.world_size
            state.total_tokens += batch_tokens
            state.total_flops += batch_flops

            curr_time = time.time()
            step_time = curr_time - prev_time
            prev_time = curr_time

            # no need to all_reduce metrics since these are not that important
            log_buffer.append({
                "train/loss": loss.item(),
                "train/lr": curr_lr,
                "train/step": step + 1,
                "train/grad_norm": norm.item(),
                "train/epoch": step / len(train_dl),
                "train/total_tokens": state.total_tokens,
                "train/total_flops": state.total_flops,
                "train/tokens_per_sec": batch_tokens / step_time,
                "train/flops_per_sec": batch_flops / step_time,
                "train/samples_per_sec": total_batch_size / step_time,
                "train/it_per_sec": 1 / step_time,
                "train/avg_it_per_sec": (step + 1) / (curr_time - state.start_time),
                **{f"train/{k}": v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()},
            })

            if ((step + 1) % config.logging.log_freq) == 0:
                metrics = {k: sum(d[k] for d in log_buffer) / len(log_buffer) for k in log_buffer[0]}
                logger.log({k: v for k, v in metrics.items()}, step=step)
                logger.log({"trainer/global_step": step}, step=step)
                log_buffer = []

            ### EVAL ###

            if ((step + 1) % config.logging.eval_freq) == 0:
                with torch.no_grad():
                    eval_start_time = time.time()
                    model.eval()

                    eval_metrics = {}
                    eval_loss = 0
                    num_eval_samples = 0
                    for i, test_batch in enumerate(tqdm.tqdm(test_dl, desc="Eval", dynamic_ncols=True, total=config.logging.num_eval_batches, disable=not is_main_process)):
                        bs = test_batch["input_ids"].size(0)

                        test_batch = {k: v.to(device, non_blocking=True) for k, v in test_batch.items()}
                        loss, metrics = ddp_trainer(test_batch)

                        for k, v in metrics.items():
                            eval_metrics[k] = eval_metrics.get(k, 0) + (v.item() if isinstance(v, torch.Tensor) else v) * bs

                        eval_loss += loss.item() * bs
                        num_eval_samples += bs

                        if i >= config.logging.num_eval_batches - 1:
                            break

                    # Compute validation perplexity from loss
                    eval_loss_avg = eval_loss / num_eval_samples
                    eval_ppl = np.exp(eval_loss_avg)

                    dist.barrier()

                    eval_elapsed_time = time.time() - eval_start_time
                    logger.log({
                        "eval/loss": eval_loss_avg,
                        "eval/ppl": eval_ppl,
                        "eval/time_taken": eval_elapsed_time,
                        **{f"eval/{k}": v / num_eval_samples for k, v in eval_metrics.items()},
                    }, step=step)
                    model.train()

            ### SAVE ###

            # increment step before saving so that resuming from the checkpoint will start at the next step
            state.step += 1
            if ((step + 1) % config.logging.save_freq) == 0:
                dist.barrier()
                output_path = Path(config.logging.save_dir, "latest")
                if is_main_process:
                    save_checkpoint(output_path, trainer, optimizer, state)
                dist.barrier()
                output_path.mkdir(exist_ok=True, parents=True)
                save_rng_state(output_path, global_rank)
                dist.barrier()

            ### GENERATIVE PPL ###

            if config.logging.gen_ppl_enabled and ((step + 1) % config.logging.gen_ppl_freq) == 0:
                if is_distributed:
                    dist.barrier()
                
                if is_main_process:
                    print(f"\n[Gen PPL] Starting generative perplexity evaluation at step {step + 1}")
                    gen_ppl_start_time = time.time()
                    
                    # Set model to eval mode
                    model.eval()
                    
                    # Compute generative PPL
                    gen_ppl_metrics = compute_generative_ppl(
                        sampler=sampler,
                        model_tokenizer=tokenizer,
                        reference_model=reference_model,
                        reference_tokenizer=reference_tokenizer,
                        config=config,
                        dtype=dtype,
                        device=device,
                        is_main_process=is_main_process,
                    )
                    
                    gen_ppl_elapsed_time = time.time() - gen_ppl_start_time
                    gen_ppl_metrics["gen_ppl/time_taken"] = gen_ppl_elapsed_time
                    
                    # Log to wandb
                    logger.log(gen_ppl_metrics, step=step)
                    
                    # Set model back to train mode
                    model.train()
                    
                    print(f"[Gen PPL] Completed in {gen_ppl_elapsed_time:.2f}s\n")
                
                if is_distributed:
                    dist.barrier()

            pbar.update(1)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
