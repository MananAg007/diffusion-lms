import json
from pathlib import Path

import hydra
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@hydra.main(config_path="../configs", config_name="gen_ppl", version_base="1.1")
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision('high')
    torch.set_grad_enabled(False)

    model_tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer)

    print(f"Loding model {args.pretrained_model}")

    model = AutoModelForCausalLM.from_pretrained(args.pretrained_model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.torch_compile:
        model = torch.compile(model)

    samples_path = hydra.utils.to_absolute_path(args.samples_path)
    samples_data = torch.load(samples_path, weights_only=False)
    
    # Handle both old format (tensor) and new format (dict)
    if isinstance(samples_data, dict):
        # New format: dictionary with 'token_ids' and 'texts'
        if 'texts' in samples_data:
            texts = samples_data['texts']
        else:
            z_ts = samples_data['token_ids']
            texts = model_tokenizer.batch_decode(z_ts, skip_special_tokens=True)
    else:
        # Old format: just a tensor
        z_ts = samples_data
        # fix for bug in self-correct script:
        if z_ts.shape[1] == 1:
            z_ts = z_ts.squeeze(1)
        texts = model_tokenizer.batch_decode(z_ts, skip_special_tokens=True)

    total_acc = 0
    total_nll = 0
    total_tokens = 0
    all_nlls = []
    per_sample_ppls = []
    per_sample_nlls = []
    
    with torch.no_grad():
        for i in tqdm.trange(0, len(texts), args.batch_size, desc="Inference", dynamic_ncols=True):
            xs = texts[i:i + args.batch_size]

            batch = tokenizer(xs, padding=True, return_tensors="pt", truncation=True, max_length=512).to(device)
            attn_mask = batch["attention_mask"]
        
            logits = model(input_ids=batch["input_ids"], attention_mask=attn_mask, use_cache=False).logits[:, :-1]

            labels = batch["input_ids"][:, 1:]
            loss_mask = attn_mask[:, :-1]

            nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(0, 1), reduction='none').view_as(labels)
            all_nlls.extend(nll[loss_mask == 1].cpu().numpy().tolist())
            total_nll += (nll * loss_mask).sum().item()

            acc = (logits.argmax(-1) == labels).float()
            total_acc += (acc * loss_mask).sum().item()

            total_tokens += loss_mask.sum().item()
            
            # Compute per-sample PPL
            for j in range(len(xs)):
                sample_mask = loss_mask[j]
                sample_nll = nll[j]
                sample_tokens = sample_mask.sum().item()
                
                if sample_tokens > 0:
                    sample_avg_nll = (sample_nll * sample_mask).sum().item() / sample_tokens
                    per_sample_nlls.append(sample_avg_nll)
                    per_sample_ppls.append(np.exp(sample_avg_nll))


    # Corpus-level metrics (original)
    corpus_nll = total_nll / total_tokens
    corpus_ppl = np.exp(total_nll / total_tokens)
    acc = total_acc / total_tokens
    
    # Per-sample statistics
    per_sample_ppls = np.array(per_sample_ppls)
    per_sample_nlls = np.array(per_sample_nlls)

    metrics = {
        "file": Path(args.samples_path).stem,
        "pretrained_model": args.pretrained_model,
        "median_nll": np.median(all_nlls),
        "avg_nll": corpus_nll,
        "total_ppl": corpus_ppl,
        "acc": acc,
        "tokens": total_tokens,
        # Per-sample PPL statistics
        "ppl_mean": float(np.mean(per_sample_ppls)),
        "ppl_std": float(np.std(per_sample_ppls)),
        "ppl_median": float(np.median(per_sample_ppls)),
        "ppl_min": float(np.min(per_sample_ppls)),
        "ppl_max": float(np.max(per_sample_ppls)),
        "ppl_q25": float(np.percentile(per_sample_ppls, 25)),
        "ppl_q75": float(np.percentile(per_sample_ppls, 75)),
        # Per-sample NLL statistics
        "nll_mean": float(np.mean(per_sample_nlls)),
        "nll_std": float(np.std(per_sample_nlls)),
    }

    print(json.dumps(metrics, indent=4))
    print("=== RESULTS ===")
    print(f"Corpus PPL: {metrics['total_ppl']:.2f}")
    print(f"Per-sample PPL: {metrics['ppl_mean']:.2f} ± {metrics['ppl_std']:.2f} (std)")
    print(f"  Min: {metrics['ppl_min']:.2f}, Max: {metrics['ppl_max']:.2f}, Median: {metrics['ppl_median']:.2f}")
    print("===============")

    with open(hydra.utils.to_absolute_path(args.metrics_path), "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
