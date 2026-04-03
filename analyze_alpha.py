import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from config import GPTConfig
from model import GPT
from tokenizer import Tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference and plot exclusive self-attention alpha distributions."
    )
    parser.add_argument("--ckpt_path", type=str, default="out/ckpt.pt")
    parser.add_argument("--tokenizer_path", type=str, default=os.path.join("data", "tok4096.model"))
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--out_dir", type=str, default="out/alpha_analysis")
    parser.add_argument("--bins", type=int, default=80)
    return parser.parse_args()


def setup_dtype_ctx(device: str, dtype_name: str):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_name]
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type="cpu", dtype=dtype, enabled=False)


def load_model(ckpt_path: str, device: str):
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def apply_sampling_filters(logits, top_k=None, top_p=None, min_p=None):
    if top_p is not None and top_p > 0.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cumulative_probs >= top_p
        mask[..., 0] = True
        cutoff_indices = mask.int().argmax(dim=-1, keepdim=True)

        top_p_mask = torch.zeros_like(logits, dtype=torch.bool)
        for b in range(logits.size(0)):
            cut = cutoff_indices[b].item()
            keep_indices = sorted_indices[b, : cut + 1]
            top_p_mask[b, keep_indices] = True
        logits[~top_p_mask] = float("-inf")

    if top_k is not None:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = float("-inf")

    if min_p is not None and min_p > 0.0:
        logit_max = logits.max(dim=-1, keepdim=True).values
        threshold = logit_max + torch.log(
            torch.tensor(min_p, device=logits.device, dtype=logits.dtype)
        )
        logits[logits < threshold] = float("-inf")

    return logits


@torch.no_grad()
def generate_and_capture(model, idx, max_new_tokens, temperature, top_k, top_p, min_p, alpha_store):
    for _ in range(max_new_tokens):
        context = idx if idx.size(1) < model.config.block_size else idx[:, -model.config.block_size :]
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature
        logits = apply_sampling_filters(logits, top_k=top_k, top_p=top_p, min_p=min_p)

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        if idx_next.item() == 2:
            break
        idx = torch.cat([idx, idx_next], dim=-1)

    for layer_idx in list(alpha_store.keys()):
        if len(alpha_store[layer_idx]) > 0:
            alpha_store[layer_idx] = torch.cat(alpha_store[layer_idx], dim=0)
        else:
            alpha_store[layer_idx] = torch.empty(0)
    return idx


def save_plots(alpha_store, out_dir, bins):
    os.makedirs(out_dir, exist_ok=True)
    stats = {}

    for layer_idx, alpha_values in alpha_store.items():
        if alpha_values.numel() == 0:
            stats[f"layer_{layer_idx}"] = {
                "count": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
            continue

        alpha_np = alpha_values.numpy()
        stats[f"layer_{layer_idx}"] = {
            "count": int(alpha_values.numel()),
            "mean": float(alpha_values.mean().item()),
            "std": float(alpha_values.std(unbiased=False).item()),
            "min": float(alpha_values.min().item()),
            "max": float(alpha_values.max().item()),
        }

        plt.figure(figsize=(8, 5))
        plt.hist(alpha_np, bins=bins, range=(0.0, 1.0), density=True, alpha=0.85)
        plt.title(f"Alpha Distribution - Layer {layer_idx}")
        plt.xlabel("alpha")
        plt.ylabel("density")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"alpha_layer_{layer_idx}.png"), dpi=150)
        plt.close()

    stats_path = os.path.join(out_dir, "alpha_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = load_model(args.ckpt_path, args.device)

    if not model.config.use_exclusive_self_attention:
        raise ValueError("Checkpoint config has use_exclusive_self_attention=False; no alpha to capture.")

    tokenizer = Tokenizer(args.tokenizer_path)
    input_ids = torch.tensor(
        tokenizer.encode(args.prompt, bos=True, eos=False),
        dtype=torch.long,
        device=args.device,
    ).unsqueeze(0)

    alpha_store = {}
    hooks = []
    for layer_idx, block in enumerate(model.transformer.h):
        alpha_store[layer_idx] = []
        if not hasattr(block.attn, "exclusive_gate"):
            continue

        def _make_hook(idx):
            def _hook(_module, _inputs, output):
                alpha_store[idx].append(output.detach().float().reshape(-1).cpu())

            return _hook

        hooks.append(block.attn.exclusive_gate.register_forward_hook(_make_hook(layer_idx)))

    ctx = setup_dtype_ctx(args.device, args.dtype)
    with torch.no_grad():
        with ctx:
            output_ids = generate_and_capture(
                model,
                input_ids,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.top_p,
                args.min_p,
                alpha_store,
            )

    for hook in hooks:
        hook.remove()

    save_plots(alpha_store, args.out_dir, args.bins)

    print("Generated text:")
    print(tokenizer.decode(output_ids[0].tolist()))
    print("-")
    print(f"Saved per-layer alpha plots and stats to: {args.out_dir}")


if __name__ == "__main__":
    main()