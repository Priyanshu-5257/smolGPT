#!/usr/bin/env python3
"""Run architecture-comparison training jobs for smolGPT.

Designed for a single larger GPU server: about 10 CPU cores, 60GB RAM, and
48GB GPU VRAM. It runs a shared-middle-MLP model, a vanilla model with the
exact same shape, and several vanilla parameter-matched baselines.

Example:
    conda run -n basics python run_architecture_comparison.py --dry-run
    conda run -n basics python run_architecture_comparison.py --max-iters 30000
    conda run -n basics python run_architecture_comparison.py --only shared_large vanilla_fewer_layers_match
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

from config import GPTConfig, TrainConfigs
from model import GPT


@dataclass(frozen=True)
class Experiment:
    key: str
    description: str
    n_layer: int
    n_embed: int
    n_head: int
    n_kv_head: int
    use_shared_middle_mlp: bool
    shared_mlp_rank: int = 16
    shared_mlp_alpha: float = 1.0
    shared_mlp_init_zero: bool = True


def count_params(cfg: GPTConfig) -> int:
    model = GPT(cfg)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_gpt_config(exp: Experiment) -> GPTConfig:
    return GPTConfig(
        block_size=512,
        vocab_size=4096,
        n_layer=exp.n_layer,
        n_head=exp.n_head,
        n_kv_head=exp.n_kv_head,
        n_embed=exp.n_embed,
        dropout=0.2,
        bias=False,
        use_rotary=True,
        use_qk_norm=True,
        use_alibi=False,
        use_exclusive_self_attention=True,
        exclusive_self_attention_eps=1e-8,
        use_gradient_checkpointing=True,
        use_shared_middle_mlp=exp.use_shared_middle_mlp,
        shared_mlp_rank=exp.shared_mlp_rank,
        shared_mlp_alpha=exp.shared_mlp_alpha,
        shared_mlp_init_zero=exp.shared_mlp_init_zero,
    )


# Target model: the new shared-middle-MLP large shape.
# Parameter target is about 49.0M with rank=16.
EXPERIMENTS = [
    Experiment(
        key="shared_large",
        description="NEW ARCH: L14 d896 shared middle MLP rank16",
        n_layer=14,
        n_embed=896,
        n_head=14,
        n_kv_head=1,
        use_shared_middle_mlp=True,
        shared_mlp_rank=16,
    ),
    Experiment(
        key="vanilla_same_config",
        description="VANILLA: exact same L14 d896 shape as new arch",
        n_layer=14,
        n_embed=896,
        n_head=14,
        n_kv_head=1,
        use_shared_middle_mlp=False,
    ),
    Experiment(
        key="vanilla_same_depth_match",
        description="VANILLA PARAM-MATCH: same depth, narrower width",
        n_layer=14,
        n_embed=570,
        n_head=10,
        n_kv_head=1,
        use_shared_middle_mlp=False,
    ),
    Experiment(
        key="vanilla_fewer_layers_match",
        description="VANILLA PARAM-MATCH: fewer layers, wider width",
        n_layer=8,
        n_embed=736,
        n_head=8,
        n_kv_head=2,
        use_shared_middle_mlp=False,
    ),
    Experiment(
        key="vanilla_balanced_match",
        description="VANILLA PARAM-MATCH: balanced 10-layer baseline",
        n_layer=10,
        n_embed=670,
        n_head=10,
        n_kv_head=2,
        use_shared_middle_mlp=False,
    ),
]


def build_child_code(gpt_cfg: GPTConfig, train_overrides: dict) -> str:
    cfg_json = json.dumps(asdict(gpt_cfg), sort_keys=True)
    train_json = json.dumps(train_overrides, sort_keys=True)
    return textwrap.dedent(
        f"""
        import json
        from config import GPTConfig, TrainConfigs

        gpt_cfg = GPTConfig(**json.loads({cfg_json!r}))
        _, train_cfg = TrainConfigs.for_model_size("large")

        for key, value in json.loads({train_json!r}).items():
            setattr(train_cfg, key, value)

        TrainConfigs.for_model_size = staticmethod(lambda _name: (gpt_cfg, train_cfg))

        import train
        """
    )


def run_experiment(exp: Experiment, args, target_params: int) -> None:
    cfg = make_gpt_config(exp)
    params = count_params(cfg)
    delta = params - target_params
    delta_pct = 100.0 * delta / target_params
    arch = "sharedmlp" if exp.use_shared_middle_mlp else "vanilla"
    run_name = (
        f"archcmp_{exp.key}_{arch}_"
        f"L{exp.n_layer}_d{exp.n_embed}_h{exp.n_head}_kv{exp.n_kv_head}_"
        f"p{params/1e6:.2f}M"
    )
    if exp.use_shared_middle_mlp:
        run_name += f"_r{exp.shared_mlp_rank}"

    out_dir = Path(args.out_root) / run_name
    print("=" * 100)
    print(f"key:         {exp.key}")
    print(f"description: {exp.description}")
    print(f"run name:    {run_name}")
    print(f"params:      {params:,} ({delta:+,}, {delta_pct:+.2f}% vs shared_large target)")
    print(f"out dir:     {out_dir}")

    if args.dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    train_overrides = {
        # Aggressive defaults for a single 48GB VRAM GPU, 10 CPU cores, 60GB RAM.
        "max_iters": args.max_iters,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_workers": args.num_workers,
        "dtype": args.dtype,
        "compile": args.compile,
        "eval_interval": args.eval_interval,
        "eval_start_iter": args.eval_start_iter,
        "eval_iters": args.eval_iters,
        "log_interval": args.log_interval,
        "learning_rate": args.learning_rate,
        "warmup_iters": args.warmup_iters,
        "lr_decay_iters": args.max_iters,
        "min_lr": args.min_lr,
        "optimizer_offload": args.optimizer_offload,
    }

    env = os.environ.copy()
    env.update(
        {
            "MODEL_SIZE": "architecture_comparison",
            "WANDB_PROJECT": args.wandb_project,
            "WANDB_RUN_NAME": run_name,
            "WANDB_GROUP": args.wandb_group,
            "OUT_DIR": str(out_dir),
            "PYTHONUNBUFFERED": "1",
        }
    )

    code = build_child_code(cfg, train_overrides)
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iters", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimizer-offload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--warmup-iters", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-start-iter", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--wandb-project", default="smolGPT")
    parser.add_argument("--wandb-group", default="architecture-comparison-shared-middle-mlp")
    parser.add_argument("--out-root", default="out/architecture_comparison")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these experiment keys")
    parser.add_argument("--skip", nargs="*", default=None, help="Skip these experiment keys")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    selected = EXPERIMENTS
    if args.only:
        wanted = set(args.only)
        selected = [exp for exp in selected if exp.key in wanted]
    if args.skip:
        skipped = set(args.skip)
        selected = [exp for exp in selected if exp.key not in skipped]

    if not selected:
        raise SystemExit("No experiments selected")

    target_params = count_params(make_gpt_config(EXPERIMENTS[0]))
    block_size = make_gpt_config(EXPERIMENTS[0]).block_size
    tokens_per_iter = args.batch_size * args.grad_accum * block_size
    print(f"Target params from {EXPERIMENTS[0].key}: {target_params:,}")
    print(f"Selected experiments: {', '.join(exp.key for exp in selected)}")
    print(
        f"Training dtype: {args.dtype}; batch_size={args.batch_size}; "
        f"grad_accum={args.grad_accum}; num_workers={args.num_workers}; "
        f"tokens_per_iter={tokens_per_iter:,}"
    )

    for exp in selected:
        run_experiment(exp, args, target_params)


if __name__ == "__main__":
    main()
