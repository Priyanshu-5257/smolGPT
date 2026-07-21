"""Run vanilla and routed-MLP experiments on a Kaggle 2x T4 notebook.

Attach a Kaggle dataset containing ``tok4096.model`` and the
``TinyStories_all_data/*.bin`` shards, then run:

    python kaggle_train.py --data-dir /kaggle/input/smolgpt-data

The script launches each experiment sequentially with both T4 GPUs through
PyTorch DDP. Outputs remain in ``/kaggle/working/smolgpt-out`` for download as
a Kaggle notebook artifact.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/kaggle/input/smolgpt-data"),
        help="Directory containing tok4096.model and TinyStories_all_data/*.bin",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/smolgpt-out"),
    )
    parser.add_argument("--model-size", default="large")
    parser.add_argument("--max-iters", type=int, default=3750)
    parser.add_argument("--warmup-iters", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64, help="Per GPU")
    # Must be divisible by nproc_per_node (2 T4s). train.py splits this across ranks.
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-iters", type=int, default=13)
    parser.add_argument(
        "--target-mlp-rate",
        type=float,
        default=0.40,
        help="Max fraction of sequences that run each routed MLP",
    )
    parser.add_argument(
        "--router-aux-weight",
        type=float,
        default=0.1,
        help="Weight on one-sided MLP-rate overshoot aux loss",
    )
    parser.add_argument(
        "--enforce-router-capacity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hard top-k capacity so selection cannot exceed --target-mlp-rate",
    )
    parser.add_argument("--prepare-data", action="store_true")
    return parser.parse_args()


def validate_or_prepare_data(data_dir: Path, prepare_data: bool) -> Path:
    tokenizer = data_dir / "tok4096.model"
    shards = data_dir / "TinyStories_all_data"
    if tokenizer.exists() and any(shards.glob("*.bin")):
        return data_dir
    if not prepare_data:
        raise FileNotFoundError(
            f"Expected {tokenizer} and {shards}/*.bin. Attach the prepared "
            "dataset in Kaggle or retry with --prepare-data and Internet enabled."
        )

    local_data = PROJECT_ROOT / "data"
    subprocess.run(
        [sys.executable, "preprocess.py", "prepare-dataset", "--vocab-size", "4096"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    return local_data


def expose_data_in_project(data_dir: Path) -> None:
    """Make Kaggle input data available at the path expected by train.py."""
    target = PROJECT_ROOT / "data"
    if data_dir.resolve() == target.resolve():
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            f"{target} already exists. Run from a clean Kaggle working copy or "
            "pass --data-dir data."
        )
    target.symlink_to(data_dir.resolve(), target_is_directory=True)


def run_variant(variant: str, args) -> None:
    output_dir = args.output_dir / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "MODEL_SIZE": args.model_size,
            "MODEL_VARIANT": variant,
            "OUT_DIR": str(output_dir),
            # T4 supports FP16 Tensor Cores but not BF16 Tensor Cores.
            "TRAIN_DTYPE": "float16",
            "BATCH_SIZE": str(args.batch_size),
            "TRAIN_GRADIENT_ACCUMULATION_STEPS": str(
                args.gradient_accumulation_steps
            ),
            "MAX_ITERS": str(args.max_iters),
            "TRAIN_WARMUP_ITERS": str(args.warmup_iters),
            "TRAIN_LR_DECAY_ITERS": str(args.max_iters),
            "TRAIN_EVAL_INTERVAL": str(args.eval_interval),
            "TRAIN_EVAL_START_ITER": str(args.eval_interval),
            "TRAIN_EVAL_ITERS": str(args.eval_iters),
            "TARGET_MLP_RATE": str(args.target_mlp_rate),
            "ROUTER_AUX_WEIGHT": str(args.router_aux_weight),
            "ENFORCE_ROUTER_CAPACITY": (
                "1" if args.enforce_router_capacity else "0"
            ),
            "WANDB_MODE": "offline",
            "WANDB_DIR": str(output_dir),
        }
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "train.py",
    ]
    print(f"\n=== Training {variant} with both T4 GPUs ===", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def main():
    args = parse_args()
    nproc_per_node = 2
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("batch size and gradient accumulation steps must be positive")
    if not 0.0 <= args.target_mlp_rate <= 1.0:
        raise ValueError("target MLP rate must be in [0, 1]")
    if args.router_aux_weight < 0.0:
        raise ValueError("router aux weight must be non-negative")
    if args.gradient_accumulation_steps % nproc_per_node != 0:
        raise ValueError(
            f"gradient accumulation steps ({args.gradient_accumulation_steps}) "
            f"must be divisible by nproc_per_node ({nproc_per_node})"
        )

    data_dir = validate_or_prepare_data(args.data_dir, args.prepare_data)
    expose_data_in_project(data_dir)
    # Vanilla already trained; only retrain the capacity-capped routed variant.
    # run_variant("vanilla", args)
    run_variant("routed_mlp", args)


if __name__ == "__main__":
    main()
