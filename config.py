from dataclasses import dataclass
import torch


@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 4096
    n_layer: int = 14  # Default: LARGE model (118M params)
    n_head: int = 14
    n_kv_head: int = 1  # Default: MQA for memory efficiency
    n_embed: int = 896
    dropout: float = 0.2
    bias: bool = False
    use_rotary: bool = True
    use_qk_norm: bool = True  # QK normalization for training stability
    use_alibi: bool = False  # ALiBi positional bias (no learnable pos embeddings)
    use_exclusive_self_attention: bool = True  # Remove self-reinforcing value component via learnable gate
    exclusive_self_attention_eps: float = 1e-8
    use_gradient_checkpointing: bool = True  # Save memory during training
    # Mixture-of-depth routing: attention always runs, while eligible MLP
    # residuals are conditionally applied once per sequence.
    use_routed_mlp: bool = False
    middle_mlp_fraction: float = 0.20
    # Target fraction of sequences that run each routed MLP (with headroom under 50%).
    target_mlp_rate: float = 0.40
    router_aux_weight: float = 0.1
    router_hidden: int = 16
    # Hard per-layer top-k so selection cannot exceed target_mlp_rate when batch > 1.
    enforce_router_capacity: bool = True


# Pre-configured model sizes optimized for different VRAM budgets
class ModelSizes:
    # These configs assume batch_size manageable with gradient accumulation

    # ~0.3GB VRAM - for integrated GPUs or very limited VRAM
    MICRO = dict(n_layer=2, n_head=2, n_embed=128, n_kv_head=1, block_size=256)

    # ~0.5GB VRAM - small but functional
    TINY = dict(n_layer=3, n_head=3, n_embed=192, n_kv_head=1, block_size=384)

    # ~0.7GB VRAM - balanced for 4GB cards
    SMALL = dict(n_layer=4, n_head=4, n_embed=256, n_kv_head=2, block_size=512)

    # ~1GB VRAM - for 6GB cards
    MEDIUM = dict(n_layer=6, n_head=6, n_embed=384, n_kv_head=2, block_size=512)

    # ~1.5GB VRAM - full model for decent GPUs
    FULL = dict(n_layer=8, n_head=8, n_embed=512, n_kv_head=2, block_size=512)

    # ~2GB VRAM - 128M parameter model (14L/896d)
    LARGE = dict(n_layer=14, n_head=14, n_embed=896, n_kv_head=1, block_size=512)


@dataclass
class TrainingConfig:
    learning_rate: float = 6e-4
    max_iters: int = 30000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    decay_lr: bool = True
    warmup_iters: int = 1000
    lr_decay_iters: int = 30000
    min_lr: float = 6e-5

    eval_interval: int = 100
    eval_start_iter: int = 100
    log_interval: int = 10
    eval_iters: int = 200
    gradient_accumulation_steps: int = 8
    batch_size: int = 8  # Smaller batch for larger models

    device: str = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    dtype: str = "bfloat16"  # Options: "bfloat16", "float16", "float32"
    compile: bool = True
    optimizer_offload: bool = False  # Move optimizer states to CPU (saves ~30% VRAM)


# Pre-configured training settings for different VRAM budgets
class TrainConfigs:
    @staticmethod
    def for_model_size(model_size_name):
        """Returns (model_config, train_config) tuples"""
        configs = {
            "micro": (
                GPTConfig(
                    n_layer=2, n_head=2, n_embed=128, n_kv_head=1, block_size=256
                ),
                TrainingConfig(
                    batch_size=32, gradient_accumulation_steps=2, max_iters=50000
                ),
            ),
            "tiny": (
                GPTConfig(
                    n_layer=3, n_head=3, n_embed=192, n_kv_head=1, block_size=384
                ),
                TrainingConfig(
                    batch_size=24, gradient_accumulation_steps=2, max_iters=40000
                ),
            ),
            "small": (
                GPTConfig(
                    n_layer=4,
                    n_head=4,
                    n_embed=256,
                    n_kv_head=2,
                    block_size=512,
                    use_gradient_checkpointing=True,
                ),
                TrainingConfig(
                    batch_size=16, gradient_accumulation_steps=4, max_iters=30000
                ),
            ),
            "medium": (
                GPTConfig(
                    n_layer=6,
                    n_head=6,
                    n_embed=384,
                    n_kv_head=2,
                    block_size=512,
                    use_gradient_checkpointing=True,
                ),
                TrainingConfig(
                    batch_size=16, gradient_accumulation_steps=4, max_iters=25000
                ),
            ),
            "full": (
                GPTConfig(
                    n_layer=8,
                    n_head=8,
                    n_embed=512,
                    n_kv_head=2,
                    block_size=512,
                    use_gradient_checkpointing=True,
                ),
                TrainingConfig(
                    batch_size=12, gradient_accumulation_steps=4, max_iters=20000
                ),
            ),
            "large": (  # ~118M params, 14L/896d, ~2GB VRAM
                GPTConfig(
                    n_layer=14,
                    n_head=14,
                    n_embed=896,
                    n_kv_head=1,
                    block_size=512,
                    use_gradient_checkpointing=True,
                    use_rotary=True,
                    use_qk_norm=True,
                ),
                TrainingConfig(
                    batch_size=4, gradient_accumulation_steps=8, max_iters=15000
                ),
            ),
        }
        return configs.get(model_size_name.lower(), configs["small"])
