# SMOL-GPT 🦾

A minimal PyTorch implementation for training your own small LLM from scratch. Designed for educational purposes and simplicity, featuring efficient training, flash attention, and modern sampling techniques.

**Now with SOTA architectures - comparable to LLaMA, Qwen, and Gemma!**

## Features ✨

### Modern Architecture (SOTA)
- **GQA (Grouped Query Attention)** - Memory-efficient attention (saves 11-13% params)
- **SwiGLU** - Gated activation used by LLaMA, Mistral, Qwen
- **RMSNorm** - Faster normalization (used by all modern LLMs)
- **RoPE** - Rotary Position Embeddings (optional)
- **ALiBi** - Attention with Linear Biases for better length extrapolation (optional)
- **QK-Norm** - Query/Key normalization for training stability (optional)
- **Exclusive Self-Attention** - Gated removal of self-reinforcing value component (optional)
- **Flash Attention** - Hardware-optimized attention when available
- **Pre-norm** - Stable training with normalization before layers

### Training Features
- **Mixed Precision** - FP16/BF16 support (saves ~25% VRAM)
- **Gradient Checkpointing** - ~50% VRAM reduction (saves up to 55% total)
- **Gradient Accumulation** - Effective larger batch sizes
- **Fused AdamW** - ~30% faster optimizer
- **TF32/FP32 MatMul** - Hardware-accelerated matrix ops
- **torch.compile** - JIT compilation for 1.5-3x speedup

### Memory Optimized
| Model Size | Params | VRAM (BF16 + Checkpoint) | Batch |
|------------|--------|--------------------------|-------|
| Micro | 0.9M | 0.3 GB | 32 |
| Tiny | 2.0M | 0.5 GB | 24 |
| Small | 4.0M | 0.7 GB | 16 |
| Medium | 11M | 1.0 GB | 12 |
| Full | 24M | 1.5 GB | 8 |

**Runs on 6GB VRAM GPUs!** ✅

## Installation 🛠️

```bash
pip install -r requirements.txt
```

**Requirements**:
- Python 3.8+
- PyTorch 2.0+ with CUDA
- 6GB+ VRAM GPU recommended (works on less with smaller models)

## Quick Start 🚀

### Option 1: Full Training Cycle

1. **Prepare Dataset**
```bash
python preprocess.py prepare-dataset --vocab-size 4096
```

2. **Start Training** (uses optimized settings for 6GB VRAM)
```bash
python train.py
```

To train the conditional-depth variant, set `MODEL_VARIANT=routed_mlp`.
It runs attention in every layer, but routes each sequence through or around
eligible MLP residuals; the central 20% of layers and final layer always keep
their MLPs.

```bash
MODEL_VARIANT=routed_mlp python train.py
```

Routing defaults are in `GPTConfig`: `target_mlp_rate=0.50`,
`router_aux_weight=0.01`, and `router_hidden=16`.

### Kaggle 2x T4 comparison run

Attach a Kaggle dataset containing the prepared `tok4096.model` and
`TinyStories_all_data/*.bin` files, then run both DDP experiments sequentially
with both T4 GPUs:

```bash
python kaggle_train.py --data-dir /kaggle/input/smolgpt-data
```

The default settings use FP16, a per-GPU batch size of 64, and retain the
original large-model token budget while writing outputs to
`/kaggle/working/smolgpt-out`.

*Training and validation metrics are logged to Weights & Biases (W&B). To run online:*
```bash
wandb login
python train.py
```

*To run without internet/sync, use offline mode:*
```bash
WANDB_MODE=offline python train.py
```

3. **Generate Text**
```bash
python sample.py \
    --prompt "Once upon a time" \
    --num_samples 3 \
    --temperature 0.7 \
    --max_new_tokens 500
```

### Option 2: Use Pre-trained Model

1. **Download Assets**
```bash
# Download tokenizer
wget https://huggingface.co/OmAlve/TinyStories-SmolGPT/resolve/main/tok4096.model -P data/

# Download pre-trained checkpoint
wget https://huggingface.co/OmAlve/TinyStories-SmolGPT/resolve/main/ckpt.pt -P out/
```

2. **Run Inference**
```bash
python sample.py \
    --prompt "Once upon a time" \
    --tokenizer_path data/tok4096.model \
    --ckpt_path out/ckpt.pt \
    --num_samples 3 \
    --max_new_tokens 200 \
    --temperature 0.7
```

## Configuration ⚙️

### Model Sizes (Pre-configured)

```python
from config import GPTConfig, TrainConfigs

# Micro (~0.3GB VRAM) - for limited GPUs
config, train_cfg = TrainConfigs.for_model_size('micro')

# Tiny (~0.5GB VRAM)
config, train_cfg = TrainConfigs.for_model_size('tiny')

# Small (~0.7GB VRAM) - recommended for 4GB cards
config, train_cfg = TrainConfigs.for_model_size('small')

# Medium (~1GB VRAM) - recommended for 6GB cards
config, train_cfg = TrainConfigs.for_model_size('medium')

# Full (~1.5GB VRAM) - for higher-end GPUs
config, train_cfg = TrainConfigs.for_model_size('full')
```

### Full Custom Configuration

```python
from config import GPTConfig, TrainingConfig
from model import GPT

# GQA + QK-Norm + RoPE (SOTA configuration)
config = GPTConfig(
    n_layer=6,           # Number of transformer layers
    n_head=6,            # Number of attention heads
    n_kv_head=2,         # KV heads for GQA (2 = 3x reduction)
    n_embed=384,         # Embedding dimension
    block_size=512,      # Context length
    vocab_size=4096,     # Vocabulary size
    dropout=0.1,         # Dropout rate
    use_rotary=True,     # Enable RoPE
    use_qk_norm=True,    # Enable QK normalization
    use_alibi=False,     # Enable ALiBi (alternative to RoPE)
    use_exclusive_self_attention=False,  # Enable ESA
    use_gradient_checkpointing=True,  # Save VRAM
)

# Training config with mixed precision
train_cfg = TrainingConfig(
    batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=6e-4,
    weight_decay=0.1,
    dtype="bfloat16",    # BF16 mixed precision
    compile=True,        # torch.compile for speed
)

model = GPT(config)
```

### Architecture Comparison with SOTA Models

| Feature | smolGPT | Qwen3 | Gemma 3 | LLaMA 3 |
|---------|---------|-------|---------|---------|
| GQA | ✅ | ✅ | ✅ | ✅ |
| SwiGLU | ✅ | ✅ | ✅ | ✅ |
| RMSNorm | ✅ | ✅ | ✅ | ✅ |
| RoPE | ✅ | ✅ | ✅ | ✅ |
| ALiBi | ✅ | - | ✅ | - |
| QK-Norm | ✅ | ✅ | ✅ | - |
| Pre-norm | ✅ | ✅ | ✅ | ✅ |

## File Structure 📁

```
smolGPT/
├── config.py           - Model & training configuration (with ModelSizes, TrainConfigs)
├── dataset.py          - Data loading & preprocessing
├── model.py            - GPT model (GQA, SwiGLU, RMSNorm, RoPE, ALiBi, QK-Norm)
├── preprocess.py       - Dataset preparation scripts
├── sample.py           - Text generation script
├── tokenizer.py        - Tokenizer wrapper
├── train.py            - Main training loop (mixed precision, gradient checkpointing)
├── assets/             - Assets (loss curves, etc.)
└── out/                - Checkpoints and logs
```

## VRAM Optimization Tips

If you encounter OOM (Out of Memory) errors:

1. **Reduce batch size**:
```python
batch_size=8  # Start here if OOM
```

2. **Enable gradient checkpointing**:
```python
use_gradient_checkpointing=True
```

3. **Reduce block size**:
```python
block_size=256  # Smaller context = less VRAM
```

4. **Use smaller model**:
```python
# Instead of 8L/512d, try 4L/256d
config = GPTConfig(n_layer=4, n_head=4, n_embed=256)
```

5. **Try FP32 if BF16 issues**:
```python
dtype="float32"  # More stable, uses more VRAM
```

## Architecture Highlights

### Grouped Query Attention (GQA)
- Multiple query heads share KV heads
- Reduces KV cache by ~75% (for n_kv_head=2)
- Maintains quality while reducing memory

### ALiBi (Attention with Linear Biases)
- No learnable positional embeddings
- Better length extrapolation than learned positions
- Especially useful for longer context than training

### QK-Normalization
- Normalizes query and key before attention
- Prevents training instability in deep models
- Used by Qwen, Gemma, and Mistral

---

**Note**: This implementation is inspired by modern LLM training practices (Qwen3, Gemma 3, LLaMA 3) and adapted for educational purposes. Run locally on a single consumer GPU!

## Contributing 🤝

Contributions welcome! Please open an issue or PR for:
- Bug fixes
- Performance improvements
- New features
