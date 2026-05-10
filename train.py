from model import GPT
from config import GPTConfig, TrainingConfig, TrainConfigs
from functools import partial
import sys
import time
import math
import os
import torch
import wandb
from tqdm import tqdm
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from dataset import Task
from tokenizer import Tokenizer

model_size = os.getenv("MODEL_SIZE")
if model_size:
    gpt_config, train_config = TrainConfigs.for_model_size(model_size)
else:
    gpt_config, train_config = GPTConfig(), TrainingConfig()

max_iters_override = os.getenv("MAX_ITERS")
if max_iters_override:
    train_config.max_iters = int(max_iters_override)

out_dir = os.getenv("OUT_DIR", "out/")
resume = False
ddp = int(os.environ.get("RANK", -1)) != -1
tokenizer = Tokenizer(f"data/tok{gpt_config.vocab_size}.model")

if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    train_config.gradient_accumulation_steps //= ddp_world_size

else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = (
    train_config.gradient_accumulation_steps
    * train_config.batch_size
    * gpt_config.block_size
    * ddp_world_size
)
if master_process:
    print("Tokens per iteration: ", tokens_per_iter)
    if model_size:
        print(f"Using model preset: {model_size.lower()}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(42 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Select dtype for mixed precision
if train_config.dtype == "float16":
    dtype = torch.float16
    scaler_enabled = True
elif train_config.dtype == "bfloat16":
    dtype = torch.bfloat16
    scaler_enabled = False  # BF16 doesn't need GradScaler (has its own loss scaling)
else:
    dtype = torch.float32
    scaler_enabled = False

ctx = torch.autocast(train_config.device, dtype=dtype)
if master_process:
    print(f"Using dtype: {train_config.dtype}, GradScaler: {scaler_enabled}")

model_args = dict(
    n_layer=gpt_config.n_layer,
    n_head=gpt_config.n_head,
    n_kv_head=gpt_config.n_kv_head,
    n_embed=gpt_config.n_embed,
    block_size=gpt_config.block_size,
    bias=gpt_config.bias,
    vocab_size=gpt_config.vocab_size,
    dropout=gpt_config.dropout,
    use_rotary=gpt_config.use_rotary,
    use_qk_norm=gpt_config.use_qk_norm,
    use_gradient_checkpointing=gpt_config.use_gradient_checkpointing,
    use_shared_middle_mlp=gpt_config.use_shared_middle_mlp,
    shared_mlp_rank=gpt_config.shared_mlp_rank,
    shared_mlp_alpha=gpt_config.shared_mlp_alpha,
    shared_mlp_init_zero=gpt_config.shared_mlp_init_zero,
)

if master_process:
    wandb.init(
        project=os.getenv("WANDB_PROJECT", "smolGPT"),
        name=os.getenv("WANDB_RUN_NAME"),
        config={
            "model": model_args,
            "training": vars(train_config),
            "tokens_per_iter": tokens_per_iter,
        },
        dir=out_dir,
    )

iter_batches = partial(
    Task.iter_batches,
    batch_size=train_config.batch_size,
    max_seq_len=gpt_config.block_size,
    device=train_config.device,
    num_workers=0,
)

best_val_loss = 1e9
if resume:
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=train_config.device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf).to(train_config.device)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    best_val_loss = checkpoint["best_val_loss"]

    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    if master_process:
        print("Loaded checkpoint")
else:
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(train_config.device)

scaler = torch.GradScaler(enabled=scaler_enabled)
optimizer = model.configure_optimizers(
    train_config.weight_decay,
    train_config.learning_rate,
    (train_config.beta1, train_config.beta2),
    train_config.device,
    optimizer_offload=train_config.optimizer_offload,
)

if sys.version_info >= (3, 13):
    train_config.compile = False

if train_config.compile:
    model = torch.compile(model)


if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(train_config.eval_iters)
        batch_iter = iter_batches(split=split)
        for k in range(train_config.eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                _, loss = raw_model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < train_config.warmup_iters:
        return train_config.learning_rate * (it + 1) / (train_config.warmup_iters + 1)
    if it > train_config.lr_decay_iters:
        return train_config.min_lr
    decay_ratio = (it - train_config.warmup_iters) / (
        train_config.lr_decay_iters - train_config.warmup_iters
    )
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return train_config.min_lr + coeff * (
        train_config.learning_rate - train_config.min_lr
    )


params = sum([p.numel() for p in model.parameters() if p.requires_grad])
if master_process:
    print("Number of params:", params)

train_batch_iter = iter_batches(split="train")
X, Y = next(train_batch_iter)

iter_num = 0
raw_model = model.module if ddp else model
t0 = time.time()
total_steps = train_config.max_iters + 1
pbar = (
    tqdm(total=total_steps, desc="Training", dynamic_ncols=True)
    if master_process
    else None
)


def log_message(msg):
    if not master_process:
        return
    if pbar is not None:
        pbar.write(msg)
    else:
        print(msg)


while True:
    lr = get_lr(iter_num) if train_config.decay_lr else train_config.learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if (
        iter_num >= train_config.eval_start_iter
        and iter_num % train_config.eval_interval == 0
        and master_process
    ):
        losses = estimate_loss()
        log_message(
            f"step {iter_num}: train_loss {losses['train']:.4f}, val_loss {losses['val']:.4f}"
        )
        wandb.log(
            {
                "train_loss": float(losses["train"]),
                "val_loss": float(losses["val"]),
                "lr": lr,
            },
            step=iter_num,
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            checkpoint = {
                "model": raw_model.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
            }
            log_message(f"Saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
        model.eval()
        prompt = "Once upon a time"
        idxs = tokenizer.encode(prompt, bos=True, eos=False)
        x = torch.tensor(idxs, dtype=torch.long, device=train_config.device).unsqueeze(
            0
        )
        y = model.generate(x, max_new_tokens=256, temperature=1.0, top_p=0.90)
        log_message(
            "===\n\nGenerated Story:\n\n" + tokenizer.decode(y[0].tolist()) + "\n\n==="
        )
        model.train()

    for micro_step in range(train_config.gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (
                micro_step == train_config.gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / train_config.gradient_accumulation_steps
        X, Y = next(train_batch_iter)
        scaler.scale(loss).backward()

    if train_config.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    lossf = loss.item() * train_config.gradient_accumulation_steps

    if pbar is not None:
        pbar.update(1)
        pbar.set_postfix(
            {
                "step": iter_num,
                "loss": f"{lossf:.4f}",
                "lr": f"{lr:.2e}",
                "ms/it": f"{dt * 1000:.1f}",
            }
        )

    if iter_num % train_config.log_interval == 0 and master_process:
        wandb.log(
            {
                "iter_loss": lossf,
                "iter_time_ms": dt * 1000.0,
                "lr": lr,
            },
            step=iter_num,
        )
        log_message(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms")

    iter_num += 1

    if iter_num > train_config.max_iters:
        break


if ddp:
    destroy_process_group()
if pbar is not None:
    pbar.close()
if master_process:
    wandb.finish()
