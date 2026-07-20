import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import inspect

from config import GPTConfig


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = None
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, q, k):
        seq_len = q.shape[1]
        if seq_len != self.seq_len_cached:
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, device=q.device) / self.dim)
            )
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=q.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            self.cos_cached = freqs.cos().type_as(q)
            self.sin_cached = freqs.sin().type_as(q)
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        q_ = self.apply_rotary_emb(q, cos, sin)
        k_ = self.apply_rotary_emb(k, cos, sin)
        return q_, k_

    def apply_rotary_emb(self, x, cos, sin):
        assert x.ndim == 4  # multihead attention
        d = x.shape[3] // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat([y1, y2], 3).type_as(x)


class ALiBi(torch.nn.Module):
    def __init__(self, n_head):
        super().__init__()
        self.n_head = n_head
        self.alibi_bias = None
        self.seq_len_cached = None

    def forward(self, attn_weights, seq_len):
        if self.seq_len_cached != seq_len:
            self._create_alibi_bias(seq_len)
            self.seq_len_cached = seq_len

        return attn_weights + self.alibi_bias[:, :, :seq_len, :seq_len]

    def _create_alibi_bias(self, seq_len):
        start = 2 ** (-8.0 / self.n_head)
        slopes = torch.tensor(
            [start**i for i in range(self.n_head)], dtype=torch.float32
        )

        positions = torch.arange(seq_len)
        distance_matrix = (
            (positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
        )

        alibi = -slopes.view(-1, 1, 1) * distance_matrix.unsqueeze(0).unsqueeze(0)

        self.register_buffer("alibi_bias", alibi.unsqueeze(0), persistent=False)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        assert config.n_embed % config.n_head == 0
        assert (
            config.n_head % config.n_kv_head == 0
        )  # Q heads must be divisible by KV heads
        self.head_dim = config.n_embed // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep_kv = (
            self.n_head // self.n_kv_head
        )  # How many Q groups share each KV head

        # GQA: Separate projections for Q, K, V
        # Q: n_head * head_dim, K/V: n_kv_head * head_dim
        self.c_q = nn.Linear(
            config.n_embed, config.n_head * self.head_dim, bias=config.bias
        )
        self.c_k = nn.Linear(
            config.n_embed, config.n_kv_head * self.head_dim, bias=config.bias
        )
        self.c_v = nn.Linear(
            config.n_embed, config.n_kv_head * self.head_dim, bias=config.bias
        )
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.use_exclusive_self_attention = config.use_exclusive_self_attention

        # QK-Norm: Learnable normalization for query and key
        if config.use_qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if not self.flash:
            print("Not using flash attention")
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

        if config.use_rotary:
            self.rotary = Rotary(self.head_dim)

        if config.use_alibi:
            self.alibi = ALiBi(self.n_head)

        if self.use_exclusive_self_attention:
            gate_hidden_dim = max(1, self.head_dim // 2)
            self.exclusive_gate = nn.Sequential(
                nn.Linear(self.head_dim * 2, gate_hidden_dim, bias=True),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, 1, bias=True),
                nn.Sigmoid(),
            )

    def forward(self, x):
        B, T, C = x.shape

        # Generate Q, K, V with separate linear layers (GQA)
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        # Reshape: Q -> (B, T, n_head, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Reshape: K, V -> (B, T, n_kv_head, head_dim)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # Apply QK-Norm if enabled
        if self.config.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # GQA: Expand KV to match number of Q heads
        # k, v: (B, n_kv_head, T, head_dim) -> (B, n_head, T, head_dim)
        if self.n_rep_kv > 1:
            k = k.repeat_interleave(self.n_rep_kv, dim=1)
            v = v.repeat_interleave(self.n_rep_kv, dim=1)

        # Apply rotary embeddings if enabled
        if self.config.use_rotary:
            q, k = self.rotary(q, k)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            attn_pattern = (q @ k.transpose(-2, -1)) * (
                1.0 / math.sqrt(k.shape[-1])
            )  # B, nh, T, T

            # Apply causal mask
            attn_pattern = attn_pattern.masked_fill(
                self.bias[:, :, :T, :T] == 0, float("-inf")
            )

            # Apply ALiBi if enabled (not compatible with Flash)
            if self.config.use_alibi:
                attn_pattern = self.alibi(attn_pattern, T)

            attn = F.softmax(attn_pattern, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v  # B, nh, T, T @ B, nh, T, hs -> B, nh, T, hs

        if self.use_exclusive_self_attention:
            dot_product = torch.sum(y * v, dim=-1, keepdim=True)
            v_norm_sq = torch.sum(v * v, dim=-1, keepdim=True)
            component = (
                dot_product / (v_norm_sq + self.config.exclusive_self_attention_eps)
            ) * v
            alpha = self.exclusive_gate(torch.cat([q, y], dim=-1))
            y = y - alpha * component

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 4 * config.n_embed
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(config.n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.n_embed, bias=False)
        self.w3 = nn.Linear(config.n_embed, hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MLPRouter(nn.Module):
    """Choose once per sequence whether an MLP residual should run."""

    def __init__(self, width, hidden_width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, hidden_width), nn.Tanh(), nn.Linear(hidden_width, 2)
        )

    def forward(self, x):
        logits = self.net(x.mean(dim=1))
        mlp_probability = F.softmax(logits, dim=-1)[:, 1]
        if self.training:
            # A hard choice in the forward pass with soft straight-through
            # gradients lets the router learn from the language-model loss.
            apply_mlp = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)[:, 1]
        else:
            apply_mlp = (mlp_probability >= 0.5).to(x.dtype)
        return apply_mlp, mlp_probability


class Block(nn.Module):
    def __init__(self, config, route_mlp=False):
        super().__init__()
        self.ln_1 = nn.RMSNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.RMSNorm(config.n_embed)
        self.ffd = FeedForward(config)
        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        self.router = (
            MLPRouter(config.n_embed, config.router_hidden) if route_mlp else None
        )

    def forward(self, x):
        if self.use_gradient_checkpointing and self.training:
            # Memory-efficient: recompute forward during backward
            x = x + torch.utils.checkpoint.checkpoint(
                self._attn_wrapper, self.ln_1(x), use_reentrant=False
            )
        else:
            x = x + self.attn(self.ln_1(x))

        normalized = self.ln_2(x)
        if self.router is None:
            if self.use_gradient_checkpointing and self.training:
                update = torch.utils.checkpoint.checkpoint(
                    self._ffn_wrapper, normalized, use_reentrant=False
                )
            else:
                update = self.ffd(normalized)
            return x + update, None, None

        if self.use_gradient_checkpointing and self.training:
            update, probability, choice = torch.utils.checkpoint.checkpoint(
                self._routed_ffn_wrapper, normalized, use_reentrant=False
            )
        else:
            update, probability, choice = self._routed_ffn_wrapper(normalized)
        return x + update, probability, choice

    def _attn_wrapper(self, x):
        return self.attn(x)

    def _ffn_wrapper(self, x):
        return self.ffd(x)

    def _routed_ffn_wrapper(self, x):
        apply_mlp, mlp_probability = self.router(x)
        if self.training:
            # Training computes every MLP so straight-through routing receives a
            # useful gradient. In evaluation/generation, skipped sequences avoid it.
            update = self.ffd(x)
        else:
            update = torch.zeros_like(x)
            selected = apply_mlp.bool()
            if selected.any():
                # Under mixed precision, the residual stream can be float32
                # while the MLP output is BF16. Cast before indexed assignment.
                update[selected] = self.ffd(x[selected]).to(update.dtype)
        return (
            apply_mlp[:, None, None] * update,
            mlp_probability,
            apply_mlp.detach().mean(),
        )


def always_on_mlp_layers(layer_count, middle_fraction):
    """Return the protected central span and final layer for routed MLPs."""
    if not 0.0 <= middle_fraction <= 1.0:
        raise ValueError("middle_mlp_fraction must be in [0, 1]")
    middle_count = max(1, round(layer_count * middle_fraction))
    middle_count = min(layer_count, middle_count)
    middle_start = (layer_count - middle_count) // 2
    protected = set(range(middle_start, middle_start + middle_count))
    protected.add(layer_count - 1)
    return protected


class GPT(nn.Module):
    def __init__(self, config, use_gradient_checkpointing=False):
        super().__init__()
        self.config = config

        if config.router_hidden < 1:
            raise ValueError("router_hidden must be at least 1")
        if not 0.0 <= config.target_mlp_rate <= 1.0:
            raise ValueError("target_mlp_rate must be in [0, 1]")

        protected_layers = (
            always_on_mlp_layers(config.n_layer, config.middle_mlp_fraction)
            if config.use_routed_mlp
            else set()
        )
        self.always_on_mlp_layers = tuple(sorted(protected_layers))

        # Create base transformer components
        transformer_dict = {
            "wte": nn.Embedding(config.vocab_size, config.n_embed),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList(
                [
                    Block(
                        config,
                        route_mlp=config.use_routed_mlp
                        and layer_index not in protected_layers,
                    )
                    for layer_index in range(config.n_layer)
                ]
            ),
            "ln_f": nn.RMSNorm(config.n_embed),
        }

        # Only add positional embeddings if not using rotary or ALiBi
        if not config.use_rotary and not config.use_alibi:
            transformer_dict["wpe"] = nn.Embedding(config.block_size, config.n_embed)

        self.transformer = nn.ModuleDict(transformer_dict)

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight
        self.last_route_stats = {}

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        x = self.transformer.wte(idx)

        # Add learnable positional embeddings (only when NOT using RoPE or ALiBi)
        if not self.config.use_rotary and not self.config.use_alibi:
            device = idx.device
            b, t = idx.shape
            pos_emb = self.transformer.wpe(
                torch.arange(0, t, dtype=torch.long, device=device)
            )
            x = x + pos_emb

        route_probabilities = []
        route_choices = []
        routed_layer_indices = []
        for layer_index, block in enumerate(self.transformer.h):
            x, probability, choice = block(x)
            if probability is not None:
                route_probabilities.append(probability)
                route_choices.append(choice)
                routed_layer_indices.append(layer_index)
        x = self.transformer.ln_f(x)

        if route_probabilities:
            route_probability = torch.stack(route_probabilities).mean()
            route_choice = torch.stack(route_choices).mean()
            router_aux_loss = (route_probability - self.config.target_mlp_rate).square()
        else:
            route_probability = x.new_tensor(1.0)
            route_choice = x.new_tensor(1.0)
            router_aux_loss = x.new_zeros(())
        self.last_route_stats = {
            "mlp_probability": route_probability.detach(),
            "mlp_selected": route_choice.detach(),
            "router_aux_loss": router_aux_loss.detach(),
            "per_layer_selected": {
                layer_index: choice.detach()
                for layer_index, choice in zip(routed_layer_indices, route_choices)
            },
        }

        if targets is not None:
            logits = self.lm_head(x)
            token_loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=-1
            )
            loss = token_loss + self.config.router_aux_weight * router_aux_loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    def configure_optimizers(
        self, weight_decay, learning_rate, betas, device_type, optimizer_offload=False
    ):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        if optimizer_offload:
            # Move optimizer states to CPU to save VRAM
            # This adds slight overhead but can save ~30% VRAM
            print("Enabling optimizer CPU offload")
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to("cpu")

        return optimizer

    @torch.no_grad()
    def generate(
        self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, min_p=None
    ):
        for _ in range(max_new_tokens):
            context = (
                idx
                if idx.size(1) < self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(context)

            logits = logits[:, -1, :] / temperature

            if top_p is not None and top_p > 0.0:
                probs = torch.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(
                    probs, descending=True, dim=-1
                )
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                mask = cumulative_probs >= top_p
                mask[..., 0] = True

                cutoff_indices = mask.int().argmax(dim=-1, keepdim=True)

                top_p_mask = torch.zeros_like(logits, dtype=torch.bool)
                for b in range(logits.size(0)):
                    cut = cutoff_indices[b].item()
                    kept_indices = sorted_indices[b, : cut + 1]
                    top_p_mask[b, kept_indices] = True
                logits[~top_p_mask] = float("-inf")

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if min_p is not None and min_p > 0.0:
                logit_max = logits.max(dim=-1, keepdim=True).values
                threshold = logit_max + torch.log(
                    torch.tensor(min_p, device=logits.device, dtype=logits.dtype)
                )
                logits[logits < threshold] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == 2:
                break
            idx = torch.cat([idx, idx_next], dim=-1)

        return idx
