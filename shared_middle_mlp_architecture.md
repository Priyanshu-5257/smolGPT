# Shared-Middle-MLP Transformer Architecture

## Block Layout by Depth

For transformer layers `l = 0 ... N-1`:

- `l = 0` (first layer):
  - Attention: layer-specific
  - MLP: full (independent)

- `0 < l < N-1` (middle layers):
  - Attention: layer-specific
  - MLP: shared base + per-layer low-rank delta

- `l = N-1` (last layer):
  - Attention: layer-specific
  - MLP: full (independent)

## MLP Parameterization

Middle-layer MLP weights are defined as:

`W_l = W_base + ΔW_l`

with low-rank factorization:

`ΔW_l = B_l A_l`

and scaling:

`W_l = W_base + (α / r) * (B_l A_l)`

where:
- `r` = low-rank dimension
- `α` = low-rank scaling factor

## SwiGLU MLP Form

For each MLP projection in middle layers:

- `w1_l = w1_base + (α/r) * (B1_l A1_l)`
- `w2_l = w2_base + (α/r) * (B2_l A2_l)`
- `w3_l = w3_base + (α/r) * (B3_l A3_l)`

Forward:

`h1 = x @ w1_l^T`

`h3 = x @ w3_l^T`

`g = silu(h1) ⊙ h3`

`y = g @ w2_l^T`

## Execution Form (without materializing full deltas)

`h1 = x @ w1_base^T + (α/r) * ((x @ A1_l^T) @ B1_l^T)`

`h3 = x @ w3_base^T + (α/r) * ((x @ A3_l^T) @ B3_l^T)`

`g = silu(h1) ⊙ h3`

`y = g @ w2_base^T + (α/r) * ((g @ A2_l^T) @ B2_l^T)`

## Component Summary

- Shared across all middle layers:
  - `w1_base`, `w2_base`, `w3_base`

- Per middle layer `l`:
  - `A1_l, B1_l`
  - `A2_l, B2_l`
  - `A3_l, B3_l`

- Kept independent per layer:
  - Attention projections
  - RMSNorm parameters
  - First-layer full MLP
  - Last-layer full MLP
