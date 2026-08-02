"""Fused decode + atomic-scatter Triton kernel that replaces a 5-kernel PyTorch
sequence on polish_emit's (sign, net_idx) markers; grid=(B,), each program
walks MAX_CELLS slots in BLK chunks."""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_scatter_kernel(
    H_idx_ptr,    # (B*MAX_CELLS,) int32
    H_val_ptr,    # (B*MAX_CELLS,) float32 (encoded marker)
    net_w_ptr,    # (num_nets,) float64
    out_ptr,      # (B*ng,) float64 - delta_H_net flattened
    ng: tl.int32,
    MAX_CELLS: tl.constexpr,
    BLK: tl.constexpr,
):
    pid = tl.program_id(0)
    base_in = pid * MAX_CELLS
    base_out = pid * ng

    n_chunks: tl.constexpr = MAX_CELLS // BLK
    for chunk in range(0, n_chunks):
        offs = chunk * BLK + tl.arange(0, BLK)
        h_val = tl.load(H_val_ptr + base_in + offs)
        h_idx = tl.load(H_idx_ptr + base_in + offs)
        # Decode marker = (net_idx + 1) * sign; marker==0 means empty slot.
        nonzero = h_val != 0.0
        sign = tl.where(h_val > 0.0, 1.0, -1.0).to(tl.float64)
        # Clamp net_idx so empty slots never load OOB (masked off anyway).
        net_idx = (tl.abs(h_val) - 1.0).to(tl.int32)
        net_idx = tl.maximum(net_idx, 0)
        weight = tl.load(net_w_ptr + net_idx, mask=nonzero, other=0.0)
        decoded = sign * weight
        tl.atomic_add(out_ptr + base_out + h_idx.to(tl.int64),
                       decoded, mask=nonzero)


def _decode_scatter_torch(H_idx, H_val, net_w_norm, out, MAX_CELLS):
    """Pure-PyTorch deterministic fallback via scatter_add; ~2-3x slower than
    the atomic kernel but bit-stable when torch deterministic mode is on."""
    B, ng = out.shape
    h_idx_2d = H_idx.view(B, MAX_CELLS).to(torch.int64)
    h_val_2d = H_val.view(B, MAX_CELLS)
    nonzero = (h_val_2d != 0.0)
    sign = torch.where(h_val_2d > 0.0, 1.0, -1.0).to(out.dtype)
    net_idx = (h_val_2d.abs() - 1.0).to(torch.int64).clamp(min=0)
    weight = net_w_norm[net_idx]
    decoded = sign * weight * nonzero.to(out.dtype)
    out.scatter_add_(1, h_idx_2d, decoded)


def decode_scatter(H_idx, H_val, net_w_norm, out, MAX_CELLS):
    """Apply decode + scatter into `out` (must be pre-zeroed); falls back to
    deterministic scatter_add when torch deterministic mode is on, because
    GPU atomic_add ordering is nondeterministic."""
    if torch.are_deterministic_algorithms_enabled():
        _decode_scatter_torch(H_idx, H_val, net_w_norm, out, MAX_CELLS)
        return
    B, ng = out.shape
    BLK = 32  # matches EMIT_BLK in polish_emit
    grid = (B,)
    _decode_scatter_kernel[grid](
        H_idx, H_val, net_w_norm, out.view(-1),
        ng=ng,
        MAX_CELLS=MAX_CELLS,
        BLK=BLK,
    )
