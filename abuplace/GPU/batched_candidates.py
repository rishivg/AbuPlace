"""Batched grid helpers shared by the polish/eval pipelines: per-macro sparse
density delta plus cuDNN conv1d smoothing and top-k cong/density reductions
over (B, ng) buffers."""

import math

import torch
import torch.nn.functional as F


_SMOOTH_KERNEL_CACHE = {}
_WINDOW_CACHE = {}


def _get_smooth_kernel(sr, dtype, device):
    """Cached all-ones 1D conv kernel of size 2*sr+1."""
    key = (sr, dtype, device)
    k = _SMOOTH_KERNEL_CACHE.get(key)
    if k is None:
        k = torch.ones(1, 1, 2 * sr + 1, dtype=dtype, device=device)
        _SMOOTH_KERNEL_CACHE[key] = k
    return k


def _get_window(n, sr, dtype, device):
    key = (n, sr, dtype, device)
    w = _WINDOW_CACHE.get(key)
    if w is None:
        rows = torch.arange(n, device=device, dtype=dtype)
        lp = torch.clamp(rows - sr, min=0.0)
        rp = torch.clamp(rows + sr, max=float(n - 1))
        w = (rp - lp + 1)
        _WINDOW_CACHE[key] = w
    return w


def batched_smooth_hnet_to_hfinal(H_net, smooth_range, gr, gc):
    """Smooth H_net (B, ng) along the row dim via cuDNN conv1d (~+28% vs the
    equivalent Triton kernel - cuDNN is the shipping path)."""
    sr = smooth_range
    B = H_net.shape[0]
    dtype = H_net.dtype
    device = H_net.device
    window_h = _get_window(gr, sr, dtype, device)
    H_norm_3d = H_net.view(B, gr, gc) / window_h.view(1, gr, 1)
    inp = H_norm_3d.permute(0, 2, 1).contiguous().view(B * gc, 1, gr)
    kernel = _get_smooth_kernel(sr, dtype, device)
    out = F.conv1d(inp, kernel, padding=sr)
    return out.view(B, gc, gr).permute(0, 2, 1).reshape(B, gr * gc)


def batched_smooth_vnet_to_vfinal(V_net, smooth_range, gr, gc):
    """Smooth V_net (B, ng) along the col dim via cuDNN conv1d; see hnet
    variant."""
    sr = smooth_range
    B = V_net.shape[0]
    dtype = V_net.dtype
    device = V_net.device
    window_v = _get_window(gc, sr, dtype, device)
    V_norm_3d = V_net.view(B, gr, gc) / window_v.view(1, 1, gc)
    inp = V_norm_3d.view(B * gr, 1, gc)
    kernel = _get_smooth_kernel(sr, dtype, device)
    out = F.conv1d(inp, kernel, padding=sr)
    return out.view(B, gr, gc).reshape(B, gr * gc)


def batched_compute_cong(V_final, H_final):
    """Per-row top-5% mean of (V_final||H_final); inputs (B, ng), output (B,)."""
    cat = torch.cat([V_final, H_final], dim=1)
    total = cat.shape[1]
    cnt = max(1, int(math.floor(total * 0.05)))
    top, _ = torch.topk(cat, cnt, dim=1, largest=True, sorted=False)
    return top.mean(dim=1)


def batched_compute_density(grid_occupied, grid_area):
    cnt = max(1, int(math.floor(grid_occupied.shape[1] * 0.10)))
    top, _ = torch.topk(grid_occupied, cnt, dim=1, largest=True, sorted=False)
    return 0.5 * top.mean(dim=1) / grid_area
