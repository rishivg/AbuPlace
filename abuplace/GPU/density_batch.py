"""Batched density deposit for all macros at once; replaces a per-macro Python
loop (4 .item() syncs/macro * ~1438 macros) with one fully on-GPU
broadcasted area computation."""

import torch


def batched_density_init(state):
    """Add ALL macros' bbox-cell overlap area into state.grid_occupied in one
    batched op (caller must zero grid_occupied first)."""
    device = state.device
    dtype = state.dtype
    gr = state.gr
    gc = state.gc
    gw = state.gw
    gh = state.gh

    pos = state.pos                # (n, 2)
    sizes = state.sizes            # (n, 2)
    cx = pos[:, 0]                 # (n,)
    cy = pos[:, 1]
    hw = sizes[:, 0] * 0.5
    hh = sizes[:, 1] * 0.5
    mlx = cx - hw                  # (n,)
    mhx = cx + hw
    mly = cy - hh
    mhy = cy + hh

    bl_row = torch.clamp(torch.floor(mly / gh).to(torch.int64), min=0, max=gr - 1)
    ur_row = torch.clamp(torch.floor(mhy / gh).to(torch.int64), min=0, max=gr - 1)
    bl_col = torch.clamp(torch.floor(mlx / gw).to(torch.int64), min=0, max=gc - 1)
    ur_col = torch.clamp(torch.floor(mhx / gw).to(torch.int64), min=0, max=gc - 1)

    # Match density_macro's out-of-grid early return by collapsing the bbox to
    # empty; clamping instead would over-deposit.
    valid = (mhy >= 0) & (mhx >= 0) & (mly < gh * gr) & (mlx < gw * gc)
    if not valid.all():
        bl_row = torch.where(valid, bl_row, ur_row + 1)
        bl_col = torch.where(valid, bl_col, ur_col + 1)

    # Per-cell overlap via the outer product yd[n,gr] * xd[n,gc] - about
    # n*gr*gc float64s, which is fine at this problem size.
    rows = torch.arange(gr, device=device, dtype=dtype)  # (gr,)
    cols = torch.arange(gc, device=device, dtype=dtype)  # (gc,)
    bin_y_lo = rows * gh                                  # (gr,)
    bin_y_hi = bin_y_lo + gh
    bin_x_lo = cols * gw                                  # (gc,)
    bin_x_hi = bin_x_lo + gw

    yd = (torch.minimum(bin_y_hi[None, :], mhy[:, None])
          - torch.maximum(bin_y_lo[None, :], mly[:, None])).clamp(min=0.0)
    xd = (torch.minimum(bin_x_hi[None, :], mhx[:, None])
          - torch.maximum(bin_x_lo[None, :], mlx[:, None])).clamp(min=0.0)
    area = yd[:, :, None] * xd[:, None, :]                # (n, gr, gc)
    grid_delta = area.sum(dim=0).reshape(-1)
    state.grid_occupied += grid_delta
