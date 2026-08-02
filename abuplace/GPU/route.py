"""Hard-macro footprint deposit and canonical H_final/V_final smoothing rebuild
used by init_state; net routing and density live in the batched Triton
paths."""

import torch


def route_macro(state, m):
    """Mirror C route_macro: deposit hard-macro footprint into H/V_macro and
    H/V_final via xd*yd partial-overlap scaling plus the C's partial-edge
    correction at ur_row/ur_col."""
    import math
    cx = float(state.pos[m, 0].item())
    cy = float(state.pos[m, 1].item())
    hw = float(state.sizes[m, 0].item()) * 0.5
    hh = float(state.sizes[m, 1].item()) * 0.5
    mlx = cx - hw; mhx = cx + hw
    mly = cy - hh; mhy = cy + hh

    ur_row = int(math.floor(mhy / state.gh))
    ur_col = int(math.floor(mhx / state.gw))
    bl_row = int(math.floor(mly / state.gh))
    bl_col = int(math.floor(mlx / state.gw))

    if ur_row < 0 or ur_col < 0:
        return
    bl_row = max(0, bl_row)
    bl_col = max(0, bl_col)
    ur_row = min(ur_row, state.gr - 1)
    ur_col = min(ur_col, state.gc - 1)
    if bl_row > ur_row or bl_col > ur_col:
        return

    v_scale = state.v_alloc / state.grid_v_routes
    h_scale = state.h_alloc / state.grid_h_routes
    eps = 1e-5

    # CPU-side yd/xd: hard-macro footprints are tiny, GPU dispatch overhead loses.
    yd_list = []
    for r in range(bl_row, ur_row + 1):
        bin_y_lo = r * state.gh
        bin_y_hi = bin_y_lo + state.gh
        v = min(bin_y_hi, mhy) - max(bin_y_lo, mly)
        yd_list.append(max(v, 0.0))
    xd_list = []
    for c in range(bl_col, ur_col + 1):
        bin_x_lo = c * state.gw
        bin_x_hi = bin_x_lo + state.gw
        v = min(bin_x_hi, mhx) - max(bin_x_lo, mlx)
        xd_list.append(max(v, 0.0))

    yd_t = torch.tensor(yd_list, device=state.device, dtype=state.dtype)
    xd_t = torch.tensor(xd_list, device=state.device, dtype=state.dtype)

    # Outer products: v_add[r,c]=xd[c]*v_scale, h_add[r,c]=yd[r]*h_scale.
    v_add = (xd_t * v_scale).unsqueeze(0).expand(yd_t.shape[0], -1).reshape(-1).contiguous()
    h_add = (yd_t * h_scale).unsqueeze(1).expand(-1, xd_t.shape[0]).reshape(-1).contiguous()

    rr = torch.arange(bl_row, ur_row + 1, device=state.device, dtype=state.idtype)
    cc = torch.arange(bl_col, ur_col + 1, device=state.device, dtype=state.idtype)
    flat_idx = (rr.unsqueeze(1) * state.gc + cc.unsqueeze(0)).reshape(-1)
    state.V_macro.scatter_add_(0, flat_idx, v_add)
    state.V_final.scatter_add_(0, flat_idx, v_add)
    state.H_macro.scatter_add_(0, flat_idx, h_add)
    state.H_final.scatter_add_(0, flat_idx, h_add)

    # Partial-edge correction, mirroring C: subtract xd*v_scale at row=ur_row
    # when any vertical edge row is partial, and symmetrically for horizontal.
    partial_v = False
    if ur_row != bl_row:
        if abs(yd_list[0] - state.gh) > eps or abs(yd_list[-1] - state.gh) > eps:
            partial_v = True
    partial_h = False
    if ur_col != bl_col:
        if abs(xd_list[0] - state.gw) > eps or abs(xd_list[-1] - state.gw) > eps:
            partial_h = True

    if partial_v:
        row_flat = ur_row * state.gc + cc
        row_v_neg = -(xd_t * v_scale)
        state.V_macro.scatter_add_(0, row_flat, row_v_neg)
        state.V_final.scatter_add_(0, row_flat, row_v_neg)
    if partial_h:
        col_flat = rr * state.gc + ur_col
        col_h_neg = -(yd_t * h_scale)
        state.H_macro.scatter_add_(0, col_flat, col_h_neg)
        state.H_final.scatter_add_(0, col_flat, col_h_neg)


def compute_smooth_into_final(state):
    """Recompute H_final/V_final = H/V_macro + smoothed H/V_net using the C
    window-spread semantics (V smoothed along cols, H along rows, divisor =
    clamped window size)."""
    sr = state.smooth_range
    gr, gc = state.gr, state.gc

    H_net_2d = state.H_net.view(gr, gc)
    V_net_2d = state.V_net.view(gr, gc)
    H_macro_2d = state.H_macro.view(gr, gc)
    V_macro_2d = state.V_macro.view(gr, gc)

    # The edge-clamped window divisor varies per column, so divide the source
    # first, then run a uniform-1 cumsum convolution.
    cols = torch.arange(gc, device=state.device, dtype=state.dtype)
    lp = torch.clamp(cols - sr, min=0.0)
    rp = torch.clamp(cols + sr, max=float(gc - 1))
    window_v = (rp - lp + 1)  # (gc,)
    V_norm = V_net_2d / window_v.unsqueeze(0)  # broadcast across rows

    # Pad cols by sr, cumsum, then windowed difference gives the centered
    # (2sr+1) sum.
    pad = sr
    V_padded = torch.nn.functional.pad(V_norm, (pad, pad))
    csum = torch.cumsum(V_padded, dim=1)
    csum_zero = torch.cat([torch.zeros((gr, 1), device=state.device, dtype=state.dtype),
                            csum], dim=1)
    V_smoothed = csum_zero[:, 2 * sr + 1: 2 * sr + 1 + gc] - csum_zero[:, :gc]

    # H smoothing: same trick along the row axis.
    rows = torch.arange(gr, device=state.device, dtype=state.dtype)
    lp_r = torch.clamp(rows - sr, min=0.0)
    rp_r = torch.clamp(rows + sr, max=float(gr - 1))
    window_h = (rp_r - lp_r + 1)
    H_norm = H_net_2d / window_h.unsqueeze(1)

    H_padded = torch.nn.functional.pad(H_norm, (0, 0, pad, pad))
    csum_h = torch.cumsum(H_padded, dim=0)
    csum_h_zero = torch.cat([torch.zeros((1, gc), device=state.device, dtype=state.dtype),
                              csum_h], dim=0)
    H_smoothed = csum_h_zero[2 * sr + 1: 2 * sr + 1 + gr, :] - csum_h_zero[:gr, :]

    state.V_final = (V_macro_2d + V_smoothed).reshape(-1).contiguous()
    state.H_final = (H_macro_2d + H_smoothed).reshape(-1).contiguous()
