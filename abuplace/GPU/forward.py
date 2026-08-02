"""Forward computations (cong / density / hpwl / proxy); ABU top-K uses
torch.topk, which matches C's min-heap top-floor(N*frac).mean modulo
tie-ordering (mean is identical)."""

import math
import torch


def abu_top_n(values, n_frac):
    """Mean of the floor(N*n_frac) largest entries of `values`; returns max()
    when cnt <= 0 (matches C abu_top_n)."""
    total = int(values.shape[0])
    cnt = int(math.floor(total * n_frac))
    if cnt <= 0:
        return values.max()
    top, _ = torch.topk(values, cnt, largest=True, sorted=False)
    return top.mean()


def compute_cong(state):
    """Top-5% mean of V_final concatenated with H_final (matches C
    compute_cong)."""
    cat = torch.cat([state.V_final, state.H_final], dim=0)
    return abu_top_n(cat, 0.05)


def compute_density(state):
    """Scalar 0.5 * abu_top_n(grid_occupied, 0.10) / grid_area."""
    a = abu_top_n(state.grid_occupied, 0.10)
    return 0.5 * a / state.grid_area


def compute_all_net_hpwl(state):
    """Vectorized per-net bbox + HPWL via scatter_reduce on pin_to_net; updates
    state.net_x/y_min/max and state.net_hpwl in place and returns total
    HPWL."""
    big = torch.finfo(state.dtype).max
    nn = state.nn
    device = state.device
    dtype = state.dtype

    ref_x = state.pin_abs_x[state.net_pin_idx]
    ref_y = state.pin_abs_y[state.net_pin_idx]

    x_min = torch.full((nn,), big, device=device, dtype=dtype)
    x_max = torch.full((nn,), -big, device=device, dtype=dtype)
    y_min = torch.full((nn,), big, device=device, dtype=dtype)
    y_max = torch.full((nn,), -big, device=device, dtype=dtype)
    x_min.scatter_reduce_(0, state.pin_to_net, ref_x, reduce='amin')
    x_max.scatter_reduce_(0, state.pin_to_net, ref_x, reduce='amax')
    y_min.scatter_reduce_(0, state.pin_to_net, ref_y, reduce='amin')
    y_max.scatter_reduce_(0, state.pin_to_net, ref_y, reduce='amax')

    # Zero out empty-net entries - sentinel values would corrupt the sum.
    z = torch.zeros((), device=device, dtype=dtype)
    x_min = torch.where(state.net_has_pins, x_min, z)
    x_max = torch.where(state.net_has_pins, x_max, z)
    y_min = torch.where(state.net_has_pins, y_min, z)
    y_max = torch.where(state.net_has_pins, y_max, z)

    state.net_xmin = x_min
    state.net_xmax = x_max
    state.net_ymin = y_min
    state.net_ymax = y_max

    bbox_span = (x_max - x_min) + (y_max - y_min)
    bbox_span = torch.where(state.net_has_pins, bbox_span, z)
    state.net_hpwl = state.net_weight * bbox_span
    total = state.net_hpwl.sum()
    state.total_hpwl = total
    return total



def update_proxy_components(state):
    """Recompute cong/density/wl/full proxy from current grids and HPWL, store
    each on state, and return the full proxy scalar."""
    state.cong_cost = compute_cong(state)
    state.density_cost = compute_density(state)
    if state.hpwl_norm > 0.0:
        state.wl_cost = state.total_hpwl / state.hpwl_norm
    else:
        state.wl_cost = torch.zeros_like(state.total_hpwl)
    state.full_cost = state.wl_cost + 0.5 * state.density_cost + 0.5 * state.cong_cost
    return state.full_cost
