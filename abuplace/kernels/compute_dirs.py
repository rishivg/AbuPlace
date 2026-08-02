"""GPU compute_dirs kernel for polish smart-cand HPWL pull (one program per
macro, density/cong neighborhood reads stay in numpy because they are only 8
cells)."""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_hpwl_pull_kernel(
    macro_ids,              # int32 [M]
    macro_pos_x_ptr,        # float64 [n_macros]
    macro_pos_y_ptr,        # float64 [n_macros]
    mn_offsets_ptr, mn_net_ids_ptr,
    net_driver_ptr, net_sinks_off_ptr, net_sinks_idx_ptr, net_weight_ptr,
    pin_macro_ptr, pin_x_off_ptr, pin_y_off_ptr,
    pin_abs_x_ptr, pin_abs_y_ptr,
    out_pull_x_ptr,         # float64 [M] output
    out_pull_y_ptr,
    MAX_NETS_PER_MACRO: tl.constexpr,
    MAX_PINS_PER_NET: tl.constexpr,
):
    """For each macro, sum weighted pull toward centroid of other pins."""
    pid = tl.program_id(0)
    m = tl.load(macro_ids + pid).to(tl.int32)
    cx = tl.load(macro_pos_x_ptr + m)
    cy = tl.load(macro_pos_y_ptr + m)

    mn_off = tl.load(mn_offsets_ptr + m).to(tl.int32)
    mn_end = tl.load(mn_offsets_ptr + m + 1).to(tl.int32)
    n_iters = mn_end - mn_off

    pull_x = tl.full((), 0.0, dtype=tl.float64)
    pull_y = tl.full((), 0.0, dtype=tl.float64)
    total_w = tl.full((), 0.0, dtype=tl.float64)

    for ki in range(0, MAX_NETS_PER_MACRO):
        if ki < n_iters:
            ni = tl.load(mn_net_ids_ptr + mn_off + ki).to(tl.int32)
            d_pin = tl.load(net_driver_ptr + ni).to(tl.int32)
            s_off = tl.load(net_sinks_off_ptr + ni).to(tl.int32)
            s_end = tl.load(net_sinks_off_ptr + ni + 1).to(tl.int32)
            w = tl.load(net_weight_ptr + ni)
            n_sinks = s_end - s_off

            sum_x = tl.full((), 0.0, dtype=tl.float64)
            sum_y = tl.full((), 0.0, dtype=tl.float64)
            count = tl.full((), 0.0, dtype=tl.float64)

            # Driver pin if not on m.
            d_macro = tl.load(pin_macro_ptr + d_pin).to(tl.int32)
            if d_macro != m:
                sum_x = sum_x + tl.load(pin_abs_x_ptr + d_pin)
                sum_y = sum_y + tl.load(pin_abs_y_ptr + d_pin)
                count = count + 1.0

            for si in range(0, MAX_PINS_PER_NET):
                if si < n_sinks:
                    p = tl.load(net_sinks_idx_ptr + s_off + si).to(tl.int32)
                    p_macro = tl.load(pin_macro_ptr + p).to(tl.int32)
                    if p_macro != m:
                        sum_x = sum_x + tl.load(pin_abs_x_ptr + p)
                        sum_y = sum_y + tl.load(pin_abs_y_ptr + p)
                        count = count + 1.0

            if count > 0.5:
                cxn = sum_x / count
                cyn = sum_y / count
                pull_x = pull_x + w * (cxn - cx)
                pull_y = pull_y + w * (cyn - cy)
                total_w = total_w + w

    # Normalize by total weight.
    if total_w > 1e-12:
        pull_x = pull_x / total_w
        pull_y = pull_y / total_w

    tl.store(out_pull_x_ptr + pid, pull_x)
    tl.store(out_pull_y_ptr + pid, pull_y)


def compute_hpwl_pull(state, cpu, macro_ids, *,
                       pin_abs_x_gpu=None, pin_abs_y_gpu=None,
                       MAX_NETS_PER_MACRO=None, MAX_PINS_PER_NET=None):
    """Compute HPWL pull (centroid_of_others - m_pos) per macro; returns (M, 2)
    numpy array (NOT unit-normalized).

    Loop bounds default to the autotuned values for this design; see the note in
    hpwl_delta.hpwl_subset_at for why they must never be hardcoded."""
    if MAX_NETS_PER_MACRO is None or MAX_PINS_PER_NET is None:
        from abuplace.GPU.polish_kernel_tune import autotune_kernel_bounds
        bnd = autotune_kernel_bounds(state)
        MAX_NETS_PER_MACRO = MAX_NETS_PER_MACRO or bnd["MAX_NETS_PER_MACRO"]
        MAX_PINS_PER_NET = MAX_PINS_PER_NET or bnd["MAX_PINS_PER_NET"]
    device = state.device
    M = len(macro_ids)
    if M == 0:
        return np.zeros((0, 2), dtype=np.float64)

    if pin_abs_x_gpu is None:
        pin_abs_x_gpu = torch.from_numpy(np.ascontiguousarray(
            cpu.pin_abs_x, dtype=np.float64)).to(device)
        pin_abs_y_gpu = torch.from_numpy(np.ascontiguousarray(
            cpu.pin_abs_y, dtype=np.float64)).to(device)

    macro_ids_t = torch.tensor(macro_ids, dtype=torch.int32, device=device)
    out_x = torch.zeros(M, device=device, dtype=torch.float64)
    out_y = torch.zeros(M, device=device, dtype=torch.float64)

    grid = (M,)
    _compute_hpwl_pull_kernel[grid](
        macro_ids_t,
        state.pos[:, 0].contiguous(),
        state.pos[:, 1].contiguous(),
        state.mn_offsets, state.mn_net_ids,
        state.net_driver, state.net_sinks_off, state.net_sinks_idx,
        state.net_weight,
        state.pin_macro, state.pin_x_off, state.pin_y_off,
        pin_abs_x_gpu, pin_abs_y_gpu,
        out_x, out_y,
        MAX_NETS_PER_MACRO=MAX_NETS_PER_MACRO,
        MAX_PINS_PER_NET=MAX_PINS_PER_NET,
    )
    return torch.stack([out_x, out_y], dim=1).cpu().numpy()
