"""GPU HPWL subset kernel: per probe, recompute HPWL across m's nets if m moved
to (nx, ny); caller differences against the precomputed baseline to get
delta. Note: a vectorized BLK=32 variant was tried and benchmarked ~5%
slower because per-lane pin gathers serialize and outweigh the
lane-utilization win."""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _hpwl_subset_kernel(
    probe_macro_ids,        # int32 [B]
    probe_x_ptr,            # float64 [B]
    probe_y_ptr,            # float64 [B]
    mn_offsets_ptr, mn_net_ids_ptr,
    net_driver_ptr, net_sinks_off_ptr, net_sinks_idx_ptr, net_weight_ptr,
    pin_macro_ptr, pin_x_off_ptr, pin_y_off_ptr,
    pin_abs_x_ptr, pin_abs_y_ptr,   # current abs positions (cpu mirror on GPU)
    out_hpwl_ptr,           # float64 [B] output: HPWL subset for m at probe pos
    MAX_NETS_PER_MACRO: tl.constexpr,
    MAX_PINS_PER_NET: tl.constexpr,
):
    """Per probe (pid): compute HPWL of m's nets if m is at probe pos."""
    pid = tl.program_id(0)
    m = tl.load(probe_macro_ids + pid).to(tl.int32)
    new_x = tl.load(probe_x_ptr + pid)
    new_y = tl.load(probe_y_ptr + pid)

    mn_off = tl.load(mn_offsets_ptr + m).to(tl.int32)
    mn_end = tl.load(mn_offsets_ptr + m + 1).to(tl.int32)
    n_iters = mn_end - mn_off

    total_hpwl = tl.full((), 0.0, dtype=tl.float64)

    for ki in range(0, MAX_NETS_PER_MACRO):
        if ki < n_iters:
            ni = tl.load(mn_net_ids_ptr + mn_off + ki).to(tl.int32)
            d_pin = tl.load(net_driver_ptr + ni).to(tl.int32)
            s_off = tl.load(net_sinks_off_ptr + ni).to(tl.int32)
            s_end = tl.load(net_sinks_off_ptr + ni + 1).to(tl.int32)
            w = tl.load(net_weight_ptr + ni)
            n_sinks = s_end - s_off

            # Driver pin abs position.
            d_macro = tl.load(pin_macro_ptr + d_pin).to(tl.int32)
            d_off_x = tl.load(pin_x_off_ptr + d_pin)
            d_off_y = tl.load(pin_y_off_ptr + d_pin)
            d_is_m = d_macro == m
            d_is_port = d_macro < 0
            # Ports store absolute pin loc in offsets; macros use center+offset.
            stored_d_x = tl.load(pin_abs_x_ptr + d_pin)
            stored_d_y = tl.load(pin_abs_y_ptr + d_pin)
            d_abs_x = tl.where(d_is_m, new_x + d_off_x, stored_d_x)
            d_abs_y = tl.where(d_is_m, new_y + d_off_y, stored_d_y)
            d_abs_x = tl.where(d_is_port, d_off_x, d_abs_x)
            d_abs_y = tl.where(d_is_port, d_off_y, d_abs_y)

            xmin = d_abs_x; xmax = d_abs_x
            ymin = d_abs_y; ymax = d_abs_y

            for si in range(0, MAX_PINS_PER_NET):
                if si < n_sinks:
                    p = tl.load(net_sinks_idx_ptr + s_off + si).to(tl.int32)
                    p_macro = tl.load(pin_macro_ptr + p).to(tl.int32)
                    p_off_x = tl.load(pin_x_off_ptr + p)
                    p_off_y = tl.load(pin_y_off_ptr + p)
                    p_is_m = p_macro == m
                    p_is_port = p_macro < 0
                    stored_p_x = tl.load(pin_abs_x_ptr + p)
                    stored_p_y = tl.load(pin_abs_y_ptr + p)
                    p_abs_x = tl.where(p_is_m, new_x + p_off_x, stored_p_x)
                    p_abs_y = tl.where(p_is_m, new_y + p_off_y, stored_p_y)
                    p_abs_x = tl.where(p_is_port, p_off_x, p_abs_x)
                    p_abs_y = tl.where(p_is_port, p_off_y, p_abs_y)

                    xmin = tl.minimum(xmin, p_abs_x)
                    xmax = tl.maximum(xmax, p_abs_x)
                    ymin = tl.minimum(ymin, p_abs_y)
                    ymax = tl.maximum(ymax, p_abs_y)

            net_hpwl = w * ((xmax - xmin) + (ymax - ymin))
            total_hpwl = total_hpwl + net_hpwl

    tl.store(out_hpwl_ptr + pid, total_hpwl)


def hpwl_subset_at(state, cpu, macro_ids, positions, *,
                    pin_abs_x_gpu=None, pin_abs_y_gpu=None,
                    MAX_NETS_PER_MACRO=None, MAX_PINS_PER_NET=None):
    """Compute HPWL subset (sum over m's nets) for B probes; optional
    pin_abs_x/y_gpu skip the host->device transfer; returns (B,) float64 GPU
    tensor.

    The loop bounds default to the autotuned values derived from this design's
    topology. They must not be hardcoded: the per-macro net loop is a compile-
    time trip count guarded by `if ki < n_iters`, so a bound below the real
    maximum silently drops a macro's remaining nets from the HPWL instead of
    failing. The largest IBM designs have macros on well over 256 nets."""
    if MAX_NETS_PER_MACRO is None or MAX_PINS_PER_NET is None:
        from abuplace.GPU.polish_kernel_tune import autotune_kernel_bounds
        bnd = autotune_kernel_bounds(state)
        MAX_NETS_PER_MACRO = MAX_NETS_PER_MACRO or bnd["MAX_NETS_PER_MACRO"]
        MAX_PINS_PER_NET = MAX_PINS_PER_NET or bnd["MAX_PINS_PER_NET"]
    device = state.device

    if pin_abs_x_gpu is None:
        pin_abs_x_gpu = torch.from_numpy(np.ascontiguousarray(
            cpu.pin_abs_x, dtype=np.float64)).to(device)
        pin_abs_y_gpu = torch.from_numpy(np.ascontiguousarray(
            cpu.pin_abs_y, dtype=np.float64)).to(device)

    if isinstance(macro_ids, np.ndarray):
        macro_ids_np = macro_ids if macro_ids.dtype == np.int32 \
            else macro_ids.astype(np.int32)
    else:
        macro_ids_np = np.asarray(macro_ids, dtype=np.int32)
    B = len(macro_ids_np)
    if B == 0:
        return torch.zeros(0, device=device, dtype=torch.float64)

    if isinstance(positions, np.ndarray):
        if positions.dtype != np.float64:
            positions = positions.astype(np.float64)
        new_x_np = np.ascontiguousarray(positions[:, 0])
        new_y_np = np.ascontiguousarray(positions[:, 1])
    else:
        pos_np = np.asarray(positions, dtype=np.float64)
        new_x_np = np.ascontiguousarray(pos_np[:, 0])
        new_y_np = np.ascontiguousarray(pos_np[:, 1])
    macro_ids_t = torch.from_numpy(macro_ids_np).to(device)
    px = torch.from_numpy(new_x_np).to(device)
    py = torch.from_numpy(new_y_np).to(device)
    out = torch.zeros(B, device=device, dtype=torch.float64)

    grid = (B,)
    _hpwl_subset_kernel[grid](
        macro_ids_t, px, py,
        state.mn_offsets, state.mn_net_ids,
        state.net_driver, state.net_sinks_off, state.net_sinks_idx,
        state.net_weight,
        state.pin_macro, state.pin_x_off, state.pin_y_off,
        pin_abs_x_gpu, pin_abs_y_gpu,
        out,
        MAX_NETS_PER_MACRO=MAX_NETS_PER_MACRO,
        MAX_PINS_PER_NET=MAX_PINS_PER_NET,
    )
    return out
