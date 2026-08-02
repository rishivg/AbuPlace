"""HPWL gradient helper for basin_jump_v2; computes the deterministic
per-soft-macro bbox-HPWL gradient that seeds the WireMask-style perturbation
chain."""

import numpy as np
import torch


# id(pn)-keyed topology cache: same pn reused across basin-jump rounds, build once.
_NET_PIN_CACHE = {}
# (id(pn), id(wl_extra_weight))-keyed cache to avoid ~20ms*rounds of redundant
# uploads.
_GRAD_GPU_CACHE = {}


def _build_net_pin_membership(pn):
    """Return (pin_idx, net_idx) flat arrays where every pin appears once per
    net it belongs to; cached per `pn` instance."""
    key = id(pn)
    cached = _NET_PIN_CACHE.get(key)
    if cached is not None:
        return cached

    drivers = np.asarray(pn['net_driver'], dtype=np.int64)
    sinks_off = np.asarray(pn['net_sinks_off'], dtype=np.int64)
    sinks_idx = np.asarray(pn['net_sinks_idx'], dtype=np.int64)
    n_nets = len(drivers)

    drv_pin = drivers
    drv_net = np.arange(n_nets, dtype=np.int64)

    # Variable-fanout sinks expanded via repeat.
    sink_counts = np.diff(sinks_off)  # (n_nets,)
    snk_net = np.repeat(np.arange(n_nets, dtype=np.int64), sink_counts)
    snk_pin = sinks_idx

    pin_idx = np.concatenate([drv_pin, snk_pin])
    net_idx = np.concatenate([drv_net, snk_net])
    _NET_PIN_CACHE[key] = (pin_idx, net_idx)
    return pin_idx, net_idx


def compute_hpwl_gradient(cur, sizes, mov_i32, n_macros, nh, pn,
                           wl_extra_weight=None, device="cuda:0"):
    """Deterministic per-soft-macro bbox-HPWL gradient with lowest-pin_idx
    tie-break (computed on CPU because torch scatter_reduce amin/amax
    backward is nondeterministic on CUDA); hard/non-movable rows zeroed;
    returns (grad_tensor, total_hpwl)."""
    cache_key = (id(pn), id(wl_extra_weight))
    cached = _GRAD_GPU_CACHE.get(cache_key)
    if cached is None:
        pin_macro_np = np.asarray(pn['pin_macro'], dtype=np.int64)
        pin_x_off_np = np.asarray(pn['pin_x'], dtype=np.float64)
        pin_y_off_np = np.asarray(pn['pin_y'], dtype=np.float64)
        net_weight_np = np.asarray(pn['net_weight'], dtype=np.float64)
        if wl_extra_weight is not None:
            net_weight_np = net_weight_np * np.asarray(
                wl_extra_weight, dtype=np.float64)
        pin_idx_np, net_idx_np = _build_net_pin_membership(pn)
        n_nets = len(pn['net_driver'])
        n_pins = pin_macro_np.shape[0]
        cached = {
            'pin_macro_np': pin_macro_np,
            'pin_x_off_np': pin_x_off_np,
            'pin_y_off_np': pin_y_off_np,
            'net_weight_np': net_weight_np,
            'pin_idx_np': pin_idx_np,
            'net_idx_np': net_idx_np,
            'n_nets': n_nets,
            'n_pins': n_pins,
        }
        _GRAD_GPU_CACHE[cache_key] = cached
    pin_macro_np = cached['pin_macro_np']
    pin_x_off_np = cached['pin_x_off_np']
    pin_y_off_np = cached['pin_y_off_np']
    net_weight_np = cached['net_weight_np']
    pin_idx_np = cached['pin_idx_np']
    net_idx_np = cached['net_idx_np']
    n_nets = cached['n_nets']
    n_pins = cached['n_pins']

    cur_np = np.ascontiguousarray(cur, dtype=np.float64)

    # Port pins (macro=-1) store absolute coords in pin_x_off/y_off.
    is_port = pin_macro_np < 0
    safe_macro = np.where(is_port, 0, pin_macro_np)
    pin_abs_x = np.where(is_port, pin_x_off_np,
                         cur_np[safe_macro, 0] + pin_x_off_np)
    pin_abs_y = np.where(is_port, pin_y_off_np,
                         cur_np[safe_macro, 1] + pin_y_off_np)

    pa_x = pin_abs_x[pin_idx_np]
    pa_y = pin_abs_y[pin_idx_np]

    # ufunc.at is deterministic - needed because we tie-break on pin_idx below.
    xmin = np.full(n_nets, np.inf, dtype=np.float64)
    xmax = np.full(n_nets, -np.inf, dtype=np.float64)
    ymin = np.full(n_nets, np.inf, dtype=np.float64)
    ymax = np.full(n_nets, -np.inf, dtype=np.float64)
    np.minimum.at(xmin, net_idx_np, pa_x)
    np.maximum.at(xmax, net_idx_np, pa_x)
    np.minimum.at(ymin, net_idx_np, pa_y)
    np.maximum.at(ymax, net_idx_np, pa_y)

    # Canonical pin = lowest pin_idx hitting the per-net min/max; sentinel =
    # n_pins for empty nets.
    sentinel = np.int64(n_pins)
    big_key = np.full(pin_idx_np.shape, sentinel, dtype=np.int64)

    def _canon(values_per_entry, target_per_net):
        is_target = values_per_entry == target_per_net[net_idx_np]
        key = np.where(is_target, pin_idx_np, big_key)
        canon = np.full(n_nets, sentinel, dtype=np.int64)
        np.minimum.at(canon, net_idx_np, key)
        return canon  # (n_nets,) - pin_idx of canonical, or `sentinel`

    canon_xmin = _canon(pa_x, xmin)
    canon_xmax = _canon(pa_x, xmax)
    canon_ymin = _canon(pa_y, ymin)
    canon_ymax = _canon(pa_y, ymax)

    # Per net, canonical pin owns gradient: +w on max, -w on min, attributed to
    # its macro.
    pos_grad = np.zeros_like(cur_np)

    def _accumulate(canon_pin_per_net, sign, axis):
        valid = canon_pin_per_net < sentinel
        if not valid.any():
            return
        pins = canon_pin_per_net[valid]
        macros = pin_macro_np[pins]
        weights = sign * net_weight_np[valid]
        # Skip port pins (macro=-1) - they aren't movable.
        real = macros >= 0
        if not real.any():
            return
        np.add.at(pos_grad[:, axis], macros[real], weights[real])

    _accumulate(canon_xmax, +1.0, 0)
    _accumulate(canon_xmin, -1.0, 0)
    _accumulate(canon_ymax, +1.0, 1)
    _accumulate(canon_ymin, -1.0, 1)

    mov_bool = np.asarray(mov_i32, dtype=np.int32) != 0
    pos_grad[~mov_bool] = 0.0
    if nh > 0:
        pos_grad[:nh] = 0.0

    # Total HPWL also returned for logging / signature parity.
    valid_nets = np.isfinite(xmin) & np.isfinite(xmax)
    bbox_w = np.where(valid_nets, xmax - xmin, 0.0)
    bbox_h = np.where(valid_nets, ymax - ymin, 0.0)
    total_hpwl = float((net_weight_np * (bbox_w + bbox_h)).sum())

    grad = torch.from_numpy(pos_grad).to(device)
    return grad, total_hpwl


