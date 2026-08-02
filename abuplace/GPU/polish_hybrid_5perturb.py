"""GPU+5perturb hybrid polish entry point: full init polish, then 5x (Gaussian
shift K random softs -> init_state -> brief polish), then optional C-short
cong_relax_v2 refine; K_perturb=max(50, round(0.087*n_soft)), sigma=1 gcell,
5 fixed seeds."""

import os
import time

import numpy as np
import torch

from .init import init_state
from .polish import polish_v1
from .state import GPUState
from .polish_kernel_tune import autotune_kernel_bounds


# Schedule defaults match C polish defaults.
_INIT_N_ITER = 12
_INIT_LR_BINS = 2.0
_INIT_EXTRAS = [(6, 1.5), (6, 0.5), (6, 0.15), (6, 0.05)]
_BRIEF_N_ITER = 2
_BRIEF_LR_BINS = 0.5
_BRIEF_EXTRAS = [(2, 0.05)]
_C_SHORT_N_ITER = 3
_C_SHORT_LR_BINS = 1.0
_C_SHORT_EXTRAS = [(3, 0.15), (3, 0.05)]
_N_PERTURBS = 5
_PERTURB_SIGMA_BINS = 1.0
_PERTURB_K_FRAC = 0.087  # K ~ round(0.087 * n_soft); ibm03 -> 100.
_PERTURB_K_MIN = 50      # floor for tiny-soft benchmarks.
_PERTURB_K_MAX = 200     # cap; sweep showed K=200 regressed.
_PERTURB_SEEDS = [42, 1337, 7, 11, 23]


def gpu_hybrid_5perturb_polish(
    cur_pos, sizes, mov_i32, n_macros, nh, cw, ch, octx, pn,
    *,
    c_short_call=None,    # callable(pos) -> (new_pos, ..., proxy_final, ...)
    n_perturbs=_N_PERTURBS,
    K_perturb=None,
    sigma_bins=_PERTURB_SIGMA_BINS,
    seeds=None,
    verbose=False,
    state=None,           # pre-built GPUState (saves ~3.4s setup); refreshed in-place
):
    """Run GPU + 5-perturb hybrid polish; returns the same 7-tuple as
    _cong_relax_v2_c (drop-in for _run_cong_c). If c_short_call is None the
    GPU-only result is returned after a canonical init_state rebuild."""
    n_soft = n_macros - nh
    if K_perturb is None:
        K_perturb = max(_PERTURB_K_MIN,
                         min(_PERTURB_K_MAX,
                             int(round(_PERTURB_K_FRAC * n_soft))))
    if seeds is None:
        seeds = _PERTURB_SEEDS

    t_total = time.time()

    # Build a fresh GPUState or refresh the caller-provided one in place.
    t0 = time.time()
    owns_state = state is None
    if owns_state:
        s = GPUState.from_pn(
            pn=pn, pin_csr=octx['pin_csr'], octx=octx,
            sizes_np=sizes, positions_np=cur_pos.copy(),
            movable_np=mov_i32,
            n_macros=n_macros, n_hard=nh)
        init_state(s)
    else:
        s = state
        s.pos.copy_(torch.from_numpy(np.ascontiguousarray(
            cur_pos, dtype=np.float64)).to(s.device, s.dtype))
        init_state(s)
    setup_wall = time.time() - t0
    # Kernel bounds depend only on static topology+sizes - cache on state.
    bounds = getattr(s, "_autotune_bounds_cache", None)
    if bounds is None:
        bounds = autotune_kernel_bounds(s)
        s._autotune_bounds_cache = bounds
    if verbose:
        print(f"  [hybrid5p] config: n_soft={n_soft} K_perturb={K_perturb} "
              f"sigma={sigma_bins:.2f}gcell n_perturbs={n_perturbs}")
        print(f"  [hybrid5p] kernel autotune: nets={bounds['MAX_NETS_PER_MACRO']} "
              f"pins={bounds['MAX_PINS_PER_NET']} slots={bounds['SLOTS']} "
              f"cells={bounds['MAX_CELLS']}  "
              f"observed: max_nets={bounds['_max_nets_observed']} "
              f"max_pins={bounds['_max_pins_observed']}")
        print(f"  [hybrid5p] GPU state setup: {setup_wall:.2f}s "
              f"(GPUState.from_pn + init_state)")

    det_trace = os.environ.get("XP_DET_TRACE", "0") == "1"
    if det_trace:
        print(f"[DET] hybrid5p start: pos_sum={float(s.pos.sum().item())!r} cur_pos_sum={float(np.asarray(cur_pos, dtype=np.float64).sum())!r} state_proxy={float(s.full_cost.item())!r}")
    t0 = time.time()
    polish_v1(s, n_iter=_INIT_N_ITER, lr_bins=_INIT_LR_BINS,
              extra_phases=list(_INIT_EXTRAS), verbose=False)
    init_wall = time.time() - t0
    init_proxy = float(s.full_cost.item())
    if det_trace:
        print(f"[DET] hybrid5p after init polish: proxy={init_proxy!r}")
    if verbose:
        print(f"  [hybrid5p] init polish ({_INIT_N_ITER}+24 iters): "
              f"{init_wall:.2f}s proxy={init_proxy:.4f}")

    soft_indices = np.where(mov_i32[nh:] != 0)[0] + nh
    gw_unit = cw / octx['gc']
    gh_unit = ch / octx['gr']

    # Persistent GPU perturb buffers sized to the largest actual_K; copy_ in
    # place each iteration to avoid fresh from_numpy().to() allocations.
    actual_K_max = min(K_perturb, len(soft_indices))
    pbuf = getattr(s, "_perturb_buffers", None)
    if (pbuf is None or pbuf['idx'].numel() < actual_K_max
            or pbuf['idx'].device != s.device):
        pbuf = {
            'idx':    torch.empty(actual_K_max, dtype=torch.int64, device=s.device),
            'noise_x':torch.empty(actual_K_max, dtype=s.dtype, device=s.device),
            'noise_y':torch.empty(actual_K_max, dtype=s.dtype, device=s.device),
            'half_w': torch.empty(actual_K_max, dtype=s.dtype, device=s.device),
            'half_h': torch.empty(actual_K_max, dtype=s.dtype, device=s.device),
        }
        s._perturb_buffers = pbuf

    # Always hand C-short the LAST perturb's state. Tracking best-by-proxy
    # instead cost ~0.003 final proxy: the intermediate proxy does not predict
    # what C-short will do with the state.
    t_pert = time.time()
    perturb_proxies = []
    for k in range(n_perturbs):
        seed = seeds[k % len(seeds)]
        rng = np.random.default_rng(seed)
        actual_K = min(K_perturb, len(soft_indices))
        # The RNG stays on CPU (numpy) for bit-exact reproducibility; only the
        # K-sized tensors are uploaded.
        perturb_set = rng.choice(soft_indices, size=actual_K, replace=False)
        noise_x_np = rng.normal(0, sigma_bins * gw_unit, actual_K)
        noise_y_np = rng.normal(0, sigma_bins * gh_unit, actual_K)
        idx_t = pbuf['idx'][:actual_K]
        nx_t = pbuf['noise_x'][:actual_K]
        ny_t = pbuf['noise_y'][:actual_K]
        hw_t = pbuf['half_w'][:actual_K]
        hh_t = pbuf['half_h'][:actual_K]
        idx_t.copy_(torch.from_numpy(perturb_set.astype(np.int64)),
                    non_blocking=True)
        nx_t.copy_(torch.from_numpy(noise_x_np), non_blocking=True)
        ny_t.copy_(torch.from_numpy(noise_y_np), non_blocking=True)
        hw_t.copy_(torch.from_numpy(sizes[perturb_set, 0] / 2.0),
                   non_blocking=True)
        hh_t.copy_(torch.from_numpy(sizes[perturb_set, 1] / 2.0),
                   non_blocking=True)
        # GPU add + clamp in float64 is bit-exact vs the prior numpy +=/np.clip
        # path.
        new_x = torch.clamp(
            s.pos[idx_t, 0] + nx_t, hw_t, cw - hw_t)
        new_y = torch.clamp(
            s.pos[idx_t, 1] + ny_t, hh_t, ch - hh_t)
        s.pos[idx_t, 0] = new_x
        s.pos[idx_t, 1] = new_y
        init_state(s)
        polish_v1(s, n_iter=_BRIEF_N_ITER, lr_bins=_BRIEF_LR_BINS,
                  extra_phases=list(_BRIEF_EXTRAS), verbose=False)
        perturb_proxies.append(float(s.full_cost.item()))
        if det_trace:
            print(f"[DET] hybrid5p perturb k={k} (seed={seed}): proxy={float(s.full_cost.item())!r}")
    pert_wall = time.time() - t_pert
    pert_proxy = perturb_proxies[-1] if perturb_proxies else init_proxy
    if verbose:
        proxies_str = " ".join(f"{p:.4f}" for p in perturb_proxies)
        print(f"  [hybrid5p] {n_perturbs}x perturb-and-brief "
              f"(K={K_perturb}, sigma={sigma_bins:.2f}gcell, "
              f"seeds={seeds[:n_perturbs]}): {pert_wall:.2f}s")
        print(f"  [hybrid5p]   per-step proxy: {proxies_str} "
              f"(final={pert_proxy:.4f})")

    final_pos = s.pos.cpu().numpy().copy()

    # GPU-only path: the incremental polish grids drift (~0.07 proxy on the
    # largest designs), so rebuild canonically via init_state before returning.
    # That keeps bit-exact agreement with the C n_iter=0 eval for 20-100ms.
    if c_short_call is None:
        init_state(s)
        pert_proxy = float(s.full_cost.item())
        if owns_state:
            del s
            torch.cuda.empty_cache()
        return (final_pos, 0.0, 0.0, init_proxy, pert_proxy, 0.0, 0.0)

    if owns_state:
        del s
        torch.cuda.empty_cache()

    t_c = time.time()
    result = c_short_call(final_pos)
    c_wall = time.time() - t_c
    if verbose:
        total_wall = time.time() - t_total
        c_short_init_proxy = result[3]
        c_short_final_proxy = result[4]
        c_short_delta = c_short_final_proxy - c_short_init_proxy
        print(f"  [hybrid5p] C-short refine "
              f"(n_iter=3 lr=1.0 +(3,0.15)+(3,0.05)): {c_wall:.2f}s")
        print(f"  [hybrid5p]   proxy {c_short_init_proxy:.4f} -> "
              f"{c_short_final_proxy:.4f} ({c_short_delta:+.4f})")
        print(f"  [hybrid5p] TOTAL: {total_wall:.2f}s "
              f"= setup {setup_wall:.2f} + init {init_wall:.2f} + "
              f"perturbs {pert_wall:.2f} + C-short {c_wall:.2f}")
        print(f"  [hybrid5p] proxy trajectory: init {init_proxy:.4f} -> "
              f"perturbs {pert_proxy:.4f} -> final {c_short_final_proxy:.4f}")
    return result
