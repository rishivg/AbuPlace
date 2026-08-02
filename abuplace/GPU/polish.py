"""GPU polish_v1 coordinate-descent loop mirroring C cong_relax_v2: phased step
decay with reheats, first-improving accept, cold-mask after iter 0,
best-proxy restore on exit; builds a macro IS partition once so independent
macros within an iter can be evaluated in one batched kernel."""

import os
import time

import numpy as np
import torch

from .build_cands_vec import build_iter_cands_vec
from .candidates import _build_hard_aabb
from .eval_is_batched import _record, eval_is_batched
from .fast_route import CPUPinCache
from .is_partition import build_macro_is_partition


def polish_v1(state, n_iter, lr_bins, extra_phases=None,
              verbose=False, max_reheats=2, step_min=0.25,
              restrict_macros=None):
    """Run polish on `state` (must be init_state'd); extra_phases is list of
    (n_iter, lr_bins); restrict_macros lets a small perturbation reuse
    polish without touching other macros; returns best_proxy."""
    if state.mn_offsets is None:
        state.build_mn_csr()
    # hard_lo/hi depend only on hard macros - cache on state once.
    if getattr(state, "_hard_aabb_cache", None) is None:
        state._hard_aabb_cache = _build_hard_aabb(state)
    hard_lo, hard_hi = state._hard_aabb_cache
    cpu = CPUPinCache(state)

    if restrict_macros is not None:
        soft_list = list(restrict_macros)
    else:
        soft_list = getattr(state, "_soft_list_cache", None)
        if soft_list is None:
            soft_list = state.soft_idx_init.cpu().numpy().tolist()
            state._soft_list_cache = soft_list
    n_soft = len(soft_list)


    # Persistent GPU pin_abs workspace, mutated by accepts; refreshed from
    # state.pin_abs_x/y on each polish_v1 call.
    if hasattr(state, "pin_abs_x") and state.pin_abs_x is not None:
        ws_x = getattr(state, "_pin_abs_ws_x", None)
        if ws_x is None or ws_x.shape != state.pin_abs_x.shape:
            state._pin_abs_ws_x = torch.empty_like(state.pin_abs_x)
            state._pin_abs_ws_y = torch.empty_like(state.pin_abs_y)
        state._pin_abs_ws_x.copy_(state.pin_abs_x)
        state._pin_abs_ws_y.copy_(state.pin_abs_y)
        pin_abs_gpu_x = state._pin_abs_ws_x
        pin_abs_gpu_y = state._pin_abs_ws_y
    else:
        pin_abs_gpu_x = torch.from_numpy(np.ascontiguousarray(
            cpu.pin_abs_x, dtype=np.float64)).to(state.device)
        pin_abs_gpu_y = torch.from_numpy(np.ascontiguousarray(
            cpu.pin_abs_y, dtype=np.float64)).to(state.device)
    pin_abs_gpu = (pin_abs_gpu_x, pin_abs_gpu_y)

    best_snap = state.snapshot()
    best_proxy = float(state.full_cost.item())
    if verbose:
        print(f"  [polish] init: proxy={best_proxy:.6f}, n_soft={n_soft}")

    # XP_DET_TRACE=1 dumps full-precision proxy each iter for bisecting
    # determinism drift.
    det_trace = os.environ.get("XP_DET_TRACE", "0") == "1"
    if det_trace:
        print(f"[DET] polish init: proxy={best_proxy!r} hpwl={float(state.total_hpwl.item())!r}")

    phases = [(n_iter, lr_bins)]
    if extra_phases:
        phases.extend(extra_phases)

    # IS partition depends only on graph topology + soft_list - cache on state.
    check_bbox = os.environ.get("XP_POLISH_IS_BBOX", "0") == "1"
    is_cache_key = (tuple(soft_list), check_bbox)
    is_cache = getattr(state, "_is_partition_cache", None)
    if is_cache is None or is_cache.get("key") != is_cache_key:
        is_partition = build_macro_is_partition(
            cpu, soft_list, state.sizes.cpu().numpy(),
            check_bbox=check_bbox)
        m_to_si = {m: si for si, m in enumerate(soft_list)}
        state._is_partition_cache = {
            "key": is_cache_key,
            "is_partition": is_partition,
            "m_to_si": m_to_si,
        }
    else:
        is_partition = is_cache["is_partition"]
        m_to_si = is_cache["m_to_si"]

    for ph_idx, (cur_n_iter, cur_lr_bins) in enumerate(phases):
        if ph_idx > 0:
            state.restore(best_snap)

        step = float(cur_lr_bins)
        reheats = 0
        # The cold mask mirrors the C polish tier-1 optimization: macros with
        # zero accepts at iteration 0 are skipped from then on.
        cold_mask = [False] * n_soft
        per_macro_iter0_acc = [0] * n_soft

        for it in range(cur_n_iter):
            accepts = 0
            t_iter = time.time()

            _ts = time.perf_counter()
            # Pull grid views once per iter, then share across all IS groups.
            H_final_2d_iter = state.H_final.cpu().numpy().reshape(state.gr, state.gc)
            V_final_2d_iter = state.V_final.cpu().numpy().reshape(state.gr, state.gc)
            grid_occ_2d_iter = state.grid_occupied.cpu().numpy().reshape(state.gr, state.gc)
            iter_grid_views = (H_final_2d_iter, V_final_2d_iter, grid_occ_2d_iter)
            _record('outer_a_grid_pull', time.perf_counter() - _ts)
            _ts = time.perf_counter()

            # One HPWL pull batch per iter shared across all candidate builds below.
            from abuplace.kernels.compute_dirs import compute_hpwl_pull
            if it == 0:
                active_soft = soft_list
            else:
                active_soft = [m for m in soft_list
                               if not cold_mask[m_to_si[m]]]
            iter_hpwl_pull = compute_hpwl_pull(state, cpu, active_soft)
            hpwl_pull_cache = {m: tuple(iter_hpwl_pull[i])
                               for i, m in enumerate(active_soft)}
            _record('outer_b_hpwl_pull', time.perf_counter() - _ts)
            _ts = time.perf_counter()

            # Vectorized per-iteration candidate build, replacing the 38xN
            # per-macro Python loop in eval_is_batched.
            iter_cands = build_iter_cands_vec(
                state, cpu, active_soft, step,
                hard_lo, hard_hi,
                iter_hpwl_pull,
                H_final_2d_iter, V_final_2d_iter, grid_occ_2d_iter)
            _record('outer_c_build_cands', time.perf_counter() - _ts)

            # Per-IS batched eval. Do NOT cache orig across IS groups: macros
            # are net-disjoint within a group but not across them, so a
            # committed move invalidates another group's cached orig (worth
            # +0.04 proxy when tried).
            for is_group in is_partition:
                if it > 0:
                    active = [m for m in is_group
                              if not cold_mask[m_to_si[m]]]
                else:
                    active = list(is_group)
                if not active:
                    continue
                n_acc = eval_is_batched(state, cpu, active, step,
                                        hard_lo, hard_hi,
                                        grid_views=iter_grid_views,
                                        hpwl_pull_cache=hpwl_pull_cache,
                                        precomputed_cands=iter_cands,
                                        pin_abs_gpu=pin_abs_gpu)
                accepts += n_acc
                if it == 0 and n_acc > 0:
                    for m in active:
                        per_macro_iter0_acc[m_to_si[m]] = 1

            # Iteration-end best snapshot. Per-accept gating makes full_cost
            # monotonic across IS groups, so this matches per-group tracking
            # with ~30x fewer .item() syncs.
            cur_full = float(state.full_cost.item())
            if det_trace:
                print(f"[DET] polish ph{ph_idx} it{it}: proxy={cur_full!r} hpwl={float(state.total_hpwl.item())!r} accepts={accepts}")
            if cur_full < best_proxy:
                best_proxy = cur_full
                best_snap = state.snapshot()

            if verbose:
                n_active = sum(1 for c in cold_mask if not c) if it > 0 else n_soft
                print(f"  [polish] ph{ph_idx} it{it}: "
                      f"accepts={accepts}/{n_active} step={step:.3f} "
                      f"proxy={float(state.full_cost.item()):.6f} "
                      f"best={best_proxy:.6f} ({time.time()-t_iter:.1f}s)")

            # Populate the cold mask after iteration 0, but only above a ~5%
            # accept rate. Below that the "cold" majority can still accept at
            # the smaller steps later in the phase.
            if it == 0:
                iter0_accepts = sum(per_macro_iter0_acc)
                if iter0_accepts >= max(4, n_soft // 20):
                    for si in range(n_soft):
                        if per_macro_iter0_acc[si] == 0:
                            cold_mask[si] = True

            if accepts == 0:
                step *= 0.5
                if step < step_min:
                    if reheats < max_reheats:
                        step = float(cur_lr_bins)
                        reheats += 1
                        soft_list = _xorshift_shuffle(soft_list, reheats)
                        cold_mask = [False] * n_soft
                        per_macro_iter0_acc = [0] * n_soft
                    else:
                        break

    state.restore(best_snap)
    return best_proxy


def _xorshift_shuffle(items, seed):
    """Fisher-Yates shuffle driven by the same xorshift32 PRNG as C polish
    reheats."""
    out = list(items)
    s = (0x9E3779B1 ^ (seed * 0x9E3779B1)) & 0xFFFFFFFF
    for i in range(len(out) - 1, 0, -1):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= (s >> 17) & 0xFFFFFFFF
        s ^= (s << 5) & 0xFFFFFFFF
        j = s % (i + 1)
        out[i], out[j] = out[j], out[i]
    return out
