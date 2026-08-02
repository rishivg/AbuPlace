"""basin_jump_v3: chain basin_jump_v2 GPU rounds with a CPU operator stack
(HM+swap+SM ladders, window-reorder, legalize, iterated soft sweeps, WM-grad
micro-perturb, final polish); drop-in replacement for v2 with the same
signature/return shape; pass gpu_state to skip ~3.4s GPUState.from_pn."""

import torch
import os
import time
from itertools import permutations

import numpy as np

from .basin_jump import compute_hpwl_gradient


_DEFAULT_BUDGET_S = float(os.environ.get("XP_BJ_BUDGET", "35.0"))


def _call_legalize(legalize_fn, pos_hard, label):
    """Call legalize_fn with an optional per-pass `label` for placer
    instrumentation; falls back when the underlying lambda doesn't accept
    the kwarg."""
    try:
        return legalize_fn(pos_hard, label=label)
    except TypeError:
        return legalize_fn(pos_hard)


# Operator passes (HM / SM / pair-swap / window-reorder) live below.

# Must match legalize.c's overlap EPS, or sub-cell hardmove commits get
# displaced by the next legalize pass and the proxy regresses.
_LEGALIZE_EPS = 0.01


def _hardmove_pass(cur, sizes, cw, ch, cs, hard_ord, moves, deadline,
                    hard_aabb):
    base_p, _, _, _ = cs.score()
    cur_p = base_p
    cur_out = cur
    hard_lo_x, hard_hi_x, hard_lo_y, hard_hi_y, hard_hw, hard_hh = hard_aabb
    for m in hard_ord:
        if time.time() > deadline:
            break
        mi = int(m)
        ox = float(cur_out[mi, 0])
        oy = float(cur_out[mi, 1])
        hw = float(hard_hw[mi])
        hh = float(hard_hh[mi])
        best_dx, best_dy, best_p = 0.0, 0.0, None
        for ddx, ddy in moves:
            nx, ny = ox + ddx, oy + ddy
            if nx - hw < 0 or nx + hw > cw:
                continue
            if ny - hh < 0 or ny + hh > ch:
                continue
            # Inflate AABB by _LEGALIZE_EPS so accepted moves survive legalize.c.
            collide = ((nx + hw > hard_lo_x - _LEGALIZE_EPS)
                       & (nx - hw < hard_hi_x + _LEGALIZE_EPS)
                       & (ny + hh > hard_lo_y - _LEGALIZE_EPS)
                       & (ny - hh < hard_hi_y + _LEGALIZE_EPS))
            collide[mi] = False
            if bool(collide.any()):
                continue
            p, _, _, _ = cs.apply([mi], [nx], [ny])
            if p < cur_p - 1e-9 and (best_p is None or p < best_p):
                best_dx, best_dy, best_p = ddx, ddy, p
            cs.revert()
        if best_p is not None:
            cs.apply([mi], [ox + best_dx], [oy + best_dy])
            cs.commit()
            new_x = ox + best_dx
            new_y = oy + best_dy
            cur_out[mi, 0] = new_x
            cur_out[mi, 1] = new_y
            hard_lo_x[mi] = new_x - hw
            hard_hi_x[mi] = new_x + hw
            hard_lo_y[mi] = new_y - hh
            hard_hi_y[mi] = new_y + hh
            cur_p = best_p
    return cur_out


def _softmove_pass(cur, sizes, cw, ch, cs, soft_ord, moves, deadline):
    base_p, _, _, _ = cs.score()
    cur_p = base_p
    cur_out = cur
    accepted = 0
    for m in soft_ord:
        if time.time() > deadline:
            break
        mi = int(m)
        ox = float(cur_out[mi, 0])
        oy = float(cur_out[mi, 1])
        hw = float(sizes[mi, 0] * 0.5)
        hh = float(sizes[mi, 1] * 0.5)
        best_dx, best_dy, best_p = 0.0, 0.0, None
        for ddx, ddy in moves:
            nx, ny = ox + ddx, oy + ddy
            if nx - hw < 0 or nx + hw > cw:
                continue
            if ny - hh < 0 or ny + hh > ch:
                continue
            p, _, _, _ = cs.apply([mi], [nx], [ny])
            if p < cur_p - 1e-9 and (best_p is None or p < best_p):
                best_dx, best_dy, best_p = ddx, ddy, p
            cs.revert()
        if best_p is not None:
            cs.apply([mi], [ox + best_dx], [oy + best_dy])
            cs.commit()
            cur_out[mi, 0] = ox + best_dx
            cur_out[mi, 1] = oy + best_dy
            cur_p = best_p
            accepted += 1
    return cur_out, accepted


def _pair_swap_pass(cur, sizes, mov_i32, nh, n_macros, pn, cs, deadline):
    pin_h = pn["hard_pin_counts"]
    buckets = {}
    for i in range(nh):
        if not mov_i32[i]:
            continue
        key = (round(float(sizes[i, 0]), 6),
               round(float(sizes[i, 1]), 6))
        buckets.setdefault(key, []).append(i)
    pairs = []
    for group in buckets.values():
        if len(group) < 2:
            continue
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                i, j = group[a], group[b]
                pairs.append((int(pin_h[i]) + int(pin_h[j]), i, j))
    pairs.sort(reverse=True)
    cur_out = cur
    cur_p, _, _, _ = cs.score()
    for (_, i, j) in pairs:
        if time.time() > deadline:
            break
        xi, yi = float(cur_out[i, 0]), float(cur_out[i, 1])
        xj, yj = float(cur_out[j, 0]), float(cur_out[j, 1])
        p, _, _, _ = cs.apply([i, j], [xj, xi], [yj, yi])
        if p < cur_p - 1e-9:
            cur_out[[i, j]] = cur_out[[j, i]]
            cur_p = p
            cs.commit()
        else:
            cs.revert()
    return cur_out


def _window_reorder_pass(cur, sizes, mov_i32, nh, n_macros, pn, cs,
                           deadline, top_W=50, K=4):
    """For each of the top_W hottest soft macros, try every non-identity
    permutation of K nearest soft macros and commit any improving swap."""
    if n_macros - nh <= K:
        return cur
    soft_movable_full = [i for i in range(nh, n_macros) if mov_i32[i]]
    if len(soft_movable_full) < K:
        return cur
    soft_idx = np.asarray(soft_movable_full, dtype=np.int64)
    heats = pn["soft_pin_counts_full"][soft_idx]
    order_idx = np.argsort(-heats)
    top_soft = soft_idx[order_idx[:top_W]]
    soft_pos = cur[soft_idx]

    identity = tuple(range(K))
    non_id_perms = [p for p in permutations(range(K)) if p != identity]
    for hot_m in top_soft:
        if time.time() > deadline:
            break
        my = cur[hot_m]
        d2 = ((soft_pos[:, 0] - my[0]) ** 2
              + (soft_pos[:, 1] - my[1]) ** 2)
        nearest = np.argpartition(d2, K - 1)[:K]
        window_macros = soft_idx[nearest].tolist()
        positions = [(float(cur[m, 0]), float(cur[m, 1]))
                     for m in window_macros]
        base_proxy, _, _, _ = cs.score()
        best_proxy = base_proxy
        best_perm = None
        for perm in non_id_perms:
            new_x = [positions[perm[i]][0] for i in range(K)]
            new_y = [positions[perm[i]][1] for i in range(K)]
            p, _, _, _ = cs.apply(window_macros, new_x, new_y)
            if p < best_proxy - 1e-7:
                best_proxy = p
                best_perm = perm
            cs.revert()
        if best_perm is not None:
            new_x = [positions[best_perm[i]][0] for i in range(K)]
            new_y = [positions[best_perm[i]][1] for i in range(K)]
            cs.apply(window_macros, new_x, new_y)
            cs.commit()
            for i, m in enumerate(window_macros):
                cur[m, 0] = new_x[i]
                cur[m, 1] = new_y[i]
    return cur


def _build_orders(sizes, mov_i32, nh, n_macros, pn):
    """Hard/soft macro indices sorted hottest-first by pin count; pin counts
    come from pn (cached static tables)."""
    pin_h = pn["hard_pin_counts"]
    pin_s = pn["soft_pin_counts_full"][nh:]
    hard_movable = np.array(
        [i for i in range(nh) if mov_i32[i]], dtype=np.int64)
    hard_ord = (hard_movable[np.argsort(-pin_h[hard_movable])]
                if hard_movable.size else hard_movable)
    soft_movable = np.array(
        [i for i in range(nh, n_macros) if mov_i32[i]], dtype=np.int64)
    soft_ord = (soft_movable[np.argsort(-pin_s[soft_movable - nh])]
                if soft_movable.size else soft_movable)
    return hard_ord, soft_ord


def _aabb_cache(cur, sizes, nh, hard_hw=None, hard_hh=None):
    """AABB lo/hi for the first nh hard macros; pass cached hard_hw/hh from
    octx to skip the static halve+cast."""
    hard_x = cur[:nh, 0].copy()
    hard_y = cur[:nh, 1].copy()
    if hard_hw is None:
        hard_hw = (sizes[:nh, 0] / 2.0).astype(np.float64)
    if hard_hh is None:
        hard_hh = (sizes[:nh, 1] / 2.0).astype(np.float64)
    return (hard_x - hard_hw, hard_x + hard_hw,
            hard_y - hard_hh, hard_y + hard_hh,
            hard_hw, hard_hh)


def basin_jump_v3(
    cur, sizes, mov_i32, n_macros, nh, cw, ch, octx, pn,
    *,
    cc_call,
    gpu_state=None,          # required for basin_jump_v2 GPU chain
    pre_bj_proxy=None,
    wl_extra_weight=None,
    verbose=False,
    time_budget_s=None,
    legalize_fn=None,
):
    """Run v3 basin-jump = GPU chain (basin_jump_v2) followed by the CPU
    operator stack; pass gpu_state to skip 3-4s of GPUState build, and
    legalize_fn(pos_hard)->(legalized,overlap_bool) to row-legalize between
    passes."""
    from .basin_jump_v2 import basin_jump_v2
    from .state import GPUState
    from .init import init_state as gpu_init_state
    from abuplace.placer import CongState

    if time_budget_s is None:
        time_budget_s = _DEFAULT_BUDGET_S

    t0 = time.time()
    # Determinism: the inner passes gate on `time.time() > deadline`, which
    # makes the output depend on host clock speed. Pin the deadline to +inf so
    # the iteration caps drive termination instead.
    _DETERMINISTIC = True

    # Apples-to-apples baseline for the chain accept gate.
    pre_eval = cc_call(cur, n_iter=0, lr_bins=2.0)
    base_proxy = float(pre_eval[3])
    cong = float(pre_eval[1])

    cur = cur.copy()

    # Phase 1: GPU basin_jump_v2 chain (~24-30s on ibm03).
    own_gpu_state = False
    if gpu_state is None:
        if verbose:
            print("    [v3] no gpu_state passed; building (3-4s)")
        gpu_state = GPUState.from_pn(
            pn=pn, pin_csr=octx["pin_csr"], octx=octx,
            sizes_np=sizes, positions_np=cur.copy(),
            movable_np=mov_i32, n_macros=n_macros, n_hard=nh)
        gpu_init_state(gpu_state)
        own_gpu_state = True

    res = basin_jump_v2(
        cur, sizes, mov_i32, n_macros, nh, cw, ch, octx, pn,
        cc_call=cc_call, gpu_state=gpu_state,
        pre_bj_proxy=pre_bj_proxy,
        wl_extra_weight=wl_extra_weight,
        verbose=verbose,
    )
    chain_best = np.asarray(res["cur"], dtype=np.float64)
    chain_p = float(cc_call(chain_best, n_iter=0, lr_bins=2.0)[3])
    chain_wall = time.time() - t0
    if verbose:
        print(f"    [v3] GPU chain p={chain_p:.6f} ({chain_wall:.2f}s)")

    # Track the best via a fresh n_iter=0 cc_call only - cs.score() drifts.
    # chain_best is the fallback for when the operator stack regresses.
    best_cur = chain_best.copy()
    best_p = chain_p

    cur = chain_best.copy()
    deadline = t0 + time_budget_s - 1.5
    if _DETERMINISTIC:
        # A far-future deadline lets the iteration caps drive termination:
        # trades a wall-time guarantee for bit-determinism.
        deadline = t0 + 1e9
    gw = cw / octx["gc"]
    gh = ch / octx["gr"]

    hard_ord, soft_ord = _build_orders(sizes, mov_i32, nh, n_macros, pn)
    hard_aabb = _aabb_cache(cur, sizes, nh,
                              hard_hw=octx.get('hard_hw'),
                              hard_hh=octx.get('hard_hh'))

    # Phase 2: pass 1 full operator stack (HM ladder + pair-swap + SM ladder).
    if verbose:
        print("    [v3] pass 1 (full stack) ...")
    with CongState(cur, sizes, mov_i32, n_macros, nh, cw, ch, octx, pn,
                    wl_extra_weight=wl_extra_weight) as cs:
        for step_rel in (1.0, 0.5, 0.25, 0.125):
            if time.time() > deadline:
                break
            dx, dy = gw * step_rel, gh * step_rel
            moves = [(dx, 0), (-dx, 0), (0, dy), (0, -dy),
                     (dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy)]
            cur = _hardmove_pass(cur, sizes, cw, ch, cs, hard_ord, moves,
                                   deadline, hard_aabb)
        if time.time() < deadline:
            cur = _pair_swap_pass(cur, sizes, mov_i32, nh, n_macros, pn,
                                    cs, deadline)
        for step_rel in (0.5, 0.25, 0.125, 0.0625):
            if time.time() > deadline:
                break
            dx, dy = gw * step_rel, gh * step_rel
            moves = [(dx, 0), (-dx, 0), (0, dy), (0, -dy),
                     (dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy)]
            cur, _ = _softmove_pass(cur, sizes, cw, ch, cs, soft_ord,
                                       moves, deadline)

    # Phase 3: snap hard macros to row-legal positions.
    if legalize_fn is not None and nh > 0:
        legalized, overlap = _call_legalize(legalize_fn, cur[:nh], "bj_pass1")
        if not overlap:
            cur[:nh] = legalized

    p1_p = float(cc_call(cur, n_iter=0, lr_bins=2.0)[3])
    if p1_p < best_p:
        best_cur = cur.copy()
        best_p = p1_p
    if verbose:
        print(f"    [v3] post-pass1 fresh p={p1_p:.6f} "
              f"(best={best_p:.6f})")

    # Phase 4: fresh CongState rebuild + pass 2 full operator stack.
    if verbose:
        print("    [v3] pass 2 (full stack on rebuilt) ...")
    hard_x = cur[:nh, 0].copy()
    hard_y = cur[:nh, 1].copy()
    hard_hw = octx['hard_hw']
    hard_hh = octx['hard_hh']
    hard_aabb = (hard_x - hard_hw, hard_x + hard_hw,
                 hard_y - hard_hh, hard_y + hard_hh,
                 hard_hw, hard_hh)
    with CongState(cur, sizes, mov_i32, n_macros, nh, cw, ch, octx, pn,
                    wl_extra_weight=wl_extra_weight) as cs:
        for step_rel in (0.5, 0.25, 0.125, 0.0625):
            if time.time() > deadline:
                break
            dx, dy = gw * step_rel, gh * step_rel
            moves = [(dx, 0), (-dx, 0), (0, dy), (0, -dy),
                     (dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy)]
            cur = _hardmove_pass(cur, sizes, cw, ch, cs, hard_ord, moves,
                                   deadline, hard_aabb)
        if time.time() < deadline:
            cur = _pair_swap_pass(cur, sizes, mov_i32, nh, n_macros, pn,
                                    cs, deadline)
        for step_rel in (0.25, 0.125, 0.0625, 0.03):
            if time.time() > deadline:
                break
            dx, dy = gw * step_rel, gh * step_rel
            moves = [(dx, 0), (-dx, 0), (0, dy), (0, -dy),
                     (dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy)]
            cur, _ = _softmove_pass(cur, sizes, cw, ch, cs, soft_ord,
                                       moves, deadline)
        if time.time() < deadline - 0.5:
            cur = _window_reorder_pass(cur, sizes, mov_i32, nh, n_macros,
                                         pn, cs, deadline, top_W=50, K=4)

    # Phase 5: legalize then iterated soft-only sweeps.
    if legalize_fn is not None and nh > 0:
        legalized, overlap = _call_legalize(legalize_fn, cur[:nh], "bj_pass2")
        if not overlap:
            cur[:nh] = legalized

    p2_p = float(cc_call(cur, n_iter=0, lr_bins=2.0)[3])
    if p2_p < best_p:
        best_cur = cur.copy()
        best_p = p2_p
    if verbose:
        print(f"    [v3] post-pass2 fresh p={p2_p:.6f} "
              f"(best={best_p:.6f})")

    if time.time() < deadline - 1.0:
        if verbose:
            print("    [v3] pass 3 (soft sweep iters) ...")
        with CongState(cur, sizes, mov_i32, n_macros, nh, cw, ch,
                        octx, pn,
                        wl_extra_weight=wl_extra_weight) as cs:
            for it in range(3):
                if time.time() > deadline:
                    break
                had_any = False
                for step_rel in (0.0625, 0.03, 0.015):
                    if time.time() > deadline:
                        break
                    dx, dy = gw * step_rel, gh * step_rel
                    moves = [(dx, 0), (-dx, 0), (0, dy), (0, -dy),
                             (dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy)]
                    cur, n_sm = _softmove_pass(cur, sizes, cw, ch, cs,
                                                  soft_ord, moves, deadline)
                    if n_sm > 0:
                        had_any = True
                if not had_any:
                    break

    p3_p = float(cc_call(cur, n_iter=0, lr_bins=2.0)[3])
    if p3_p < best_p:
        best_cur = cur.copy()
        best_p = p3_p
    if verbose:
        print(f"    [v3] post-pass3 fresh p={p3_p:.6f} "
              f"(best={best_p:.6f})")

    # Phase 5b: revisit the full stack until a pass improves by < 1.5e-4, or 10
    # iterations. The cap replaces a wall-clock cutoff, for determinism.
    iter_idx = 4
    last_p = p3_p
    _revisit_max = 10
    while iter_idx - 4 < _revisit_max:
        if verbose:
            print(f"    [v3] pass {iter_idx} (full-stack revisit) ...")
        hard_x = cur[:nh, 0].copy()
        hard_y = cur[:nh, 1].copy()
        hard_aabb = (hard_x - hard_hw, hard_x + hard_hw,
                     hard_y - hard_hh, hard_y + hard_hh,
                     hard_hw, hard_hh)
        with CongState(cur, sizes, mov_i32, n_macros, nh, cw, ch,
                        octx, pn,
                        wl_extra_weight=wl_extra_weight) as cs:
            for step_rel in (0.25, 0.125, 0.0625):
                if time.time() > deadline:
                    break
                dx, dy = gw * step_rel, gh * step_rel
                moves = [(dx, 0), (-dx, 0), (0, dy), (0, -dy),
                         (dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy)]
                cur = _hardmove_pass(cur, sizes, cw, ch, cs, hard_ord,
                                       moves, deadline, hard_aabb)
            if time.time() < deadline:
                cur = _pair_swap_pass(cur, sizes, mov_i32, nh, n_macros,
                                        pn, cs, deadline)
            for step_rel in (0.0625, 0.03, 0.015):
                if time.time() > deadline:
                    break
                dx, dy = gw * step_rel, gh * step_rel
                moves = [(dx, 0), (-dx, 0), (0, dy), (0, -dy),
                         (dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy)]
                cur, _ = _softmove_pass(cur, sizes, cw, ch, cs, soft_ord,
                                           moves, deadline)
        if legalize_fn is not None and nh > 0:
            legalized, overlap = _call_legalize(legalize_fn, cur[:nh], "bj_revisit")
            if not overlap:
                cur[:nh] = legalized
        pn_p = float(cc_call(cur, n_iter=0, lr_bins=2.0)[3])
        if pn_p < best_p:
            best_cur = cur.copy()
            best_p = pn_p
        if verbose:
            print(f"    [v3] post-pass{iter_idx} fresh p={pn_p:.6f} "
                  f"(best={best_p:.6f})")
        # Stop revisiting below 1.5e-4 per pass - past that point WM-μ finds
        # roughly 10x more lift per second.
        if pn_p > last_p - 1.5e-4:
            if verbose:
                print(f"    [v3] converged (Δ={last_p - pn_p:+.6f}); "
                      f"exiting revisit loop")
            break
        last_p = pn_p
        iter_idx += 1

    # Phase 5d: WM-gradient micro-perturb chain over the saturated state - tiny
    # lr (<=0.002), brief polish, strict fresh-proxy accept, lr halved on every
    # rejection. Iteration-capped rather than wall-clocked, for determinism.
    if time.time() < time_budget_s + t0 - 2.0:
        micro_lr = 0.002
        consecutive_rejects = 0
        max_iters = 8
        for _it in range(max_iters):
            grad, _ = compute_hpwl_gradient(
                best_cur, sizes, mov_i32, n_macros, nh, pn,
                wl_extra_weight=wl_extra_weight, device="cpu")
            grad_np = (grad.cpu().numpy() if hasattr(grad, "cpu")
                       else np.asarray(grad))
            soft_grad = grad_np[nh:]
            soft_norm = np.linalg.norm(soft_grad, axis=1)
            mx = float(soft_norm.max()) if soft_norm.size else 0.0
            if mx < 1e-12:
                break
            scale = (micro_lr * min(gw, gh)) / mx
            stepped = best_cur.copy()
            stepped[nh:, 0] -= scale * soft_grad[:, 0]
            stepped[nh:, 1] -= scale * soft_grad[:, 1]
            hw_arr = sizes[:, 0] * 0.5
            hh_arr = sizes[:, 1] * 0.5
            stepped[:, 0] = np.clip(stepped[:, 0], hw_arr, cw - hw_arr)
            stepped[:, 1] = np.clip(stepped[:, 1], hh_arr, ch - hh_arr)
            r = cc_call(stepped, n_iter=2, lr_bins=0.5,
                        extra_phases=[(2, 0.15), (2, 0.05)])
            polished = r[0]
            polished_p = float(cc_call(polished, n_iter=0, lr_bins=2.0)[3])
            if polished_p < best_p - 1e-6:
                if verbose:
                    print(f"    [v3] WM-μ lr={micro_lr:.4f}: "
                          f"p={polished_p:.6f} ACCEPT")
                best_cur = polished
                best_p = polished_p
                consecutive_rejects = 0
            else:
                if verbose:
                    print(f"    [v3] WM-μ lr={micro_lr:.4f}: "
                          f"p={polished_p:.6f} . reject")
                consecutive_rejects += 1
                if consecutive_rejects >= 2:
                    break
                micro_lr *= 0.5

    # Phase 6: unconditional final polish on best_cur. WM-μ's own polish
    # targeted a perturbed state, so this closes the convergence gap (~0.001
    # typical). Always on, so the result never depends on wall-clock.
    cur = best_cur.copy()
    r = cc_call(cur, n_iter=2, lr_bins=0.15,
                extra_phases=[(2, 0.05), (2, 0.02)])
    polished_cur = r[0]
    polished_p = float(cc_call(polished_cur, n_iter=0, lr_bins=2.0)[3])
    if polished_p < best_p:
        best_cur = polished_cur
        best_p = polished_p
    if verbose:
        print(f"    [v3] final polish; fresh p={polished_p:.6f} "
              f"(best={best_p:.6f})")

    # placer stage-3 gate reads accepted_rounds as a "did v3 improve?" flag.
    accepted_rounds = 1 if best_p < base_proxy - 1e-6 else 0

    if own_gpu_state:
        # This GPUState was built here (the caller passed none, which happens
        # when the placer's async build failed), so the caller has no handle to
        # free it. Release it now: everything returned below is numpy, and the
        # best-of-N loop runs several full pipelines back-to-back in one
        # process, where a stranded state fragments the allocator into OOM.
        gpu_state = None
        torch.cuda.empty_cache()

    return {
        "cur": best_cur,
        "final_proxy": best_p,
        "base_proxy": base_proxy,
        "cong": cong,
        "accepted_rounds": accepted_rounds,
        "rounds_run": res.get("rounds_run", 0),
        "total_rounds": res.get("total_rounds", 0),
        "aborted_first": False,  # v3 always proceeds to operators even if chain rejects
    }
