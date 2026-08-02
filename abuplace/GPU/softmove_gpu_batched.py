"""GPU-batched softmove: score every (macro, candidate) probe via
BatchedProbeContext, pick the per-macro best with delta<0, then sequentially
verify-or-revert via the CongState to catch stale-base basin shift;
multi-pass re-batches after each commit pass."""

import time
import numpy as np
import torch

from .gpu_batched_probes import BatchedProbeContext
from .init import init_state


def gpu_softmove_pass(
    state, cs, cur, sizes, mov_i32, n_macros, nh, cw, ch, octx, *,
    step_rel=0.25, n_passes=2, budget_s=3.0, accept_tol=0.0,
    verbose=False,
):
    """GPU-batched softmove; mutates cur in place and returns (cur, stats); cs
    must already be initialized at cur, accept_tol=0.0 means strict
    improvement, budget_s is a hard wall budget."""
    t0 = time.time()
    if n_macros - nh <= 0:
        return cur, {"passes": 0, "probes": 0, "accepts": 0, "wall": 0.0}

    gw = cw / octx['gc']
    gh = ch / octx['gr']
    dx = gw * step_rel
    dy = gh * step_rel
    moves = np.array([
        (dx, 0), (-dx, 0), (0, dy), (0, -dy),
        (dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy),
        (2 * dx, dy), (-2 * dx, dy), (2 * dx, -dy), (-2 * dx, -dy),
        (dx, 2 * dy), (-dx, 2 * dy), (dx, -2 * dy), (-dx, -2 * dy),
    ], dtype=np.float64)

    soft_movable = np.array(
        [i for i in range(nh, n_macros) if mov_i32[i]], dtype=np.int64)
    if soft_movable.size == 0:
        return cur, {"passes": 0, "probes": 0, "accepts": 0, "wall": 0.0}

    hard_x = cur[:nh, 0].copy()
    hard_y = cur[:nh, 1].copy()
    hard_hw = (sizes[:nh, 0] / 2.0).astype(np.float64)
    hard_hh = (sizes[:nh, 1] / 2.0).astype(np.float64)

    def _overlaps_hard(nx, ny, hw, hh):
        x_o = (nx + hw > hard_x - hard_hw) & (nx - hw < hard_x + hard_hw)
        y_o = (ny + hh > hard_y - hard_hh) & (ny - hh < hard_y + hard_hh)
        return bool((x_o & y_o).any())

    ctx = BatchedProbeContext(state)

    total_probes = 0
    total_accepts = 0
    passes_done = 0
    base_proxy, _, _, _ = cs.score()
    cur_proxy = base_proxy

    for pass_i in range(n_passes):
        if time.time() - t0 > budget_s:
            break

        # Soft-macro x-move probe batch, filtered by canvas bounds and hard
        # overlap. probe_owner carries (m, k) so the results can be regrouped
        # afterwards.
        probe_macros = []
        probe_pos = []
        probe_owner = []
        for m in soft_movable:
            ox = float(cur[m, 0])
            oy = float(cur[m, 1])
            hw = float(sizes[m, 0]) / 2.0
            hh = float(sizes[m, 1]) / 2.0
            for k, (ddx, ddy) in enumerate(moves):
                nx = ox + ddx
                ny = oy + ddy
                if nx - hw < 0 or nx + hw > cw:
                    continue
                if ny - hh < 0 or ny + hh > ch:
                    continue
                if _overlaps_hard(nx, ny, hw, hh):
                    continue
                probe_macros.append(int(m))
                probe_pos.append((nx, ny))
                probe_owner.append((int(m), k))

        if not probe_macros:
            break

        probe_macros_np = np.array(probe_macros, dtype=np.int32)
        probe_pos_np = np.array(probe_pos, dtype=np.float64)
        B = len(probe_macros_np)
        total_probes += B

        t_gpu = time.time()
        # Prior pass may have committed; refresh ctx so pin_abs etc. are fresh.
        ctx.refresh()
        deltas_gpu = ctx.evaluate(probe_macros_np, probe_pos_np)
        gpu_wall = time.time() - t_gpu

        best_per_macro = {}  # m -> (delta, nx, ny)
        for i, (m, _k) in enumerate(probe_owner):
            d = float(deltas_gpu[i])
            if d < accept_tol:
                if m not in best_per_macro or d < best_per_macro[m][0]:
                    best_per_macro[m] = (d, probe_pos_np[i, 0], probe_pos_np[i, 1])

        # Most-negative GPU delta first to minimize basin-shift conflicts as
        # commits compound.
        ordered = sorted(best_per_macro.items(), key=lambda kv: kv[1][0])
        accepts_this_pass = 0
        for m, (gpu_delta, nx, ny) in ordered:
            if time.time() - t0 > budget_s:
                break
            new_proxy, _, _, _ = cs.apply([int(m)], [float(nx)], [float(ny)])
            verified_delta = new_proxy - cur_proxy
            if verified_delta < accept_tol:
                cs.commit()
                cur[m, 0] = nx
                cur[m, 1] = ny
                cur_proxy = new_proxy
                accepts_this_pass += 1
            else:
                cs.revert()

        passes_done = pass_i + 1
        total_accepts += accepts_this_pass
        if verbose:
            print(f"    [softmove gpu pass {pass_i+1}] "
                  f"probes={B} gpu_wall={gpu_wall*1000:.0f}ms "
                  f"gpu_picks={len(best_per_macro)} accepts={accepts_this_pass}")
        if accepts_this_pass == 0:
            break
        cs.rebuild()
        # Sync GPUState to post-commit positions so next pass scores fresh.
        state.pos.copy_(torch.from_numpy(np.ascontiguousarray(
            cur, dtype=np.float64)).to(state.device, state.dtype))
        init_state(state)

    return cur, {
        "passes": passes_done,
        "probes": total_probes,
        "accepts": total_accepts,
        "wall": time.time() - t0,
        "final_proxy": cur_proxy,
    }
