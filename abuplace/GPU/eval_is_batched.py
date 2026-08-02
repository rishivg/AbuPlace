"""Multi-macro batched candidate eval over one IS (net+bbox-disjoint macros):
one polish_emit call per IS using a dual-pose delta (orig vs cand) for
routing/density/HPWL, top_k-capped parallel Jacobi commit; ~28x launch
reduction vs per-macro polish."""

import os
import time

import numpy as np
import torch

from .batched_candidates import (
    batched_compute_cong, batched_compute_density,
    batched_smooth_hnet_to_hfinal, batched_smooth_vnet_to_vfinal,
)
from .candidates import (
    _basic_candidates, _check_overlap_with_hards,
    _smart_candidates, _smart_candidates_with_pull,
)


_PROFILE_TOTALS = {}
_PROFILE_ENABLED = os.environ.get("XP_POLISH_PROFILE", "0") == "1"


def _record(name, dt):
    if _PROFILE_ENABLED:
        _PROFILE_TOTALS[name] = _PROFILE_TOTALS.get(name, 0.0) + dt


def _maybe_sync():
    """GPU sync only when profiling - production keeps streams async so
    density/HPWL/smooth overlap polish_emit."""
    if _PROFILE_ENABLED:
        torch.cuda.synchronize()


def _compute_pin_crossed_mask(cpu, macro_ids, nx_list, ny_list):
    """Return per-probe bool = any of m's pins crosses a gcell at (nx, ny) vs
    cpu.pin_row/col; routing is gcell-quantized so non-crossed probes have
    delta_H=delta_V=0 and can skip the emit kernel."""
    B = len(macro_ids)
    if B == 0:
        return np.zeros(0, dtype=bool)
    macros_a = np.asarray(macro_ids, dtype=np.int64)
    nx_a = np.asarray(nx_list, dtype=np.float64)
    ny_a = np.asarray(ny_list, dtype=np.float64)
    gw = cpu.gw; gh = cpu.gh
    gc = cpu.gc; gr = cpu.gr

    mp_off = cpu.mp_offsets
    crossed = np.zeros(B, dtype=bool)
    pin_x_off = cpu.pin_x_off
    pin_y_off = cpu.pin_y_off
    pin_row = cpu.pin_row
    pin_col = cpu.pin_col

    # Vectorized fan-out: flatten (probe, pin) pairs into a single global pin
    # index list.
    starts = mp_off[macros_a]
    ends = mp_off[macros_a + 1]
    counts = ends - starts
    total = int(counts.sum())
    if total == 0:
        return crossed
    probe_idx_per_pin = np.repeat(np.arange(B, dtype=np.int64), counts)
    # global_slot = starts[probe] + slot_within_probe.
    out_offsets = np.zeros(B + 1, dtype=np.int64)
    np.cumsum(counts, out=out_offsets[1:])
    slot_within = np.arange(total, dtype=np.int64) - out_offsets[probe_idx_per_pin]
    global_slot = starts[probe_idx_per_pin] + slot_within
    pin_global_idx = cpu.mp_pin_ids[global_slot].astype(np.int64)

    pos_x_per_pin = nx_a[probe_idx_per_pin] + pin_x_off[pin_global_idx]
    pos_y_per_pin = ny_a[probe_idx_per_pin] + pin_y_off[pin_global_idx]
    new_c = np.clip(np.floor(pos_x_per_pin / gw).astype(np.int64), 0, gc - 1)
    new_r = np.clip(np.floor(pos_y_per_pin / gh).astype(np.int64), 0, gr - 1)
    old_c = pin_col[pin_global_idx]
    old_r = pin_row[pin_global_idx]
    pin_changed = (new_c != old_c) | (new_r != old_r)

    np.logical_or.at(crossed, probe_idx_per_pin, pin_changed)
    return crossed


def eval_is_batched(state, cpu, macros_in_is, step, hard_lo, hard_hi,
                     grid_views=None, hpwl_pull_cache=None,
                     precomputed_cands=None, pin_abs_gpu=None):
    """Dual-pose delta evaluation of all macros in one IS in parallel;
    grid_views (H_final_2d, V_final_2d, grid_occ_2d) is per-iter pre-pull,
    precomputed_cands skips per-macro candidate gen, returns number of
    accepted macros."""
    if not macros_in_is:
        return 0

    device = state.device
    dtype = state.dtype
    idtype = state.idtype
    ng = state.ng
    gr = state.gr; gc = state.gc; sr = state.smooth_range
    eps = 1e-12
    _t = time.perf_counter()

    # 1. Build candidates per macro and filter valid.
    if precomputed_cands is not None:
        per_macro_cands = []
        for m in macros_in_is:
            cands_m = precomputed_cands.get(m)
            if cands_m:
                cx = float(cpu.pos[m, 0])
                cy = float(cpu.pos[m, 1])
                per_macro_cands.append((m, cx, cy, cands_m))
        if not per_macro_cands:
            return 0
        M = len(per_macro_cands)
        _record('1_build_cands', time.perf_counter() - _t)
        _t = time.perf_counter()
    else:
        if grid_views is None:
            H_final_2d = state.H_final.cpu().numpy().reshape(gr, gc)
            V_final_2d = state.V_final.cpu().numpy().reshape(gr, gc)
            grid_occ_2d = state.grid_occupied.cpu().numpy().reshape(gr, gc)
        else:
            H_final_2d, V_final_2d, grid_occ_2d = grid_views

        _use_basic_cands = os.environ.get("XP_POLISH_IS_BASIC", "0") == "1"
        per_macro_cands = []
        for m in macros_in_is:
            cx = float(cpu.pos[m, 0])
            cy = float(cpu.pos[m, 1])
            hw = float(cpu.sizes[m, 0]) * 0.5
            hh = float(cpu.sizes[m, 1]) * 0.5
            if _use_basic_cands:
                raw = _basic_candidates(state, m, step)
            elif hpwl_pull_cache is not None and m in hpwl_pull_cache:
                raw = _smart_candidates_with_pull(
                    state, cpu, m, step,
                    hpwl_pull_xy=hpwl_pull_cache[m],
                    H_final_2d=H_final_2d,
                    V_final_2d=V_final_2d,
                    grid_occ_2d=grid_occ_2d)
            else:
                raw = _smart_candidates(state, cpu, m, step,
                                         H_final_2d=H_final_2d,
                                         V_final_2d=V_final_2d,
                                         grid_occ_2d=grid_occ_2d)
            valid = []
            for ci, (dx, dy) in enumerate(raw):
                nx = cx + dx; ny = cy + dy
                if nx < hw: nx = hw
                if nx > state.cw - hw: nx = state.cw - hw
                if ny < hh: ny = hh
                if ny > state.ch - hh: ny = state.ch - hh
                if abs(nx - cx) < 1e-9 and abs(ny - cy) < 1e-9:
                    continue
                if _check_overlap_with_hards(state, hard_lo, hard_hi, m, nx, ny):
                    continue
                valid.append((ci, nx, ny))
            if valid:
                per_macro_cands.append((m, cx, cy, valid))
        if not per_macro_cands:
            return 0
        M = len(per_macro_cands)
        _record('1_build_cands', time.perf_counter() - _t)
        _t = time.perf_counter()

    # 2. Flatten cands into probe lists keyed back to (m, mi, ci_idx).
    all_macro_ids = []
    all_nx = []
    all_ny = []
    macro_ranges = []
    probe_to_macro_idx = []  # which macro index in per_macro_cands each probe belongs to
    for mi, (m, cx, cy, cands) in enumerate(per_macro_cands):
        start = len(all_macro_ids)
        for (_ci, nx, ny) in cands:
            all_macro_ids.append(m)
            all_nx.append(nx)
            all_ny.append(ny)
            probe_to_macro_idx.append(mi)
        macro_ranges.append((start, len(all_macro_ids)))
    B = len(all_macro_ids)
    probe_to_macro_idx_t = torch.tensor(probe_to_macro_idx,
                                         device=device, dtype=torch.long)

    # 3. One polish_emit over M orig + B cand probes; the splits below extract
    # the orig (M, ng) and cand (B, ng) halves.
    orig_macro_ids_np = np.asarray(
        [per_macro_cands[mi][0] for mi in range(M)], dtype=np.int32)
    orig_pos_np = np.empty((M, 2), dtype=np.float64)
    for mi in range(M):
        orig_pos_np[mi, 0] = per_macro_cands[mi][1]
        orig_pos_np[mi, 1] = per_macro_cands[mi][2]
    all_macro_ids_np = np.asarray(all_macro_ids, dtype=np.int32)
    all_nx_np = np.asarray(all_nx, dtype=np.float64)
    all_ny_np = np.asarray(all_ny, dtype=np.float64)

    # Skip candidate probes whose m-pins cross no gcell: routing is
    # gcell-quantized, so their delta_H/V is exactly zero (50-70% of probes in
    # the fine phases). Density and HPWL are still computed for every probe.
    use_skip = os.environ.get("XP_POLISH_EARLY_SKIP", "1") == "1"
    if use_skip and B > 0:
        cand_active_mask = _compute_pin_crossed_mask(
            cpu, all_macro_ids_np, all_nx_np, all_ny_np)
        active_idx = np.where(cand_active_mask)[0]
        n_active = len(active_idx)
    else:
        active_idx = np.arange(B, dtype=np.int64)
        n_active = B

    if n_active < B:
        active_macro_ids_np = all_macro_ids_np[active_idx]
        active_pos_np = np.stack(
            [all_nx_np[active_idx], all_ny_np[active_idx]], axis=1)
    else:
        active_macro_ids_np = all_macro_ids_np
        active_pos_np = np.stack([all_nx_np, all_ny_np], axis=1)

    combined_macro_ids_np = np.concatenate(
        [orig_macro_ids_np, active_macro_ids_np])
    combined_pos_np = np.concatenate([orig_pos_np, active_pos_np], axis=0)

    # Coalesce the 4 coord and 2 id H2D copies into one contiguous upload each,
    # so density and HPWL can launch on the aux stream while the long
    # polish_emit kernel is still running.
    gw_t = state.gw; gh_t = state.gh
    coords_cpu = np.empty(2 * M + 2 * B, dtype=np.float64)
    coords_cpu[:M] = orig_pos_np[:, 0]
    coords_cpu[M:2 * M] = orig_pos_np[:, 1]
    coords_cpu[2 * M:2 * M + B] = all_nx_np
    coords_cpu[2 * M + B:] = all_ny_np
    coords_t = torch.from_numpy(coords_cpu).to(device, torch.float64)
    orig_x = coords_t[:M]
    orig_y = coords_t[M:2 * M]
    cand_x = coords_t[2 * M:2 * M + B]
    cand_y = coords_t[2 * M + B:]

    ids_cpu = np.empty(M + B, dtype=np.int64)
    ids_cpu[:M] = orig_macro_ids_np
    ids_cpu[M:] = all_macro_ids_np
    ids_t = torch.from_numpy(ids_cpu).to(device)
    orig_ids_long = ids_t[:M]
    cand_ids_long = ids_t[M:]
    sizes_orig = state.sizes[orig_ids_long]
    sizes_cand = state.sizes[cand_ids_long]
    hw_orig = sizes_orig[:, 0] * 0.5; hh_orig = sizes_orig[:, 1] * 0.5
    hw_cand = sizes_cand[:, 0] * 0.5; hh_cand = sizes_cand[:, 1] * 0.5
    bin_cache = getattr(state, "_bin_edge_cache", None)
    if bin_cache is None:
        cells_r = torch.arange(gr, device=device, dtype=torch.float64)
        cells_c = torch.arange(gc, device=device, dtype=torch.float64)
        state._bin_edge_cache = {
            "y_lo": cells_r * gh_t, "y_hi": cells_r * gh_t + gh_t,
            "x_lo": cells_c * gw_t, "x_hi": cells_c * gw_t + gw_t,
        }
        bin_cache = state._bin_edge_cache
    bin_y_lo = bin_cache["y_lo"]; bin_y_hi = bin_cache["y_hi"]
    bin_x_lo = bin_cache["x_lo"]; bin_x_hi = bin_cache["x_hi"]

    def _density_grid(px, py, hw, hh):
        ly = py - hh; hy = py + hh
        lx = px - hw; hx = px + hw
        oy_lo = torch.maximum(bin_y_lo[None, :], ly[:, None])
        oy_hi = torch.minimum(bin_y_hi[None, :], hy[:, None])
        oy = (oy_hi - oy_lo).clamp(min=0.0)
        ox_lo = torch.maximum(bin_x_lo[None, :], lx[:, None])
        ox_hi = torch.minimum(bin_x_hi[None, :], hx[:, None])
        ox = (ox_hi - ox_lo).clamp(min=0.0)
        return (oy[:, :, None] * ox[:, None, :]).reshape(-1, ng).to(dtype)

    from abuplace.kernels.hpwl_delta import hpwl_subset_at
    if pin_abs_gpu is not None:
        pin_abs_x_gpu, pin_abs_y_gpu = pin_abs_gpu
    else:
        pin_abs_x_gpu = torch.from_numpy(np.ascontiguousarray(
            cpu.pin_abs_x, dtype=np.float64)).to(device)
        pin_abs_y_gpu = torch.from_numpy(np.ascontiguousarray(
            cpu.pin_abs_y, dtype=np.float64)).to(device)
    combined_macro_ids_hp_np = np.concatenate(
        [orig_macro_ids_np, all_macro_ids_np])
    combined_pos_hp_np = np.concatenate(
        [orig_pos_np, np.stack([all_nx_np, all_ny_np], axis=1)], axis=0)

    # XP_POLISH_AUX_STREAM=1 runs density and HPWL on an aux stream concurrent
    # with polish_emit. Off by default: on a shared GPU the SM contention can
    # cost more than the overlap saves.
    use_aux_stream = (
        os.environ.get("XP_POLISH_AUX_STREAM", "0") == "1")
    if use_aux_stream:
        aux_stream = getattr(state, "_aux_stream", None)
        if aux_stream is None:
            aux_stream = torch.cuda.Stream(device=device)
            state._aux_stream = aux_stream
        main_stream = torch.cuda.current_stream(device)
        aux_stream.wait_stream(main_stream)
        with torch.cuda.stream(aux_stream):
            new_density_pc = _density_grid(cand_x, cand_y, hw_cand, hh_cand)
            orig_density_pm = _density_grid(orig_x, orig_y, hw_orig, hh_orig)
            delta_density = new_density_pc - orig_density_pm[probe_to_macro_idx_t]
            hpwl_combined = hpwl_subset_at(
                state, cpu, combined_macro_ids_hp_np, combined_pos_hp_np,
                pin_abs_x_gpu=pin_abs_x_gpu, pin_abs_y_gpu=pin_abs_y_gpu)
            orig_hpwl_pm = hpwl_combined[:M]
            cand_hpwl_pb = hpwl_combined[M:]
            delta_hpwl_t = cand_hpwl_pb - orig_hpwl_pm[probe_to_macro_idx_t]
    else:
        new_density_pc = _density_grid(cand_x, cand_y, hw_cand, hh_cand)
        orig_density_pm = _density_grid(orig_x, orig_y, hw_orig, hh_orig)
        delta_density = new_density_pc - orig_density_pm[probe_to_macro_idx_t]
        hpwl_combined = hpwl_subset_at(
            state, cpu, combined_macro_ids_hp_np, combined_pos_hp_np,
            pin_abs_x_gpu=pin_abs_x_gpu, pin_abs_y_gpu=pin_abs_y_gpu)
        orig_hpwl_pm = hpwl_combined[:M]
        cand_hpwl_pb = hpwl_combined[M:]
        delta_hpwl_t = cand_hpwl_pb - orig_hpwl_pm[probe_to_macro_idx_t]

    # polish_emit stays on main stream (concurrent with aux density+HPWL when
    # enabled).
    combined_H, combined_V = _polish_emit_multi(
        state, cpu, combined_macro_ids_np, combined_pos_np)
    orig_H_net = combined_H[:M]
    orig_V_net = combined_V[:M]
    active_cand_H = combined_H[M:]
    active_cand_V = combined_V[M:]

    # delta_*_net_active stays compressed - a (B, ng) zero-padded tensor is
    # never materialized. smooth/cong consume the active subset, and only
    # active accepts contribute to the commit sums.
    if n_active > 0:
        if n_active < B:
            active_idx_t = torch.from_numpy(active_idx).to(device)
            active_orig_macro_t = probe_to_macro_idx_t[active_idx_t]
        else:
            active_idx_t = None
            active_orig_macro_t = probe_to_macro_idx_t
        delta_H_net_active = active_cand_H - orig_H_net[active_orig_macro_t]
        delta_V_net_active = active_cand_V - orig_V_net[active_orig_macro_t]
    else:
        active_idx_t = None
        delta_H_net_active = torch.zeros(0, ng, device=device, dtype=dtype)
        delta_V_net_active = torch.zeros(0, ng, device=device, dtype=dtype)
    _maybe_sync()
    _record('3_polish_emit', time.perf_counter() - _t); _t = time.perf_counter()

    # Sync aux stream before smooth+topk consumes density/HPWL results.
    if use_aux_stream:
        torch.cuda.current_stream(device).wait_stream(aux_stream)
    _record('4_routing_delta', time.perf_counter() - _t); _t = time.perf_counter()
    _record('5_density_gpu', 0.0)
    _record('6_hpwl_gpu', 0.0)

    # 7. Smooth + per-cand grids; only active probes contribute.
    if n_active > 0:
        delta_H_final_active = batched_smooth_hnet_to_hfinal(
            delta_H_net_active, sr, gr, gc)
        delta_V_final_active = batched_smooth_vnet_to_vfinal(
            delta_V_net_active, sr, gr, gc)
        H_final_pc_active = state.H_final.unsqueeze(0) + delta_H_final_active
        V_final_pc_active = state.V_final.unsqueeze(0) + delta_V_final_active
        cong_pc_active = batched_compute_cong(V_final_pc_active, H_final_pc_active)
    if n_active < B:
        # Inactive probes have routing delta=0, so their cong stays at
        # baseline. Expanding on-device avoids a CPU sync.
        cong_pc = state.cong_cost.expand(B).clone()
        if n_active > 0:
            cong_pc[active_idx_t] = cong_pc_active
    else:
        cong_pc = cong_pc_active

    # Density delta is continuous (nonzero for sub-gcell moves), so compute for
    # all probes.
    grid_occ_pc = state.grid_occupied.unsqueeze(0) + delta_density
    den_pc = batched_compute_density(grid_occ_pc, state.grid_area)
    _maybe_sync()
    _record('7_smooth_topk', time.perf_counter() - _t); _t = time.perf_counter()

    # 9. Per-candidate WL and full proxy on GPU. One .cpu() pull combines
    # new_full_pc with state.full_cost, replacing a double sync.
    if state.hpwl_norm > 0.0:
        new_wl_t = (state.total_hpwl + delta_hpwl_t) / state.hpwl_norm
    else:
        new_wl_t = torch.zeros(B, device=device, dtype=dtype)
    new_full_pc = new_wl_t + 0.5 * den_pc + 0.5 * cong_pc
    combined_np = torch.cat(
        [new_full_pc, state.full_cost.reshape(1)]).cpu().numpy()
    new_full_np = combined_np[:B]
    cur_full = float(combined_np[B])

    # 10. Per-macro best-improving accept. XP_POLISH_IS_TOPK caps commits per
    # independent set, bounding the drift from committing in parallel.
    top_k = int(os.environ.get("XP_POLISH_IS_TOPK", "8"))
    macro_best = []  # list of (gain, mi, gi)
    for mi, ((m, cx, cy, cands), (start, end)) in enumerate(
            zip(per_macro_cands, macro_ranges)):
        best_gi = -1
        best_full = cur_full
        for gi in range(start, end):
            if new_full_np[gi] < best_full - eps:
                best_full = new_full_np[gi]
                best_gi = gi
        if best_gi >= 0:
            macro_best.append((best_full - cur_full, mi, best_gi))
    macro_best.sort(key=lambda x: x[0])
    accepts = [(mi, gi) for (_, mi, gi) in macro_best[:top_k]]
    _record('8_accept_logic', time.perf_counter() - _t); _t = time.perf_counter()

    # 11. Commit accepted macros - IS-disjoint so safe to apply in parallel.
    if accepts:
        accept_idxs_np = np.array([gi for (_, gi) in accepts], dtype=np.int64)
        accept_idxs = torch.from_numpy(accept_idxs_np).to(device)
        state.grid_occupied = state.grid_occupied + delta_density[accept_idxs].sum(dim=0)
        # Routing delta exists only for active probes; map the accepted gi
        # through a CPU inverse table rather than paying a GPU->CPU sync.
        if n_active > 0:
            if active_idx_t is None:
                # n_active == B: every accept is active.
                idx_in_active = accept_idxs
            else:
                # Build (B,) -> idx_in_active or -1 inverse table.
                pos_np = np.full(B, -1, dtype=np.int64)
                pos_np[active_idx] = np.arange(n_active, dtype=np.int64)
                accept_active_np = pos_np[accept_idxs_np]
                valid_np = accept_active_np >= 0
                if valid_np.any():
                    idx_in_active = torch.from_numpy(
                        accept_active_np[valid_np]).to(device)
                else:
                    idx_in_active = None
            if idx_in_active is not None:
                state.H_net = state.H_net + delta_H_net_active[idx_in_active].sum(dim=0)
                state.V_net = state.V_net + delta_V_net_active[idx_in_active].sum(dim=0)
                state.H_final = state.H_final + delta_H_final_active[idx_in_active].sum(dim=0)
                state.V_final = state.V_final + delta_V_final_active[idx_in_active].sum(dim=0)

        # Per macro: update pos and refresh the CPU pin cache, coalescing all
        # per-pin GPU updates into a single upload.
        all_pin_ids = []
        new_pos_x = []
        new_pos_y = []
        new_pos_idx = []
        for (mi, gi) in accepts:
            m, cx, cy, cands = per_macro_cands[mi]
            ci_in_macro = gi - macro_ranges[mi][0]
            _, nx_a, ny_a = cands[ci_in_macro]
            new_pos_idx.append(m)
            new_pos_x.append(nx_a)
            new_pos_y.append(ny_a)
            cpu.pos[m, 0] = nx_a; cpu.pos[m, 1] = ny_a
            off_p = cpu.mp_offsets[m]; end_p = cpu.mp_offsets[m + 1]
            if off_p < end_p:
                pin_ids = cpu.mp_pin_ids[off_p:end_p]
                cpu.pin_abs_x[pin_ids] = nx_a + cpu.pin_x_off[pin_ids]
                cpu.pin_abs_y[pin_ids] = ny_a + cpu.pin_y_off[pin_ids]
                cpu.pin_col[pin_ids] = np.clip(
                    np.floor(cpu.pin_abs_x[pin_ids] / cpu.gw).astype(np.int64),
                    0, cpu.gc - 1)
                cpu.pin_row[pin_ids] = np.clip(
                    np.floor(cpu.pin_abs_y[pin_ids] / cpu.gh).astype(np.int64),
                    0, cpu.gr - 1)
                all_pin_ids.append(pin_ids)
        if new_pos_idx:
            pos_idx_t = torch.tensor(new_pos_idx, device=device, dtype=torch.long)
            pos_xy = torch.tensor(list(zip(new_pos_x, new_pos_y)),
                                    device=device, dtype=dtype)
            state.pos[pos_idx_t] = pos_xy
        if all_pin_ids:
            all_pin_ids_np = np.concatenate(all_pin_ids).astype(np.int64)
            all_pin_ids_t = torch.from_numpy(all_pin_ids_np).to(device)
            new_rows = torch.from_numpy(
                cpu.pin_row[all_pin_ids_np]).to(device, idtype)
            new_cols = torch.from_numpy(
                cpu.pin_col[all_pin_ids_np]).to(device, idtype)
            state.pin_row[all_pin_ids_t] = new_rows
            state.pin_col[all_pin_ids_t] = new_cols
            if pin_abs_gpu is not None:
                pax, pay = pin_abs_gpu
                new_abs_x = torch.from_numpy(
                    cpu.pin_abs_x[all_pin_ids_np]).to(device, torch.float64)
                new_abs_y = torch.from_numpy(
                    cpu.pin_abs_y[all_pin_ids_np]).to(device, torch.float64)
                pax[all_pin_ids_t] = new_abs_x
                pay[all_pin_ids_t] = new_abs_y

        # state.total_hpwl stays a tensor - sum accepted HPWL deltas on GPU, no
        # scalar pull.
        accept_gi_t = torch.tensor(
            [gi for (_, gi) in accepts], device=device, dtype=torch.long)
        state.total_hpwl = state.total_hpwl + delta_hpwl_t[accept_gi_t].sum()

        from .forward import compute_cong, compute_density
        state.cong_cost = compute_cong(state)
        state.density_cost = compute_density(state)
        if state.hpwl_norm > 0.0:
            state.wl_cost = state.total_hpwl / state.hpwl_norm
        else:
            state.wl_cost = torch.tensor(0.0, device=device, dtype=dtype)
        state.full_cost = state.wl_cost + 0.5 * state.density_cost + 0.5 * state.cong_cost
        # With precomputed_cands the caller already manages per-iteration
        # grid_views, so skip the per-IS sync.
        if precomputed_cands is None:
            cpu.sync_grids_from_gpu(state)
    _record('9_commit', time.perf_counter() - _t)
    return len(accepts)


def _polish_emit_multi(state, cpu, macro_ids, cand_positions, *,
                        MAX_CELLS=None, MAX_NETS_PER_MACRO=None,
                        MAX_PINS_PER_NET=None, EMIT_BLK=32, SLOTS=None):
    """Multi-macro polish_emit using the kernel's per-probe macro_ids; kernel
    bounds autotuned from state topology and cached on state when not
    overridden."""
    if (MAX_CELLS is None or MAX_NETS_PER_MACRO is None
        or MAX_PINS_PER_NET is None or SLOTS is None):
        from .polish_kernel_tune import autotune_kernel_bounds
        bnd = autotune_kernel_bounds(state)
        MAX_CELLS = MAX_CELLS or bnd["MAX_CELLS"]
        MAX_NETS_PER_MACRO = MAX_NETS_PER_MACRO or bnd["MAX_NETS_PER_MACRO"]
        MAX_PINS_PER_NET = MAX_PINS_PER_NET or bnd["MAX_PINS_PER_NET"]
        SLOTS = SLOTS or bnd["SLOTS"]
    from abuplace.kernels.polish_emit import _polish_emit_kernel

    device = state.device
    dtype = state.dtype
    B = len(macro_ids)
    if B == 0:
        return (torch.zeros(0, state.ng, device=device, dtype=dtype),
                torch.zeros(0, state.ng, device=device, dtype=dtype))

    _t = time.perf_counter()
    # Coerce inputs to contiguous numpy then to GPU.
    if isinstance(macro_ids, np.ndarray):
        macro_ids_np = macro_ids if macro_ids.dtype == np.int32 \
            else macro_ids.astype(np.int32)
    else:
        macro_ids_np = np.asarray(macro_ids, dtype=np.int32)
    if isinstance(cand_positions, np.ndarray):
        pos_np = cand_positions if cand_positions.dtype == np.float64 \
            else cand_positions.astype(np.float64)
        if pos_np.ndim == 2:
            new_x_np = np.ascontiguousarray(pos_np[:, 0])
            new_y_np = np.ascontiguousarray(pos_np[:, 1])
        else:
            new_x_np = pos_np
            new_y_np = pos_np  # defensive: caller should pass (B, 2)
    elif isinstance(cand_positions, list):
        pos_np = np.asarray(cand_positions, dtype=np.float64)
        new_x_np = np.ascontiguousarray(pos_np[:, 0])
        new_y_np = np.ascontiguousarray(pos_np[:, 1])
    else:
        new_x_np = cand_positions[0]
        new_y_np = cand_positions[1]
    macro_ids_t = torch.from_numpy(macro_ids_np).to(device)
    new_x_t = torch.from_numpy(new_x_np).to(device)
    new_y_t = torch.from_numpy(new_y_np).to(device)
    _record('emit_a_tensors', time.perf_counter() - _t); _t = time.perf_counter()

    # Scratch buffers cached on state, resized only when B grows.
    cache = getattr(state, "_emit_buf_cache", None)
    needed = B * MAX_CELLS
    if cache is None or cache["sz"] < needed:
        H_idx = torch.zeros(needed, dtype=torch.int32, device=device)
        H_val = torch.zeros(needed, dtype=torch.float32, device=device)
        V_idx = torch.zeros(needed, dtype=torch.int32, device=device)
        V_val = torch.zeros(needed, dtype=torch.float32, device=device)
        state._emit_buf_cache = {
            "sz": needed,
            "H_idx": H_idx, "H_val": H_val, "V_idx": V_idx, "V_val": V_val,
        }
    else:
        H_idx = cache["H_idx"][:needed]; H_idx.zero_()
        H_val = cache["H_val"][:needed]; H_val.zero_()
        V_idx = cache["V_idx"][:needed]; V_idx.zero_()
        V_val = cache["V_val"][:needed]; V_val.zero_()
    _record('emit_b_alloc', time.perf_counter() - _t); _t = time.perf_counter()

    grid = (B,)
    _polish_emit_kernel[grid](
        macro_ids_t, new_x_t, new_y_t,
        state.mn_offsets, state.mn_net_ids,
        state.net_driver, state.net_sinks_off, state.net_sinks_idx,
        state.net_weight,
        state.pin_macro, state.pin_x_off, state.pin_y_off,
        state.pin_row, state.pin_col,
        H_idx, H_val, V_idx, V_val,
        state.gw, state.gh, state.gc, state.gr,
        MAX_CELLS=MAX_CELLS,
        MAX_NETS_PER_MACRO=MAX_NETS_PER_MACRO,
        MAX_PINS_PER_NET=MAX_PINS_PER_NET,
        EMIT_BLK=EMIT_BLK,
        SLOTS=SLOTS,
    )
    _maybe_sync()
    _record('emit_c_kernel', time.perf_counter() - _t); _t = time.perf_counter()

    ng = state.ng
    # Power-of-2 grown (max_B, ng) scatter buffers cached on state; slice and
    # zero per IS rather than allocating fresh.
    sc_cache = getattr(state, "_emit_scatter_cache", None)
    if sc_cache is None or sc_cache["max_B"] < B or sc_cache["ng"] != ng:
        max_B = max(B, (sc_cache["max_B"] if sc_cache else 0))
        max_B = max(max_B, B)
        max_B = 1 << (max(1, max_B - 1).bit_length())
        delta_H_buf = torch.zeros(max_B, ng, device=device, dtype=dtype)
        delta_V_buf = torch.zeros(max_B, ng, device=device, dtype=dtype)
        state._emit_scatter_cache = {
            "max_B": max_B, "ng": ng,
            "H": delta_H_buf, "V": delta_V_buf,
        }
        sc_cache = state._emit_scatter_cache
    delta_H_net = sc_cache["H"][:B]
    delta_V_net = sc_cache["V"][:B]
    delta_H_net.zero_()
    delta_V_net.zero_()

    # net_weight and grid_h/v_routes are constant post-init - cache the
    # normalized weights once.
    if getattr(state, "_net_weight_h_norm_cache", None) is None:
        state._net_weight_h_norm_cache = (
            state.net_weight.to(dtype) / float(state.grid_h_routes))
        state._net_weight_v_norm_cache = (
            state.net_weight.to(dtype) / float(state.grid_v_routes))
    h_net_w_norm = state._net_weight_h_norm_cache
    v_net_w_norm = state._net_weight_v_norm_cache

    # The fused decode_scatter kernel replaces a 5-op PyTorch chain per axis:
    # decode the (net_idx+1)*sign marker and atomic_add into delta_H/V_net.
    from abuplace.kernels.decode_scatter import decode_scatter
    decode_scatter(H_idx, H_val, h_net_w_norm, delta_H_net, MAX_CELLS)
    decode_scatter(V_idx, V_val, v_net_w_norm, delta_V_net, MAX_CELLS)
    _maybe_sync()
    _record('emit_d_scatter', time.perf_counter() - _t)
    return delta_H_net, delta_V_net
