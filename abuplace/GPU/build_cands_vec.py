import os
"""Vectorized per-iter candidate builder; computes all macros' candidate
offsets in numpy at once, batch-validates against canvas bounds and
hard-macro AABBs, returns dict m -> list[(ci, nx, ny)] (replaces a
37%-of-polish-wall Python loop in eval_is_batched.build_cands)."""

import numpy as np

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @numba.njit(parallel=True, cache=True, fastmath=True)
    def _hard_overlap_numba(qlo_x, qhi_x, qlo_y, qhi_y, hl_x, hh_x, hl_y, hh_y, eps):
        """Per-cand overlap check vs all hards with early-break, parallel over
        candidates (~10x faster than numpy chunked)."""
        flat_n = qlo_x.shape[0]
        nh = hl_x.shape[0]
        out = np.zeros(flat_n, dtype=np.bool_)
        for i in numba.prange(flat_n):
            qlx = qlo_x[i]; qhx = qhi_x[i]
            qly = qlo_y[i]; qhy = qhi_y[i]
            hit = False
            for h in range(nh):
                if (qhx - hl_x[h] > eps and hh_x[h] - qlx > eps
                    and qhy - hl_y[h] > eps and hh_y[h] - qly > eps):
                    hit = True
                    break
            out[i] = hit
        return out


def _neighborhood_grad_vec(grid_2d, mr_arr, mc_arr, win=2):
    """Vectorized 4-strip (left/right/down/up) neighborhood gradient via
    row+col cumsum integral images so each macro's mean is an O(1) diff
    lookup."""
    gr, gc = grid_2d.shape
    cumsum_r = np.zeros((gr, gc + 1), dtype=np.float64)
    cumsum_r[:, 1:] = grid_2d.cumsum(axis=1)
    cumsum_c = np.zeros((gr + 1, gc), dtype=np.float64)
    cumsum_c[1:, :] = grid_2d.cumsum(axis=0)

    lc = np.maximum(mc_arr - win, 0)
    rc = np.minimum(mc_arr + win, gc - 1)
    dr = np.maximum(mr_arr - win, 0)
    ur = np.minimum(mr_arr + win, gr - 1)

    # Left strip mean over [lc, mc).
    left_count = mc_arr - lc                                 # (n,)
    left_sum = cumsum_r[mr_arr, mc_arr] - cumsum_r[mr_arr, lc]
    l_ = np.where(left_count > 0,
                   left_sum / np.maximum(left_count, 1),
                   0.0)
    # Right strip mean over [mc+1, rc+1).
    right_count = rc - mc_arr
    right_sum = cumsum_r[mr_arr, rc + 1] - cumsum_r[mr_arr, mc_arr + 1]
    r_ = np.where(right_count > 0,
                   right_sum / np.maximum(right_count, 1),
                   0.0)
    dx = l_ - r_

    # Down strip (rows below) mean over [dr, mr).
    down_count = mr_arr - dr
    down_sum = cumsum_c[mr_arr, mc_arr] - cumsum_c[dr, mc_arr]
    d_ = np.where(down_count > 0,
                   down_sum / np.maximum(down_count, 1),
                   0.0)
    # Up strip (rows above) mean over [mr+1, ur+1).
    up_count = ur - mr_arr
    up_sum = cumsum_c[ur + 1, mc_arr] - cumsum_c[mr_arr + 1, mc_arr]
    u_ = np.where(up_count > 0,
                   up_sum / np.maximum(up_count, 1),
                   0.0)
    dy = d_ - u_
    return dx, dy


def _hard_overlap_check_batch(macros_arr, nx_arr, ny_arr, hw_arr, hh_arr,
                                hard_lo, hard_hi):
    """Batched (M, K) overlap check against all hards; uses numba kernel when
    available, else chunked numpy; returns (M, K) bool."""
    if hard_lo is None or hard_hi is None:
        return np.zeros_like(nx_arr, dtype=bool)

    M, K = nx_arr.shape
    eps = 1e-6
    flat_n = M * K
    nx = nx_arr.reshape(-1)
    ny = ny_arr.reshape(-1)
    hw = hw_arr.reshape(-1)
    hh = hh_arr.reshape(-1)
    qlo_x = nx - hw; qhi_x = nx + hw
    qlo_y = ny - hh; qhi_y = ny + hh

    nh = hard_lo.shape[0]
    if nh == 0:
        return np.zeros(flat_n, dtype=bool).reshape(M, K)

    if _HAS_NUMBA:
        hl_x = np.ascontiguousarray(hard_lo[:, 0])
        hh_x = np.ascontiguousarray(hard_hi[:, 0])
        hl_y = np.ascontiguousarray(hard_lo[:, 1])
        hh_y = np.ascontiguousarray(hard_hi[:, 1])
        out = _hard_overlap_numba(
            np.ascontiguousarray(qlo_x), np.ascontiguousarray(qhi_x),
            np.ascontiguousarray(qlo_y), np.ascontiguousarray(qhi_y),
            hl_x, hh_x, hl_y, hh_y, eps)
        return out.reshape(M, K)

    # Numpy fallback: chunked broadcast keeps peak memory bounded.
    out = np.zeros(flat_n, dtype=bool)
    CHUNK = 4096
    for start in range(0, flat_n, CHUNK):
        end = min(start + CHUNK, flat_n)
        ox = (qhi_x[start:end, None] - hard_lo[None, :, 0] > eps) & (
              hard_hi[None, :, 0] - qlo_x[start:end, None] > eps)
        oy = (qhi_y[start:end, None] - hard_lo[None, :, 1] > eps) & (
              hard_hi[None, :, 1] - qlo_y[start:end, None] > eps)
        out[start:end] = (ox & oy).any(axis=1)
    return out.reshape(M, K)


def build_iter_cands_vec(state, cpu, active_macros, step,
                          hard_lo, hard_hi,
                          hpwl_pull_arr,
                          H_final_2d, V_final_2d, grid_occ_2d):
    """Vectorized candidate build for active_macros (post cold-mask);
    hpwl_pull_arr (n,2) is un-normalized in the same order; returns dict
    m->list[(ci,nx,ny)] omitting macros with zero valid cands."""
    n = len(active_macros)
    if n == 0:
        return {}
    sizes_np = getattr(state, '_sizes_np_cache', None)
    if sizes_np is None:
        sizes_np = state.sizes.cpu().numpy()
        state._sizes_np_cache = sizes_np

    macros_a = np.asarray(active_macros, dtype=np.int64)
    pos = cpu.pos[macros_a]                   # (n, 2)
    cx_arr = pos[:, 0].astype(np.float64)
    cy_arr = pos[:, 1].astype(np.float64)
    sz = sizes_np[macros_a]                    # (n, 2)
    hw_arr = sz[:, 0].astype(np.float64) * 0.5
    hh_arr = sz[:, 1].astype(np.float64) * 0.5

    gw = state.gw; gh = state.gh
    s_x = step * gw; s_y = step * gh

    # 8 axis/diag offsets shared across all macros.
    base_dx = np.array([s_x, -s_x, 0.0, 0.0, s_x, -s_x, s_x, -s_x],
                        dtype=np.float64)
    base_dy = np.array([0.0, 0.0, s_y, -s_y, s_y, s_y, -s_y, -s_y],
                        dtype=np.float64)

    # Smart directions: HPWL pull comes in pre-computed.
    hp_x = hpwl_pull_arr[:, 0].astype(np.float64)
    hp_y = hpwl_pull_arr[:, 1].astype(np.float64)
    h_mag = np.sqrt(hp_x * hp_x + hp_y * hp_y)
    hp_ux = np.where(h_mag > 1e-12, hp_x / np.maximum(h_mag, 1e-30), 0.0)
    hp_uy = np.where(h_mag > 1e-12, hp_y / np.maximum(h_mag, 1e-30), 0.0)

    # Density + cong neighborhood reads via the 4-strip cumsum helper.
    mc_arr = np.clip(np.floor(cx_arr / cpu.gw).astype(np.int64),
                      0, cpu.gc - 1)
    mr_arr = np.clip(np.floor(cy_arr / cpu.gh).astype(np.int64),
                      0, cpu.gr - 1)
    dens_dx, dens_dy = _neighborhood_grad_vec(grid_occ_2d, mr_arr, mc_arr, 2)
    de_mag = np.sqrt(dens_dx * dens_dx + dens_dy * dens_dy)
    de_ux = np.where(de_mag > 1e-12, dens_dx / np.maximum(de_mag, 1e-30), 0.0)
    de_uy = np.where(de_mag > 1e-12, dens_dy / np.maximum(de_mag, 1e-30), 0.0)

    cong_grid = H_final_2d + V_final_2d
    cong_dx, cong_dy = _neighborhood_grad_vec(cong_grid, mr_arr, mc_arr, 2)
    co_mag = np.sqrt(cong_dx * cong_dx + cong_dy * cong_dy)
    co_ux = np.where(co_mag > 1e-12, cong_dx / np.maximum(co_mag, 1e-30), 0.0)
    co_uy = np.where(co_mag > 1e-12, cong_dy / np.maximum(co_mag, 1e-30), 0.0)

    # Blended unit vec: 1.0*hpwl + 0.5*density + 0.5*cong, then re-normalize.
    bl_x = hp_ux * 1.0 + de_ux * 0.5 + co_ux * 0.5
    bl_y = hp_uy * 1.0 + de_uy * 0.5 + co_uy * 0.5
    bl_mag = np.sqrt(bl_x * bl_x + bl_y * bl_y)
    bl_ux = np.where(bl_mag > 1e-12, bl_x / np.maximum(bl_mag, 1e-30), 0.0)
    bl_uy = np.where(bl_mag > 1e-12, bl_y / np.maximum(bl_mag, 1e-30), 0.0)

    # 4 direction vectors: [blended, hpwl, dens, cong].
    dirs_x = np.stack([bl_ux, hp_ux, de_ux, co_ux], axis=0)  # (4, n)
    dirs_y = np.stack([bl_uy, hp_uy, de_uy, co_uy], axis=0)  # (4, n)

    # Smart-cand step multipliers; XP_POLISH_EXPANDED_CANDS={1,2} densifies for
    # more basin coverage.
    _expand = os.environ.get("XP_POLISH_EXPANDED_CANDS", "0")
    if _expand == "2":
        blended_mults = np.array(
            [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0],
            dtype=np.float64)
        comp_mults = np.array(
            [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0], dtype=np.float64)
    elif _expand == "1":
        blended_mults = np.array(
            [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0], dtype=np.float64)
        comp_mults = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    else:
        blended_mults = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
        comp_mults = np.array([1.0, 2.0], dtype=np.float64)
    # blended uses blended_mults, three component dirs use comp_mults; total
    # stays under CMAX=24.
    smart_offsets = []
    for di in range(4):
        mults = blended_mults if di == 0 else comp_mults
        for mi in mults:
            sx = dirs_x[di]; sy = dirs_y[di]
            cand_dx = step * mi * sx * gw         # (n,)
            cand_dy = step * mi * sy * gh         # (n,)
            valid_dir = (sx * sx + sy * sy) > 1e-12
            cand_dx = np.where(valid_dir, cand_dx, 0.0)
            cand_dy = np.where(valid_dir, cand_dy, 0.0)
            smart_offsets.append((cand_dx, cand_dy, valid_dir))

    K_basic = 8
    K_smart = len(smart_offsets)
    K = K_basic + K_smart
    dx_all = np.zeros((n, K), dtype=np.float64)
    dy_all = np.zeros((n, K), dtype=np.float64)
    valid_all = np.ones((n, K), dtype=bool)
    dx_all[:, :K_basic] = base_dx[None, :]
    dy_all[:, :K_basic] = base_dy[None, :]
    for k, (cdx, cdy, vd) in enumerate(smart_offsets):
        dx_all[:, K_basic + k] = cdx
        dy_all[:, K_basic + k] = cdy
        valid_all[:, K_basic + k] = vd

    nx_all = cx_arr[:, None] + dx_all
    ny_all = cy_arr[:, None] + dy_all

    cw = state.cw; ch = state.ch
    hw_b = hw_arr[:, None]
    hh_b = hh_arr[:, None]
    nx_all = np.clip(nx_all, hw_b, cw - hw_b)
    ny_all = np.clip(ny_all, hh_b, ch - hh_b)

    # Drop candidates whose clamp collapsed back to current position.
    no_move = (np.abs(nx_all - cx_arr[:, None]) < 1e-9) & (
              np.abs(ny_all - cy_arr[:, None]) < 1e-9)
    valid_all &= ~no_move

    macros_b = np.broadcast_to(macros_a[:, None], (n, K))
    hw_b_full = np.broadcast_to(hw_arr[:, None], (n, K))
    hh_b_full = np.broadcast_to(hh_arr[:, None], (n, K))
    overlap = _hard_overlap_check_batch(
        macros_b, nx_all, ny_all, hw_b_full, hh_b_full, hard_lo, hard_hi)
    valid_all &= ~overlap

    # Group valid (i, k) pairs into the output dict via np.diff boundaries.
    valid_i, valid_k = np.where(valid_all)
    if len(valid_i) == 0:
        return {}
    nx_v = nx_all[valid_i, valid_k]
    ny_v = ny_all[valid_i, valid_k]
    # np.where returns valid_i sorted, so boundaries split groups directly.
    boundaries = np.where(np.diff(valid_i) > 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(valid_i)]])
    macros_for_groups = valid_i[starts]
    out = {}
    valid_k_l = valid_k.tolist()
    nx_l = nx_v.tolist()
    ny_l = ny_v.tolist()
    macros_l = macros_a.tolist()
    starts_l = starts.tolist()
    ends_l = ends.tolist()
    macros_for_groups_l = macros_for_groups.tolist()
    for gi, i in enumerate(macros_for_groups_l):
        s = starts_l[gi]; e = ends_l[gi]
        out[macros_l[i]] = list(zip(valid_k_l[s:e], nx_l[s:e], ny_l[s:e]))
    return out
