"""Per-macro candidate position generators (axis/diag + smart-dir) plus AABB
hard-overlap filter; consumed by polish.py and eval_is_batched.py."""

import math


def _build_hard_aabb(state):
    """Cache hard-macro (lo, hi) AABBs of shape (nh, 2) for CPU-side overlap
    queries (the GPU version paid an .item() sync per candidate)."""
    nh = state.nh
    if nh == 0:
        return None, None
    hpos = state.pos[:nh].cpu().numpy()
    hsz = state.sizes[:nh].cpu().numpy()
    half = hsz * 0.5
    lo = hpos - half
    hi = hpos + half
    return lo.copy(), hi.copy()


def _check_overlap_with_hards(state, hard_lo, hard_hi, m, nx, ny):
    """Return True if placing macro m at (nx, ny) overlaps any hard macro;
    eps=1e-6 to match C `d_abs(...) < sepx - 1e-6`."""
    import numpy as np
    if hard_lo is None:
        return False
    sizes_np = getattr(state, '_sizes_np_cache', None)
    if sizes_np is None:
        sizes_np = state.sizes.cpu().numpy()
        state._sizes_np_cache = sizes_np
    hw = float(sizes_np[m, 0]) * 0.5
    hh = float(sizes_np[m, 1]) * 0.5
    qlo_x = nx - hw
    qhi_x = nx + hw
    qlo_y = ny - hh
    qhi_y = ny + hh
    eps = 1e-6
    overlap_x = (qhi_x - hard_lo[:, 0] > eps) & (hard_hi[:, 0] - qlo_x > eps)
    overlap_y = (qhi_y - hard_lo[:, 1] > eps) & (hard_hi[:, 1] - qlo_y > eps)
    return bool(np.any(overlap_x & overlap_y))


def _neighborhood_avg(grid_2d, mr, mc, win):
    gr, gc = grid_2d.shape
    lc = max(0, mc - win); rc = min(gc - 1, mc + win)
    dr = max(0, mr - win); ur = min(gr - 1, mr + win)
    l = grid_2d[mr, lc:mc].mean() if mc > lc else 0.0
    r = grid_2d[mr, mc + 1:rc + 1].mean() if rc > mc else 0.0
    d = grid_2d[dr:mr, mc].mean() if mr > dr else 0.0
    u = grid_2d[mr + 1:ur + 1, mc].mean() if ur > mr else 0.0
    return float(l - r), float(d - u)


def compute_dirs_fast(cpu, m, H_final_2d=None, V_final_2d=None,
                      grid_occ_2d=None):
    """Return 8 direction floats for macro m: blended (out[0,1]) plus
    hpwl/density/cong unit vectors (out[2..7]); only HPWL pull is filled
    when grids are not passed."""
    out = [0.0] * 8
    cx = float(cpu.pos[m, 0])
    cy = float(cpu.pos[m, 1])
    mc = max(0, min(cpu.gc - 1, int(math.floor(cx / cpu.gw))))
    mr = max(0, min(cpu.gr - 1, int(math.floor(cy / cpu.gh))))

    # HPWL pull: weighted centroid of OTHER pins on m's nets.
    off = int(cpu.mn_offsets[m]); end = int(cpu.mn_offsets[m + 1])
    hpwl_dx = 0.0; hpwl_dy = 0.0; tw = 0.0
    if end > off:
        net_subset = cpu.mn_net_ids[off:end]
        for ni in net_subset:
            ni = int(ni)
            d_pin = int(cpu.net_driver[ni])
            sum_x = 0.0; sum_y = 0.0; count = 0
            if int(cpu.pin_macro[d_pin]) != m:
                sum_x += float(cpu.pin_abs_x[d_pin])
                sum_y += float(cpu.pin_abs_y[d_pin])
                count += 1
            s_off = int(cpu.net_sinks_off[ni]); s_end = int(cpu.net_sinks_off[ni + 1])
            for k in range(s_off, s_end):
                p = int(cpu.net_sinks_idx[k])
                if int(cpu.pin_macro[p]) != m:
                    sum_x += float(cpu.pin_abs_x[p])
                    sum_y += float(cpu.pin_abs_y[p])
                    count += 1
            if count > 0:
                cxn = sum_x / count; cyn = sum_y / count
                w = float(cpu.net_weight[ni])
                hpwl_dx += w * (cxn - cx); hpwl_dy += w * (cyn - cy); tw += w
    if tw > 0:
        hpwl_dx /= tw; hpwl_dy /= tw

    h_mag = math.sqrt(hpwl_dx * hpwl_dx + hpwl_dy * hpwl_dy)
    if h_mag > 1e-12:
        out[2] = hpwl_dx / h_mag
        out[3] = hpwl_dy / h_mag

    # Density / congestion neighborhood gradients from passed-in 2D grids.
    if grid_occ_2d is not None:
        dens_dx, dens_dy = _neighborhood_avg(grid_occ_2d, mr, mc, 2)
        de_mag = math.sqrt(dens_dx * dens_dx + dens_dy * dens_dy)
        if de_mag > 1e-12:
            out[4] = dens_dx / de_mag
            out[5] = dens_dy / de_mag
    if H_final_2d is not None and V_final_2d is not None:
        cong_grid = H_final_2d + V_final_2d
        cong_dx, cong_dy = _neighborhood_avg(cong_grid, mr, mc, 2)
        co_mag = math.sqrt(cong_dx * cong_dx + cong_dy * cong_dy)
        if co_mag > 1e-12:
            out[6] = cong_dx / co_mag
            out[7] = cong_dy / co_mag

    # Blended dir uses proxy weights 1.0/0.5/0.5 on hpwl/density/cong unit vecs.
    ux = out[2] * 1.0 + out[4] * 0.5 + out[6] * 0.5
    uy = out[3] * 1.0 + out[5] * 0.5 + out[7] * 0.5
    fm = math.sqrt(ux * ux + uy * uy)
    if fm > 1e-12:
        out[0] = ux / fm
        out[1] = uy / fm

    return out


def _basic_candidates(state, m, step):
    """8 axis/diag (dx, dy) candidate offsets at `step` bin units, returned in
    canvas coords."""
    gw = state.gw
    gh = state.gh
    s_x = step * gw
    s_y = step * gh
    return [
        ( s_x,  0.0),
        (-s_x,  0.0),
        ( 0.0,  s_y),
        ( 0.0, -s_y),
        ( s_x,  s_y),
        (-s_x,  s_y),
        ( s_x, -s_y),
        (-s_x, -s_y),
    ]


def _smart_candidates_with_pull(state, cpu, m, step,
                                  hpwl_pull_xy,
                                  H_final_2d=None, V_final_2d=None,
                                  grid_occ_2d=None):
    """Like _smart_candidates but takes a precomputed (hp_x, hp_y) pull so we
    skip recomputing HPWL per macro; density/cong reads still happen here
    (~8 numpy reads/macro, cheap)."""
    gw = state.gw
    gh = state.gh
    s_x = step * gw
    s_y = step * gh
    cands = [
        ( s_x,  0.0), (-s_x,  0.0),
        ( 0.0,  s_y), ( 0.0, -s_y),
        ( s_x,  s_y), (-s_x,  s_y),
        ( s_x, -s_y), (-s_x, -s_y),
    ]
    cx = float(cpu.pos[m, 0]); cy = float(cpu.pos[m, 1])
    mc = max(0, min(cpu.gc - 1, int(math.floor(cx / cpu.gw))))
    mr = max(0, min(cpu.gr - 1, int(math.floor(cy / cpu.gh))))
    out = [0.0] * 8

    hp_x, hp_y = hpwl_pull_xy
    h_mag = math.sqrt(hp_x * hp_x + hp_y * hp_y)
    if h_mag > 1e-12:
        out[2] = hp_x / h_mag
        out[3] = hp_y / h_mag

    if grid_occ_2d is not None:
        dens_dx, dens_dy = _neighborhood_avg(grid_occ_2d, mr, mc, 2)
        de_mag = math.sqrt(dens_dx * dens_dx + dens_dy * dens_dy)
        if de_mag > 1e-12:
            out[4] = dens_dx / de_mag
            out[5] = dens_dy / de_mag
    if H_final_2d is not None and V_final_2d is not None:
        cong_grid = H_final_2d + V_final_2d
        cong_dx, cong_dy = _neighborhood_avg(cong_grid, mr, mc, 2)
        co_mag = math.sqrt(cong_dx * cong_dx + cong_dy * cong_dy)
        if co_mag > 1e-12:
            out[6] = cong_dx / co_mag
            out[7] = cong_dy / co_mag

    # Blended unit vec uses the same proxy weights (1.0/0.5/0.5).
    ux = out[2] * 1.0 + out[4] * 0.5 + out[6] * 0.5
    uy = out[3] * 1.0 + out[5] * 0.5 + out[7] * 0.5
    fm = math.sqrt(ux * ux + uy * uy)
    if fm > 1e-12:
        out[0] = ux / fm
        out[1] = uy / fm

    blended_mults = [1.0, 2.0, 4.0, 8.0]
    comp_mults = [1.0, 2.0]
    CMAX = 24
    for di in range(4):
        if len(cands) >= CMAX:
            break
        sx, sy = out[2 * di], out[2 * di + 1]
        if sx * sx + sy * sy <= 1e-12:
            continue
        mults = blended_mults if di == 0 else comp_mults
        for mi in mults:
            if len(cands) >= CMAX:
                break
            cands.append((step * mi * sx * gw, step * mi * sy * gh))
    return cands


def _smart_candidates(state, cpu, m, step,
                       H_final_2d=None, V_final_2d=None, grid_occ_2d=None):
    """Up to 24 candidates: 8 axis/diag + smart-dir (blended*4 mults + 3
    components*2 mults); without grid views, only HPWL pull contributes to
    direction."""
    gw = state.gw
    gh = state.gh
    s_x = step * gw
    s_y = step * gh
    cands = [
        ( s_x,  0.0), (-s_x,  0.0),
        ( 0.0,  s_y), ( 0.0, -s_y),
        ( s_x,  s_y), (-s_x,  s_y),
        ( s_x, -s_y), (-s_x, -s_y),
    ]
    dirs = compute_dirs_fast(cpu, m, H_final_2d=H_final_2d,
                             V_final_2d=V_final_2d, grid_occ_2d=grid_occ_2d)
    blended_mults = [1.0, 2.0, 4.0, 8.0]
    comp_mults = [1.0, 2.0]
    CMAX = 24
    for di in range(4):
        if len(cands) >= CMAX:
            break
        sx, sy = dirs[2 * di], dirs[2 * di + 1]
        if sx * sx + sy * sy <= 1e-12:
            continue
        mults = blended_mults if di == 0 else comp_mults
        for mi in mults:
            if len(cands) >= CMAX:
                break
            cands.append((step * mi * sx * gw, step * mi * sy * gh))
    return cands
