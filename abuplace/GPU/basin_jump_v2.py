"""Basin-jump V2: cong-adaptive perturb-polish chain (perturb -> GPU polish ->
optional C-short refine -> bit-exact C n_iter=0 accept gate, abort on first
reject); three bands chosen by post-polish cong, with high-cong band
hard-capped at 2 rounds for the 60s budget."""

import os
import time

import numpy as np
import torch

from .basin_jump import compute_hpwl_gradient as _compute_hpwl_gradient
from .init import init_state as _gpu_init_state


# (init_n_iter, extra_phases) presets, pushed into
# polish_hybrid_5perturb._INIT_N_ITER/_INIT_EXTRAS before each polish call.
SCHED_FULL  = (12, [(6, 1.5), (6, 0.5), (6, 0.15), (6, 0.05),
                    (3, 1.0), (3, 0.15), (3, 0.05)])
SCHED_TIGHT = (4,  [(3, 1.5), (3, 0.5), (3, 0.15), (3, 0.05),
                    (2, 1.0), (2, 0.15), (2, 0.05)])
SCHED_SUPER = (3,  [(2, 1.5), (2, 0.5), (2, 0.15), (2, 0.05),
                    (2, 1.0), (1, 0.15), (1, 0.05)])
SCHED_LIGHT = (4,  [(2, 1.5), (2, 0.5), (2, 0.15), (2, 0.05)])
SCHED_TINY  = (2,  [(1, 1.5), (1, 0.5), (1, 0.15), (1, 0.05)])
SCHED_MED   = (6,  [(3, 1.5), (3, 0.5), (3, 0.15), (3, 0.05), (2, 1.0)])


_GAUSS_CACHE = {}


def _gaussian_perturb(cur_pos, sizes, mov_i32, n_macros, nh, cw, ch, octx,
                       sigma_bins, seed=42, K_frac=0.087):
    """Random Gaussian perturb (sigma in bin units) on K random soft macros,
    then clamp to canvas."""
    rng = np.random.default_rng(seed)
    soft_idx = np.where(mov_i32 != 0)[0]
    soft_idx = soft_idx[soft_idx >= nh]
    K = max(50, min(200, int(round(K_frac * len(soft_idx)))))
    K = min(K, len(soft_idx))
    chosen = rng.choice(soft_idx, size=K, replace=False)
    gw = cw / octx['gc']
    gh = ch / octx['gr']
    new_pos = cur_pos.copy()
    new_pos[chosen, 0] += rng.normal(0, sigma_bins * gw, K)
    new_pos[chosen, 1] += rng.normal(0, sigma_bins * gh, K)
    new_pos[:, 0] = np.clip(new_pos[:, 0], sizes[:, 0] / 2,
                              cw - sizes[:, 0] / 2)
    new_pos[:, 1] = np.clip(new_pos[:, 1], sizes[:, 1] / 2,
                              ch - sizes[:, 1] / 2)
    return new_pos


def _cong_gradient_perturb(cur_pos, cong_state, sizes, mov_i32, n_macros,
                             nh, cw, ch, octx, lr_grad):
    """Vectorized cong-gradient perturb: shift each soft macro AWAY from its
    3x3 cong-weighted center of mass by lr_grad*min(gw,gh)."""
    cong_state.pos.copy_(torch.from_numpy(np.ascontiguousarray(
        cur_pos, dtype=np.float64)).to(cong_state.device, cong_state.dtype))
    _gpu_init_state(cong_state)
    GR = octx['gr']
    GC = octx['gc']
    H = cong_state.H_final.cpu().numpy().reshape(GR, GC)
    V = cong_state.V_final.cpu().numpy().reshape(GR, GC)
    cong = (H + V).astype(np.float64)
    bin_w = cw / GC
    bin_h = ch / GR
    step = lr_grad * min(bin_w, bin_h)

    movable_soft = (mov_i32 != 0).copy()
    movable_soft[:nh] = False
    idxs = np.where(movable_soft)[0]
    if len(idxs) == 0:
        return cur_pos.copy()

    xs = cur_pos[idxs, 0]
    ys = cur_pos[idxs, 1]
    cols = np.clip((xs / bin_w).astype(np.int64), 0, GC - 1)
    rows = np.clip((ys / bin_h).astype(np.int64), 0, GR - 1)
    pad = np.pad(cong, 1, mode='constant', constant_values=0.0)
    rr = rows[:, None, None] + np.array([0, 1, 2])[None, :, None]
    cc_idx = cols[:, None, None] + np.array([0, 1, 2])[None, None, :]
    nbrs = pad[rr, cc_idx]
    nbr_rows = rows[:, None, None] + np.array([-1, 0, 1])[None, :, None]
    nbr_cols = cols[:, None, None] + np.array([-1, 0, 1])[None, None, :]
    w = nbrs.reshape(len(idxs), 9)
    w_sum = w.sum(axis=1)
    canvas_x = (nbr_cols + 0.5) * bin_w
    canvas_y = (nbr_rows + 0.5) * bin_h
    canvas_x = np.broadcast_to(canvas_x, nbrs.shape).reshape(len(idxs), 9)
    canvas_y = np.broadcast_to(canvas_y, nbrs.shape).reshape(len(idxs), 9)
    safe_w = np.where(w_sum > 1e-9, w_sum, 1.0)
    com_x = (canvas_x * w).sum(axis=1) / safe_w
    com_y = (canvas_y * w).sum(axis=1) / safe_w
    dx = xs - com_x
    dy = ys - com_y
    d = np.sqrt(dx * dx + dy * dy)
    valid = (d > 1e-9) & (w_sum > 1e-9)
    new_x = xs.copy()
    new_y = ys.copy()
    new_x[valid] = xs[valid] + step * dx[valid] / d[valid]
    new_y[valid] = ys[valid] + step * dy[valid] / d[valid]

    new_pos = cur_pos.copy()
    new_pos[idxs, 0] = new_x
    new_pos[idxs, 1] = new_y
    new_pos[:, 0] = np.clip(new_pos[:, 0], sizes[:, 0] / 2,
                              cw - sizes[:, 0] / 2)
    new_pos[:, 1] = np.clip(new_pos[:, 1], sizes[:, 1] / 2,
                              ch - sizes[:, 1] / 2)
    return new_pos


def _wire_mask_perturb_local(cur_pos, sizes, mov_i32, n_macros, nh, pn,
                              cw, ch, octx, lr_grad, wl_extra_weight):
    """WireMask perturb: step soft macros along -HPWL gradient with magnitude
    lr_grad*min(gw,gh)/max_norm, then clamp."""
    grad, _ = _compute_hpwl_gradient(
        cur_pos, sizes, mov_i32, n_macros, nh, pn,
        wl_extra_weight=wl_extra_weight)
    device = grad.device
    gw = cw / octx['gc']
    gh = ch / octx['gr']
    soft_grad = grad[nh:]
    if soft_grad.shape[0] == 0:
        return cur_pos.copy()
    soft_norm = soft_grad.norm(dim=1)
    max_norm = float(soft_norm.max().item())
    if max_norm < 1e-12:
        return cur_pos.copy()
    scale = (lr_grad * min(gw, gh)) / max_norm
    cur_gpu = torch.from_numpy(np.ascontiguousarray(
        cur_pos, dtype=np.float64)).to(device)
    pos_t = cur_gpu.clone()
    pos_t[nh:] = pos_t[nh:] - scale * soft_grad
    sizes_t = torch.from_numpy(np.ascontiguousarray(
        sizes, dtype=np.float64)).to(device)
    half_w = sizes_t[:, 0] * 0.5
    half_h = sizes_t[:, 1] * 0.5
    pos_t[:, 0] = torch.clamp(pos_t[:, 0], half_w, cw - half_w)
    pos_t[:, 1] = torch.clamp(pos_t[:, 1], half_h, ch - half_h)
    pos_t[:nh] = cur_gpu[:nh]
    return pos_t.cpu().numpy()


def _adaptive_rounds(cong):
    """Return per-band round list of (lr, schedule, c_short, ptype, seed);
    ptype in {'wm','cg','gs'}."""
    if cong < 1.0:
        # Low-cong (ibm01-class): WM-only, c_short on last 3 rounds.
        return [(0.10, SCHED_MED,  False, 'wm', 42),
                (0.05, SCHED_MED,  False, 'wm', 42),
                (0.025, SCHED_MED, True,  'wm', 42),
                (0.01,  SCHED_MED, True,  'wm', 42),
                (0.005, SCHED_TINY, True, 'wm', 42)]
    elif cong < 1.5:
        # Mid-cong (ibm03-class): WM full chain with c_short on last 2.
        return [(0.25,   SCHED_FULL,  False, 'wm', 42),
                (0.10,   SCHED_TIGHT, False, 'wm', 42),
                (0.05,   SCHED_SUPER, False, 'wm', 42),
                (0.025,  SCHED_SUPER, True,  'wm', 42),
                (0.0125, SCHED_TINY,  True,  'wm', 42)]
    else:
        # High-congestion band: a 2-round chain (WM then cg) under a 60s
        # budget. The round-1 WM(0.10) is load-bearing - without it the cg
        # round rejects.
        return [(0.10, SCHED_LIGHT, True, 'wm', 42),
                (0.10, SCHED_LIGHT, True, 'cg', 42)]


def basin_jump_v2(
    cur, sizes, mov_i32, n_macros, nh, cw, ch, octx, pn,
    *,
    cc_call,
    gpu_state,
    pre_bj_proxy=None,
    wl_extra_weight=None,
    verbose=False,
):
    """Run V2 basin-jump; cc_call mirrors placer._cc_call; one persistent
    GPUState is fine (each call re-runs init_state); pre_bj_proxy is purely
    for the caller's revert decisions, chain gating uses fresh n_iter=0 cong
    eval."""
    from .polish_hybrid_5perturb import gpu_hybrid_5perturb_polish
    from . import polish_hybrid_5perturb as _ph5
    from .state import GPUState

    # Pre-built c_short callable to skip per-round lambda allocation.
    _cs_full = lambda p: cc_call(p, n_iter=3, lr_bins=1.0,
                                  extra_phases=[(3, 0.15), (3, 0.05)])

    def _gpu_polish(cur_pos, init_n, init_extras, c_short):
        saved = (_ph5._INIT_N_ITER, _ph5._INIT_EXTRAS)
        _ph5._INIT_N_ITER = init_n
        _ph5._INIT_EXTRAS = list(init_extras)
        try:
            r = gpu_hybrid_5perturb_polish(
                cur_pos, sizes, mov_i32, n_macros, nh, cw, ch, octx, pn,
                c_short_call=(_cs_full if c_short else None),
                n_perturbs=0, verbose=False, state=gpu_state)
            return r[0]
        finally:
            _ph5._INIT_N_ITER, _ph5._INIT_EXTRAS = saved

    def c_eval(p):
        return float(cc_call(p, n_iter=0, lr_bins=2.0)[3])

    # A fresh n_iter=0 cc-eval gives the chain's accept gate an
    # apples-to-apples baseline; the upstream pre_bj_proxy is post-polish and
    # not comparable.
    pre_eval = cc_call(cur, n_iter=0, lr_bins=2.0)
    base_proxy = float(pre_eval[3])
    cong = float(pre_eval[1])

    rounds = _adaptive_rounds(cong)

    # Build cong_state lazily: only the high-congestion band runs 'cg' rounds,
    # so everything else skips ~0.1-0.3s of from_pn + init_state.
    needs_cong_state = any(r[3] == 'cg' for r in rounds)
    cong_state = None
    if needs_cong_state:
        cong_state = GPUState.from_pn(
            pn=pn, pin_csr=octx['pin_csr'], octx=octx,
            sizes_np=sizes, positions_np=cur.copy(),
            movable_np=mov_i32, n_macros=n_macros, n_hard=nh)
        _gpu_init_state(cong_state)

    accepted_rounds = 0
    aborted_first = False
    cur_p = base_proxy
    bj_cur = cur.copy()

    _bj_det = os.environ.get("XP_DET_TRACE", "0") == "1"
    for ri, (lr, (init_n, init_extras), c_short, ptype, seed) in enumerate(rounds):
        t_r = time.time()
        if ptype == 'wm':
            perturbed = _wire_mask_perturb_local(
                bj_cur, sizes, mov_i32, n_macros, nh, pn,
                cw, ch, octx, lr, wl_extra_weight)
        elif ptype == 'cg':
            perturbed = _cong_gradient_perturb(
                bj_cur, cong_state, sizes, mov_i32, n_macros, nh,
                cw, ch, octx, lr)
        else:  # gs
            perturbed = _gaussian_perturb(
                bj_cur, sizes, mov_i32, n_macros, nh, cw, ch, octx,
                sigma_bins=lr, seed=seed)
        if _bj_det:
            print(f"[DET] v2 B{ri+1} pre-polish ptype={ptype} perturb_sum={float(np.asarray(perturbed, dtype=np.float64).sum())!r}")

        new_state = _gpu_polish(perturbed, init_n, init_extras, c_short)
        new_p = c_eval(new_state)
        if _bj_det:
            print(f"[DET] v2 B{ri+1} post-polish state_sum={float(np.asarray(new_state, dtype=np.float64).sum())!r} new_p={new_p!r}")
        accepted = new_p < cur_p - 1e-9
        r_wall = time.time() - t_r
        if verbose:
            tag = '+' if accepted else '.'
            print(f"    B{ri+1} {ptype} lr={lr:<6.4f} "
                  f"c-proxy {cur_p:.8f}->{new_p:.8f} {tag} "
                  f"({r_wall:.2f}s)")
        if accepted:
            bj_cur = new_state
            cur_p = new_p
            accepted_rounds += 1
        elif ri == 0:
            aborted_first = True
            break

    return {
        'cur': bj_cur,
        'final_proxy': cur_p,
        'base_proxy': base_proxy,
        'cong': cong,
        'accepted_rounds': accepted_rounds,
        'rounds_run': ri + 1,
        'total_rounds': len(rounds),
        'aborted_first': aborted_first,
    }
