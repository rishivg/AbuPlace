"""Xplace gradient calculator — MODIFIED from upstream.

Upstream: https://github.com/cuhk-eda/Xplace (BSD 3-Clause, see ../../LICENSE).

Local additions by this project (not present upstream): the RUDY congestion
losses (_compute_rudy_bbox_loss, _compute_rudy_hv_loss, _compute_rudy_abu_loss)
and the _maybe_apply_cong_loss hook that injects them into the Nesterov
gradient. They are driven by the XP_CONG_LOSS / XP_CONG_LOSS_KERNEL environment
variables and are what the placer's best-of-N trajectories select between.
Everything else is upstream Xplace.
"""

import os
import torch
from .param_scheduler import ParamScheduler
from .core import merged_wl_loss_grad, merged_wl_loss_grad_timing

def apply_precond(mov_node_pos: torch.Tensor, ps: ParamScheduler, args):
    if not args.use_precond:
        return
    mov_node_pos.grad /= ps.precond_weight
    return mov_node_pos.grad


def _compute_rudy_bbox_loss(conn_node_pos, data, num_bin_x, num_bin_y,
                              cw, ch, cap_frac):
    """True RUDY: spread net HPWL × net_weight uniformly over each bbox.

    For each net e:
      hpwl_e = (xmax-xmin) + (ymax-ymin)
      bbox_area_e = (xmax-xmin) × (ymax-ymin)  (clamped)
    For each bin (i,j):
      contrib_e_i_j = hpwl_e × frac × indicator(bin in bbox_e)
        where frac = bin_area / bbox_area_e
    Sum contributions per bin → routing demand map.
    Loss: mean(relu(d/mean - cap_frac))^2.

    Soft indicator via two sigmoids per axis. Sigma=0.5×bin_size for sharp
    transitions yet differentiable.
    """
    pin_pos = conn_node_pos[data.pin_id2node_id] + data.pin_rel_cpos
    px = pin_pos[:, 0]
    py = pin_pos[:, 1]
    pin_net = data.pin_id2net_id
    n_nets = data.hyperedge_list_end.shape[0]
    device = px.device
    dtype = px.dtype

    init_pos = torch.full((n_nets,), -1e10, device=device, dtype=dtype)
    init_neg = torch.full((n_nets,), 1e10, device=device, dtype=dtype)
    net_xmax = init_pos.clone().scatter_reduce_(
        0, pin_net, px, reduce='amax', include_self=False)
    net_xmin = init_neg.clone().scatter_reduce_(
        0, pin_net, px, reduce='amin', include_self=False)
    net_ymax = init_pos.clone().scatter_reduce_(
        0, pin_net, py, reduce='amax', include_self=False)
    net_ymin = init_neg.clone().scatter_reduce_(
        0, pin_net, py, reduce='amin', include_self=False)

    valid = data.net_mask.to(dtype=dtype)
    net_w = (net_xmax - net_xmin).clamp(min=0)
    net_h = (net_ymax - net_ymin).clamp(min=0)
    net_hpwl = (net_w + net_h) * data.net_weight * valid
    bbox_area = (net_w * net_h).clamp(min=1e-3)
    rudy_per_area = net_hpwl / bbox_area  # demand density (per unit area)

    # Bin centers
    bin_w = cw / num_bin_x
    bin_h = ch / num_bin_y
    bx = (torch.arange(num_bin_x, device=device, dtype=dtype) + 0.5) * bin_w
    by = (torch.arange(num_bin_y, device=device, dtype=dtype) + 0.5) * bin_h

    # For each (net, bin_x): soft indicator of bbox containing bin x-center
    # = sigmoid((xmax - bx) / sigma) × sigmoid((bx - xmin) / sigma)
    sigma_x = bin_w * 0.5
    sigma_y = bin_h * 0.5
    # Shape: (n_nets, num_bin_x)
    Ix = (torch.sigmoid((net_xmax.unsqueeze(1) - bx.unsqueeze(0)) / sigma_x)
          * torch.sigmoid((bx.unsqueeze(0) - net_xmin.unsqueeze(1)) / sigma_x))
    Iy = (torch.sigmoid((net_ymax.unsqueeze(1) - by.unsqueeze(0)) / sigma_y)
          * torch.sigmoid((by.unsqueeze(0) - net_ymin.unsqueeze(1)) / sigma_y))
    # rudy_per_area broadcast to (n_nets, 1, 1) then × Ix.unsqueeze(2) × Iy.unsqueeze(1)
    # -> (n_nets, num_bin_x, num_bin_y) — too big. Sum along net dim:
    # density[i,j] = sum_e rudy_per_area_e × Ix[e,i] × Iy[e,j] × bin_area
    # Compute via einsum: density = (rudy * Ix.T) @ Iy  [contract over nets]
    weighted_Ix = rudy_per_area.unsqueeze(1) * Ix  # (n_nets, num_bin_x)
    # (num_bin_x, n_nets) @ (n_nets, num_bin_y) = (num_bin_x, num_bin_y)
    density = weighted_Ix.transpose(0, 1) @ Iy
    density = density * (bin_w * bin_h)  # per-bin contribution

    density_flat = density.reshape(-1)
    n_bins = num_bin_x * num_bin_y
    m = density_flat.sum() / n_bins + 1e-12
    excess = torch.relu(density_flat / m - cap_frac)
    # Robust peak/mean (99th pctile, not raw max — degenerate-bbox nets can
    # spike a single bin) — how congestion-bound this design is, for adaptive W.
    peak_ratio = float((torch.quantile(density_flat, 0.99) / m).item())
    return (excess * excess).mean(), peak_ratio


def _compute_rudy_hv_loss(conn_node_pos, data, num_bin_x, num_bin_y,
                          cw, ch, cap_frac):
    """Directional RUDY (H/V split). Unlike _compute_rudy_bbox_loss, which
    collapses demand into one scalar (net_w + net_h) per net, this spreads the
    HORIZONTAL routing demand (∝ net width) and the VERTICAL demand (∝ net
    height) into SEPARATE per-bin maps, then penalizes each against its own
    per-direction mean. This mirrors how the proxy (cong_relax_v2) scores H and
    V congestion separately via hrpm/vrpm — a net's horizontal span loads
    horizontal tracks, its vertical span loads vertical tracks.

    Loss = mean(relu(d_h/mean_h - cap)^2) + mean(relu(d_v/mean_v - cap)^2).
    """
    pin_pos = conn_node_pos[data.pin_id2node_id] + data.pin_rel_cpos
    px = pin_pos[:, 0]
    py = pin_pos[:, 1]
    pin_net = data.pin_id2net_id
    n_nets = data.hyperedge_list_end.shape[0]
    device = px.device
    dtype = px.dtype

    init_pos = torch.full((n_nets,), -1e10, device=device, dtype=dtype)
    init_neg = torch.full((n_nets,), 1e10, device=device, dtype=dtype)
    net_xmax = init_pos.clone().scatter_reduce_(0, pin_net, px, reduce='amax', include_self=False)
    net_xmin = init_neg.clone().scatter_reduce_(0, pin_net, px, reduce='amin', include_self=False)
    net_ymax = init_pos.clone().scatter_reduce_(0, pin_net, py, reduce='amax', include_self=False)
    net_ymin = init_neg.clone().scatter_reduce_(0, pin_net, py, reduce='amin', include_self=False)

    valid = data.net_mask.to(dtype=dtype)
    net_w = (net_xmax - net_xmin).clamp(min=0)
    net_h = (net_ymax - net_ymin).clamp(min=0)
    bbox_area = (net_w * net_h).clamp(min=1e-3)
    # Directional demand densities: H ∝ width, V ∝ height (each × net_weight).
    h_per_area = (net_w * data.net_weight * valid) / bbox_area
    v_per_area = (net_h * data.net_weight * valid) / bbox_area

    bin_w = cw / num_bin_x
    bin_h = ch / num_bin_y
    bx = (torch.arange(num_bin_x, device=device, dtype=dtype) + 0.5) * bin_w
    by = (torch.arange(num_bin_y, device=device, dtype=dtype) + 0.5) * bin_h
    sigma_x = bin_w * 0.5
    sigma_y = bin_h * 0.5
    Ix = (torch.sigmoid((net_xmax.unsqueeze(1) - bx.unsqueeze(0)) / sigma_x)
          * torch.sigmoid((bx.unsqueeze(0) - net_xmin.unsqueeze(1)) / sigma_x))
    Iy = (torch.sigmoid((net_ymax.unsqueeze(1) - by.unsqueeze(0)) / sigma_y)
          * torch.sigmoid((by.unsqueeze(0) - net_ymin.unsqueeze(1)) / sigma_y))

    bin_area = bin_w * bin_h
    density_h = ((h_per_area.unsqueeze(1) * Ix).transpose(0, 1) @ Iy) * bin_area
    density_v = ((v_per_area.unsqueeze(1) * Ix).transpose(0, 1) @ Iy) * bin_area

    n_bins = num_bin_x * num_bin_y
    dh = density_h.reshape(-1)
    dv = density_v.reshape(-1)
    mh = dh.sum() / n_bins + 1e-12
    mv = dv.sum() / n_bins + 1e-12
    eh = torch.relu(dh / mh - cap_frac)
    ev = torch.relu(dv / mv - cap_frac)
    peak_ratio = float(max((torch.quantile(dh, 0.99) / mh).item(),
                           (torch.quantile(dv, 0.99) / mv).item()))
    return (eh * eh).mean() + (ev * ev).mean(), peak_ratio


def _compute_rudy_abu_loss(conn_node_pos, data, num_bin_x, num_bin_y,
                           cw, ch, cap_frac, top_frac=0.05):
    """ABU-matched directional congestion loss. The evaluator scores congestion
    as ABU = average of the top ~5% most-congested gcells (plc_client_os.py:912),
    a PEAK metric — not the mean-excess that _compute_rudy_bbox_loss penalizes.
    torch.topk(demand, k).mean() IS that functional and is differentiable
    (gradient flows to the macros forming the worst bins), so minimizing it
    directly reduces the scored quantity's shape. Directional H/V split mirrors
    the per-direction H/V routing the evaluator accumulates.

    Loss = mean(topk(d_h/mean_h, 5%)) + mean(topk(d_v/mean_v, 5%)). (cap_frac
    unused; kept for signature parity with the other kernels.)
    """
    pin_pos = conn_node_pos[data.pin_id2node_id] + data.pin_rel_cpos
    px = pin_pos[:, 0]
    py = pin_pos[:, 1]
    pin_net = data.pin_id2net_id
    n_nets = data.hyperedge_list_end.shape[0]
    device = px.device
    dtype = px.dtype

    init_pos = torch.full((n_nets,), -1e10, device=device, dtype=dtype)
    init_neg = torch.full((n_nets,), 1e10, device=device, dtype=dtype)
    net_xmax = init_pos.clone().scatter_reduce_(0, pin_net, px, reduce='amax', include_self=False)
    net_xmin = init_neg.clone().scatter_reduce_(0, pin_net, px, reduce='amin', include_self=False)
    net_ymax = init_pos.clone().scatter_reduce_(0, pin_net, py, reduce='amax', include_self=False)
    net_ymin = init_neg.clone().scatter_reduce_(0, pin_net, py, reduce='amin', include_self=False)

    valid = data.net_mask.to(dtype=dtype)
    net_w = (net_xmax - net_xmin).clamp(min=0)
    net_h = (net_ymax - net_ymin).clamp(min=0)
    bbox_area = (net_w * net_h).clamp(min=1e-3)
    h_per_area = (net_w * data.net_weight * valid) / bbox_area
    v_per_area = (net_h * data.net_weight * valid) / bbox_area

    bin_w = cw / num_bin_x
    bin_h = ch / num_bin_y
    bx = (torch.arange(num_bin_x, device=device, dtype=dtype) + 0.5) * bin_w
    by = (torch.arange(num_bin_y, device=device, dtype=dtype) + 0.5) * bin_h
    sigma_x = bin_w * 0.5
    sigma_y = bin_h * 0.5
    Ix = (torch.sigmoid((net_xmax.unsqueeze(1) - bx.unsqueeze(0)) / sigma_x)
          * torch.sigmoid((bx.unsqueeze(0) - net_xmin.unsqueeze(1)) / sigma_x))
    Iy = (torch.sigmoid((net_ymax.unsqueeze(1) - by.unsqueeze(0)) / sigma_y)
          * torch.sigmoid((by.unsqueeze(0) - net_ymin.unsqueeze(1)) / sigma_y))
    bin_area = bin_w * bin_h
    density_h = ((h_per_area.unsqueeze(1) * Ix).transpose(0, 1) @ Iy) * bin_area
    density_v = ((v_per_area.unsqueeze(1) * Ix).transpose(0, 1) @ Iy) * bin_area

    n_bins = num_bin_x * num_bin_y
    k = max(1, int(top_frac * n_bins))
    dh = density_h.reshape(-1)
    dv = density_v.reshape(-1)
    mh = dh.sum() / n_bins + 1e-12
    mv = dv.sum() / n_bins + 1e-12
    nh_ratio = dh / mh
    nv_ratio = dv / mv
    abu_h = torch.topk(nh_ratio, k).values.mean()
    abu_v = torch.topk(nv_ratio, k).values.mean()
    peak_ratio = float(max(abu_h.item(), abu_v.item()))
    return abu_h + abu_v, peak_ratio


def _compute_pindensity_loss(conn_node_pos, data, num_bin_x, num_bin_y,
                              cw, ch, cap_frac):
    """Pin-density congestion proxy, fully differentiable.

    Each pin contributes 1.0 (bilinear-distributed) to its bin → pin count
    per bin. High pin count = high routing demand (well-known proxy).

    loss = mean(relu(pc / mean(pc) - cap_frac))^2  ∈ [0, ~few]
    """
    pin_pos = conn_node_pos[data.pin_id2node_id] + data.pin_rel_cpos
    px = pin_pos[:, 0].clamp(min=0, max=cw)
    py = pin_pos[:, 1].clamp(min=0, max=ch)
    device = px.device
    dtype = px.dtype

    bx_f = (px / cw * num_bin_x).clamp(0, num_bin_x - 1.001)
    by_f = (py / ch * num_bin_y).clamp(0, num_bin_y - 1.001)
    bx_lo = torch.floor(bx_f).long()
    by_lo = torch.floor(by_f).long()
    bx_hi = bx_lo + 1
    by_hi = by_lo + 1
    fx = bx_f - bx_lo.to(dtype=dtype)
    fy = by_f - by_lo.to(dtype=dtype)

    n_bins = num_bin_x * num_bin_y
    density = torch.zeros(n_bins, device=device, dtype=dtype)
    flat_ll = bx_lo * num_bin_y + by_lo
    flat_hl = bx_hi * num_bin_y + by_lo
    flat_lh = bx_lo * num_bin_y + by_hi
    flat_hh = bx_hi * num_bin_y + by_hi
    density = density.scatter_add(0, flat_ll, (1 - fx) * (1 - fy))
    density = density.scatter_add(0, flat_hl, fx * (1 - fy))
    density = density.scatter_add(0, flat_lh, (1 - fx) * fy)
    density = density.scatter_add(0, flat_hh, fx * fy)

    m = density.sum() / n_bins + 1e-12
    excess = torch.relu(density / m - cap_frac)
    peak_ratio = float((torch.quantile(density, 0.99) / m).item())
    return (excess * excess).mean(), peak_ratio


def _maybe_apply_cong_loss(mov_node_pos, ps, data, args,
                            mov_lhs, mov_rhs, conn_fix_node_pos):
    """If XP_CONG_LOSS=1, compute a differentiable congestion loss + gradient.

    Off unless XP_CONG_LOSS=1; the placer's best-of-N loop turns it on for the
    RUDY trajectories. Every knob is documented in abuplace/TUNING.md — defaults
    are not repeated here, so they cannot drift out of sync with the code below.

    Uses a fresh-leaf clone of mov_node_pos so its autograd graph is independent
    of the existing manually-computed WL/density gradients. Returns the scalar
    loss, already added to mov_node_pos.grad.
    """
    if os.environ.get("XP_CONG_LOSS", "0") != "1":
        return 0.0
    start_iter = int(os.environ.get("XP_CONG_LOSS_START", "100"))
    if ps.iter < start_iter:
        return 0.0
    # 2C: subsample (apply every K-th iter) to reduce per-iter wall.
    # Gradient is amplified by K to keep average effect ~constant.
    every = int(os.environ.get("XP_CONG_LOSS_EVERY", "1"))
    if every > 1:
        if (ps.iter - start_iter) % every != 0:
            return 0.0
    ramp_iter = int(os.environ.get("XP_CONG_LOSS_RAMP", "100"))
    weight = float(os.environ.get("XP_CONG_LOSS_W", "0.05"))
    bins = int(os.environ.get("XP_CONG_LOSS_BINS", "32"))
    cap_frac = float(os.environ.get("XP_CONG_LOSS_CAP", "1.5"))
    macros_only = os.environ.get("XP_CONG_LOSS_MACROS_ONLY", "1") == "1"
    # Adaptive scaling by current overflow: when GP is near-converged (low
    # overflow), scale down so cong_loss doesn't disrupt the basin.
    # adaptive=0 disables (constant scale); >0 = multiplier on overflow.
    adaptive = float(os.environ.get("XP_CONG_LOSS_ADAPT_OF", "0.0"))

    cw = float((data.die_ur[0] - data.die_ll[0]).item())
    ch = float((data.die_ur[1] - data.die_ll[1]).item())

    progress = min(1.0, max(0.0, (ps.iter - start_iter) / max(1, ramp_iter)))
    eff_w = weight * progress * float(ps.density_weight)
    if adaptive > 0 and len(ps.recorder.overflow) > 0:
        of = float(ps.recorder.overflow[-1])
        eff_w *= min(1.0, of * adaptive)
    # 2B: smooth ramp-down as GP converges (overflow drops). When the
    # natural gradient is small (overflow << 0.10), our cong gradient can
    # disrupt; taper smoothly. XP_CONG_LOSS_RAMPDOWN_OF: overflow threshold
    # below which to start scaling down (default 0 = disabled).
    rampdown_of = float(os.environ.get("XP_CONG_LOSS_RAMPDOWN_OF", "0.0"))
    if rampdown_of > 0 and len(ps.recorder.overflow) > 0:
        of = float(ps.recorder.overflow[-1])
        if of < rampdown_of:
            # Scale linearly from 1.0 (at of=rampdown_of) to floor (at of=0)
            floor = float(os.environ.get("XP_CONG_LOSS_RAMPDOWN_FLOOR", "0.2"))
            scale = floor + (1.0 - floor) * (of / rampdown_of)
            eff_w *= scale
    if eff_w <= 0:
        return 0.0

    pos_leaf = mov_node_pos.detach().clone().requires_grad_(True)
    cn = torch.cat([pos_leaf[mov_lhs:mov_rhs], conn_fix_node_pos], dim=0)
    # rudy = HPWL × net_weight spread over bbox (DCGP-style). Validated
    # ~2× stronger than pindensity at same W on ibm18 (V4 sweep).
    kernel = os.environ.get("XP_CONG_LOSS_KERNEL", "rudy")
    if kernel == "rudy":
        cong_loss, peak_ratio = _compute_rudy_bbox_loss(
            cn, data, bins, bins, cw, ch, cap_frac)
    elif kernel == "rudy_hv":
        cong_loss, peak_ratio = _compute_rudy_hv_loss(
            cn, data, bins, bins, cw, ch, cap_frac)
    elif kernel == "rudy_abu":
        cong_loss, peak_ratio = _compute_rudy_abu_loss(
            cn, data, bins, bins, cw, ch, cap_frac)
    else:
        cong_loss, peak_ratio = _compute_pindensity_loss(
            cn, data, bins, bins, cw, ch, cap_frac)
    # Demand-peak-adaptive weighting (uniformity fix): scale the congestion push
    # by how congestion-bound the design actually is (99th-pctile/mean of the
    # demand map). Designs without real hotspots (low peak_ratio) get ~0 weight
    # => do-no-harm on WL/density-bound benches (ibm02/03/04); congested benches
    # (ibm06/18) get full strength. Off by default. XP_CONG_LOSS_ADAPT_PEAK=1.
    if os.environ.get("XP_CONG_LOSS_ADAPT_PEAK", "0") == "1":
        plo = float(os.environ.get("XP_CONG_LOSS_ADAPT_PEAK_LO", "1.5"))
        phi = float(os.environ.get("XP_CONG_LOSS_ADAPT_PEAK_HI", "2.5"))
        pscale = min(1.0, max(0.0, (peak_ratio - plo) / max(1e-6, phi - plo)))
        eff_w = eff_w * pscale
        if ps.iter % 100 == 0:
            print(f"  [cong_loss] iter={ps.iter} peak_ratio={peak_ratio:.3f} "
                  f"pscale={pscale:.3f} eff_w={eff_w:.4e}", flush=True)
    cong_loss.backward()
    grad_to_add = eff_w * pos_leaf.grad
    if macros_only:
        # zero out the std-cell gradient — only push macros around
        is_macro = data.is_mov_macro
        if is_macro is not None:
            # Build full-length macro mask matching mov_node_pos shape
            mask = torch.zeros_like(grad_to_add[:, 0], dtype=torch.bool)
            mask[mov_lhs:mov_rhs] = is_macro[mov_lhs:mov_rhs]
            grad_to_add = grad_to_add * mask.unsqueeze(1).to(grad_to_add.dtype)
    mov_node_pos.grad += grad_to_add
    return float(eff_w * cong_loss.item())


def calc_obj_and_grad(
    mov_node_pos,
    constraint_fn=None,
    route_fn=None,
    mov_node_size=None,
    expand_ratio=None,
    init_density_map=None,
    density_map_layer=None,
    conn_fix_node_pos=None,
    ps=None,
    data=None,
    args=None,
    merged_forward_backward=True,
):
    mov_lhs, mov_rhs = data.movable_index
    mov_node_pos = constraint_fn(mov_node_pos)
    conn_node_pos = mov_node_pos[mov_lhs:mov_rhs, ...]
    conn_node_pos = torch.cat([conn_node_pos, conn_fix_node_pos], dim=0)

    assert merged_forward_backward
    if merged_forward_backward:
        if mov_node_pos.grad is not None:
            mov_node_pos.grad.zero_()
        else:
            mov_node_pos.grad = torch.zeros_like(mov_node_pos).detach()

        if ps.use_route_force and ps.start_route_opt:
            mov_route_grad, mov_congest_grad, mov_pseudo_grad = route_fn(
                mov_node_pos, mov_node_size, expand_ratio, constraint_fn
            )
            mov_node_pos.grad += mov_route_grad * ps.route_weight
            mov_node_pos.grad += mov_congest_grad * ps.congest_weight
            mov_node_pos.grad += mov_pseudo_grad * ps.pseudo_weight

        wl_loss, conn_node_grad_by_wl = merged_wl_loss_grad(
            conn_node_pos, data.pin_id2node_id, data.pin_rel_cpos,
            data.node2pin_list, data.node2pin_list_end,
            data.hyperedge_list, data.hyperedge_list_end, data.net_mask,
            data.hpwl_scale, ps.wa_coeff, args.deterministic
        )
        mov_node_pos.grad[mov_lhs:mov_rhs] += conn_node_grad_by_wl[mov_lhs:mov_rhs]

        if ps.enable_timing:
            wl_loss_timing, conn_node_grad_by_timing = merged_wl_loss_grad_timing(
                conn_node_pos, data.gputimer.timing_pin_weight,
                data.pin_id2node_id, data.pin_rel_cpos,
                data.node2pin_list, data.node2pin_list_end, data.hyperedge_list, data.hyperedge_list_end,
                data.net_mask, data.net_weight, data.hpwl_scale, ps.wa_coeff, args.deterministic
            )
            mov_node_pos.grad[mov_lhs:mov_rhs] += conn_node_grad_by_timing[mov_lhs:mov_rhs]
            wl_loss += wl_loss_timing

        if ps.enable_sample_force:
            if ps.iter > 3 and ps.iter % 20 == 0:
                # ps.iter > 3 for warmup
                density_loss, _, node_grad_by_density = density_map_layer.merged_density_loss_grad(
                    mov_node_pos, mov_node_size, init_density_map, calc_overflow=False
                )
                ps.force_ratio = (
                    ps.density_weight * node_grad_by_density[mov_lhs:mov_rhs].norm(p=1) /
                    conn_node_grad_by_wl[mov_lhs:mov_rhs].norm(p=1)
                ).clamp_(max=10)
                mov_node_pos.grad += node_grad_by_density * ps.density_weight
            else:
                density_loss = 0.0
            if (ps.iter > 3 and ps.recorder.force_ratio[-1] > 1e-2) or ps.iter > 100:
                # no longer enable sampling back
                ps.enable_sample_force = False
        else:
            density_loss, _, node_grad_by_density = density_map_layer.merged_density_loss_grad(
                mov_node_pos, mov_node_size, init_density_map, calc_overflow=False
            )
            mov_node_pos.grad += node_grad_by_density * ps.density_weight

        if ps.zero_macro_grad:
            mov_node_pos.grad[mov_lhs:mov_rhs].masked_fill_(data.is_mov_macro[mov_lhs:mov_rhs].unsqueeze(1), 0)

        # Differentiable RUDY congestion gradient (XP_CONG_LOSS=1).
        # Applied AFTER zero_macro_grad — so the cong gradient does affect
        # macros (which is the point — we want to push macros out of
        # congested regions). Skip if disabled to keep legacy bit-identical.
        cong_loss_val = _maybe_apply_cong_loss(
            mov_node_pos, ps, data, args, mov_lhs, mov_rhs, conn_fix_node_pos)

        grad = apply_precond(mov_node_pos, ps, args)
        loss = wl_loss + ps.density_weight * density_loss + cong_loss_val

    return loss, grad
