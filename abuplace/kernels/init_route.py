"""Batched init_state routing: one Triton program per net dedups unique gcells,
dispatches Steiner (2/3/N-pin star) and emits sparse (cell, sign*marker)
tuples that the caller scatters into state.H_net/V_net."""

import torch
import triton
import triton.language as tl

from .routing import route_star, route_three_pin, route_two_pin


@triton.jit
def _init_route_kernel(
    net_driver_ptr, net_sinks_off_ptr, net_sinks_idx_ptr, net_weight_ptr,
    pin_row_ptr, pin_col_ptr,
    H_idx_ptr, H_val_ptr, V_idx_ptr, V_val_ptr,
    gc, gr,
    MAX_CELLS: tl.constexpr,
    MAX_PINS_PER_NET: tl.constexpr,
    EMIT_BLK: tl.constexpr,
    SLOTS: tl.constexpr,
):
    """One program per net; writes sparse (cell_idx, marker=net_idx+1) into
    per-net buffer slots [ni*MAX_CELLS : (ni+1)*MAX_CELLS]."""
    ni = tl.program_id(0)
    d_pin = tl.load(net_driver_ptr + ni).to(tl.int32)
    s_off = tl.load(net_sinks_off_ptr + ni).to(tl.int32)
    s_end = tl.load(net_sinks_off_ptr + ni + 1).to(tl.int32)
    n_sinks = s_end - s_off

    ni_marker = (ni + 1).to(tl.float32)
    sign_h = ni_marker
    sign_v = ni_marker

    d_r = tl.load(pin_row_ptr + d_pin).to(tl.int32)
    d_c = tl.load(pin_col_ptr + d_pin).to(tl.int32)
    d_gid = d_r * gc + d_c

    h_cursor = 0
    v_cursor = 0

    slot_arange = tl.arange(0, SLOTS)
    init_zero = slot_arange == 0
    NEG_ONE = tl.full((SLOTS,), -1, dtype=tl.int32)
    g_slots = tl.where(init_zero, d_gid, NEG_ONE)
    r_slots = tl.where(init_zero, d_r, tl.zeros((SLOTS,), dtype=tl.int32))
    c_slots = tl.where(init_zero, d_c, tl.zeros((SLOTS,), dtype=tl.int32))
    n_unique = 1

    for si in range(0, MAX_PINS_PER_NET):
        if si < n_sinks:
            p = tl.load(net_sinks_idx_ptr + s_off + si).to(tl.int32)
            p_r = tl.load(pin_row_ptr + p).to(tl.int32)
            p_c = tl.load(pin_col_ptr + p).to(tl.int32)
            p_gid = p_r * gc + p_c
            seen_vec = g_slots == p_gid
            seen = tl.sum(seen_vec.to(tl.int32)) > 0
            is_new = (~seen) & (n_unique < SLOTS)
            is_target = (slot_arange == n_unique) & is_new
            g_slots = tl.where(is_target, p_gid, g_slots)
            r_slots = tl.where(is_target, p_r, r_slots)
            c_slots = tl.where(is_target, p_c, c_slots)
            n_unique = n_unique + tl.where(is_new, 1, 0)

    _zero_int = tl.zeros((SLOTS,), dtype=tl.int32)
    r0 = tl.sum(tl.where(slot_arange == 0, r_slots, _zero_int))
    c0 = tl.sum(tl.where(slot_arange == 0, c_slots, _zero_int))
    r1 = tl.sum(tl.where(slot_arange == 1, r_slots, _zero_int))
    c1 = tl.sum(tl.where(slot_arange == 1, c_slots, _zero_int))
    r2 = tl.sum(tl.where(slot_arange == 2, r_slots, _zero_int))
    c2 = tl.sum(tl.where(slot_arange == 2, c_slots, _zero_int))

    if n_unique == 2:
        h_cursor, v_cursor = route_two_pin(
            H_idx_ptr, H_val_ptr, V_idx_ptr, V_val_ptr, ni,
            h_cursor, v_cursor, d_r, d_c, r1, c1,
            sign_h, sign_v, gc, MAX_CELLS, EMIT_BLK)
    elif n_unique == 3:
        h_cursor, v_cursor = route_three_pin(
            H_idx_ptr, H_val_ptr, V_idx_ptr, V_val_ptr, ni,
            h_cursor, v_cursor, r0, c0, r1, c1, r2, c2,
            sign_h, sign_v, gc, MAX_CELLS, EMIT_BLK)
    elif n_unique >= 4:
        h_cursor, v_cursor = route_star(
            H_idx_ptr, H_val_ptr, V_idx_ptr, V_val_ptr, ni,
            h_cursor, v_cursor, d_r, d_c,
            slot_arange, g_slots, r_slots, c_slots, _zero_int,
            sign_h, sign_v, gc, SLOTS, MAX_CELLS, EMIT_BLK)


def init_route_all_nets(state, *, MAX_CELLS=512, MAX_PINS_PER_NET=24,
                         EMIT_BLK=16, SLOTS=32):
    """Route ALL nets at current state.pin_row/col and add the +1.0
    contributions to state.H_net/V_net in place (caller zeros first if
    needed)."""
    device = state.device
    dtype = state.dtype
    nn = state.nn
    if nn == 0:
        return

    H_idx = torch.zeros(nn * MAX_CELLS, dtype=torch.int32, device=device)
    H_val = torch.zeros(nn * MAX_CELLS, dtype=torch.float32, device=device)
    V_idx = torch.zeros(nn * MAX_CELLS, dtype=torch.int32, device=device)
    V_val = torch.zeros(nn * MAX_CELLS, dtype=torch.float32, device=device)

    grid = (nn,)
    _init_route_kernel[grid](
        state.net_driver, state.net_sinks_off, state.net_sinks_idx,
        state.net_weight,
        state.pin_row, state.pin_col,
        H_idx, H_val, V_idx, V_val,
        state.gc, state.gr,
        MAX_CELLS=MAX_CELLS,
        MAX_PINS_PER_NET=MAX_PINS_PER_NET,
        EMIT_BLK=EMIT_BLK,
        SLOTS=SLOTS,
    )

    # Decode (sign, net_idx) markers and scatter into state.H_net / state.V_net.
    H_idx_2d = H_idx.view(nn, MAX_CELLS).to(torch.int64)
    V_idx_2d = V_idx.view(nn, MAX_CELLS).to(torch.int64)
    H_val_2d = H_val.view(nn, MAX_CELLS)
    V_val_2d = V_val.view(nn, MAX_CELLS)

    h_mask = H_val_2d != 0.0
    v_mask = V_val_2d != 0.0
    h_net_idx = (H_val_2d.abs() - 1.0).to(torch.int64).clamp(min=0)
    h_sign = torch.sign(H_val_2d).to(dtype)
    h_weight = state.net_weight.to(dtype)[h_net_idx]
    h_val_dec = h_sign * (h_weight / float(state.grid_h_routes))
    h_val_dec = torch.where(h_mask, h_val_dec, torch.zeros_like(h_val_dec))

    v_net_idx = (V_val_2d.abs() - 1.0).to(torch.int64).clamp(min=0)
    v_sign = torch.sign(V_val_2d).to(dtype)
    v_weight = state.net_weight.to(dtype)[v_net_idx]
    v_val_dec = v_sign * (v_weight / float(state.grid_v_routes))
    v_val_dec = torch.where(v_mask, v_val_dec, torch.zeros_like(v_val_dec))

    # Flatten (nn, MAX_CELLS) and scatter_add into state.H_net/V_net (ng,).
    state.H_net.scatter_add_(0, H_idx_2d.reshape(-1), h_val_dec.reshape(-1))
    state.V_net.scatter_add_(0, V_idx_2d.reshape(-1), v_val_dec.reshape(-1))
