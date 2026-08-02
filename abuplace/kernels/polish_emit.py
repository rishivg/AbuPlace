"""Fused per-macro candidate emit Triton kernel; assumes caller already removed
m via vec_move_macro_out and emits routing-demand for B placements of that
macro in parallel as sparse (cell, marker) tuples for downstream
scatter+smooth+topk+density."""

import triton
import triton.language as tl

from .routing import route_star, route_three_pin, route_two_pin


# Single-pose polish emit: emits the NEW pose with sign=+1 only. The caller has
# already removed m via vec_move_macro_out, so adding this yields the state
# with m at cand_pos.
@triton.jit
def _polish_emit_kernel(
    probe_macro_ids,           # int32 [B] - usually all same m
    probe_new_x_ptr,           # float64 [B]
    probe_new_y_ptr,           # float64 [B]
    mn_offsets_ptr, mn_net_ids_ptr,
    net_driver_ptr, net_sinks_off_ptr, net_sinks_idx_ptr, net_weight_ptr,
    pin_macro_ptr, pin_x_off_ptr, pin_y_off_ptr,
    pin_row_old_ptr, pin_col_old_ptr,
    H_idx_ptr, H_val_ptr,      # output: (B*MAX_CELLS,) sparse cell idx + marker
    V_idx_ptr, V_val_ptr,
    gw, gh, gc, gr,
    MAX_CELLS: tl.constexpr,
    MAX_NETS_PER_MACRO: tl.constexpr,
    MAX_PINS_PER_NET: tl.constexpr,
    EMIT_BLK: tl.constexpr,
    SLOTS: tl.constexpr,
):
    """One program per (m, candidate); emits (cell, marker=(net_idx+1)*sign)
    tuples into [pid*MAX_CELLS : (pid+1)*MAX_CELLS]; caller decodes marker
    -> sign(marker)*net_weight[abs(marker)-1]/grid_routes so emit stays fp32
    while weights stay fp64."""
    pid = tl.program_id(0)
    m = tl.load(probe_macro_ids + pid).to(tl.int32)
    new_x = tl.load(probe_new_x_ptr + pid)
    new_y = tl.load(probe_new_y_ptr + pid)

    mn_off = tl.load(mn_offsets_ptr + m).to(tl.int32)
    mn_end = tl.load(mn_offsets_ptr + m + 1).to(tl.int32)
    n_iters = mn_end - mn_off

    h_cursor = 0
    v_cursor = 0

    for ki in range(0, MAX_NETS_PER_MACRO):
        if ki < n_iters:
            ni = tl.load(mn_net_ids_ptr + mn_off + ki).to(tl.int32)
            d_pin = tl.load(net_driver_ptr + ni).to(tl.int32)
            s_off = tl.load(net_sinks_off_ptr + ni).to(tl.int32)
            s_end = tl.load(net_sinks_off_ptr + ni + 1).to(tl.int32)
            n_sinks = s_end - s_off

            ni_marker = (ni + 1).to(tl.float32)
            sign_h = ni_marker
            sign_v = ni_marker

            # Driver gcell: new pos if m owns it, else old stored row/col.
            d_macro = tl.load(pin_macro_ptr + d_pin).to(tl.int32)
            d_off_x = tl.load(pin_x_off_ptr + d_pin)
            d_off_y = tl.load(pin_y_off_ptr + d_pin)
            d_old_r = tl.load(pin_row_old_ptr + d_pin).to(tl.int32)
            d_old_c = tl.load(pin_col_old_ptr + d_pin).to(tl.int32)
            d_is_m = d_macro == m

            d_abs_x = new_x + d_off_x
            d_abs_y = new_y + d_off_y
            d_new_c = tl.minimum(tl.maximum(
                (d_abs_x / gw).to(tl.int32), 0), gc - 1)
            d_new_r = tl.minimum(tl.maximum(
                (d_abs_y / gh).to(tl.int32), 0), gr - 1)
            d_r = tl.where(d_is_m, d_new_r, d_old_r)
            d_c = tl.where(d_is_m, d_new_c, d_old_c)
            d_gid = d_r * gc + d_c

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
                    p_macro = tl.load(pin_macro_ptr + p).to(tl.int32)
                    p_off_x = tl.load(pin_x_off_ptr + p)
                    p_off_y = tl.load(pin_y_off_ptr + p)
                    p_old_r = tl.load(pin_row_old_ptr + p).to(tl.int32)
                    p_old_c = tl.load(pin_col_old_ptr + p).to(tl.int32)
                    p_is_m = p_macro == m
                    p_abs_x = new_x + p_off_x
                    p_abs_y = new_y + p_off_y
                    p_new_c = tl.minimum(tl.maximum(
                        (p_abs_x / gw).to(tl.int32), 0), gc - 1)
                    p_new_r = tl.minimum(tl.maximum(
                        (p_abs_y / gh).to(tl.int32), 0), gr - 1)
                    p_r = tl.where(p_is_m, p_new_r, p_old_r)
                    p_c = tl.where(p_is_m, p_new_c, p_old_c)
                    p_gid = p_r * gc + p_c

                    seen_vec = g_slots == p_gid
                    seen = tl.sum(seen_vec.to(tl.int32)) > 0
                    is_new = (~seen) & (n_unique < SLOTS)
                    is_target = (slot_arange == n_unique) & is_new
                    g_slots = tl.where(is_target, p_gid, g_slots)
                    r_slots = tl.where(is_target, p_r, r_slots)
                    c_slots = tl.where(is_target, p_c, c_slots)
                    n_unique = n_unique + tl.where(is_new, 1, 0)

            # Extract slots 0/1/2 for 2-pin and 3-pin Steiner dispatch.
            _zero_int = tl.zeros((SLOTS,), dtype=tl.int32)
            d_r = tl.sum(tl.where(slot_arange == 0, r_slots, _zero_int))
            d_c = tl.sum(tl.where(slot_arange == 0, c_slots, _zero_int))
            r1 = tl.sum(tl.where(slot_arange == 1, r_slots, _zero_int))
            c1 = tl.sum(tl.where(slot_arange == 1, c_slots, _zero_int))
            r2 = tl.sum(tl.where(slot_arange == 2, r_slots, _zero_int))
            c2 = tl.sum(tl.where(slot_arange == 2, c_slots, _zero_int))

            # 2-pin: route_two_pin between driver and the only-other unique gcell.
            r0 = d_r
            c0 = d_c
            if n_unique == 2:
                h_cursor, v_cursor = route_two_pin(
                    H_idx_ptr, H_val_ptr, V_idx_ptr, V_val_ptr, pid,
                    h_cursor, v_cursor, d_r, d_c, r1, c1,
                    sign_h, sign_v, gc, MAX_CELLS, EMIT_BLK)
            elif n_unique == 3:
                h_cursor, v_cursor = route_three_pin(
                    H_idx_ptr, H_val_ptr, V_idx_ptr, V_val_ptr, pid,
                    h_cursor, v_cursor, r0, c0, r1, c1, r2, c2,
                    sign_h, sign_v, gc, MAX_CELLS, EMIT_BLK)
            elif n_unique >= 4:
                h_cursor, v_cursor = route_star(
                    H_idx_ptr, H_val_ptr, V_idx_ptr, V_val_ptr, pid,
                    h_cursor, v_cursor, d_r, d_c,
                    slot_arange, g_slots, r_slots, c_slots, _zero_int,
                    sign_h, sign_v, gc, SLOTS, MAX_CELLS, EMIT_BLK)
            # n_unique == 1: single gcell, nothing to route.

