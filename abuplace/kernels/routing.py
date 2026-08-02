"""The GPU routing model: how a net becomes routing demand on the gcell grid.

`init_route` builds the baseline H/V demand maps; `polish_emit` produces the
deltas applied against them when a macro moves. Both must lay a net down
identically: a delta computed under different rules than its baseline does
not reconcile, and surfaces much later as an unexplained proxy mismatch rather
than a routing bug. One implementation here makes that divergence impossible.

Mirrors C route_net, the scoring oracle the placer optimizes against:

  2 unique gcells   -> route_two_pin    an L: H along the driver row, V down
                                        the sink column
  3 unique gcells   -> route_three_pin  a Steiner tree, dispatched on geometry
  4+ unique gcells  -> route_star       one L from the driver to each other
                                        occupied gcell

All write through `emit_h_span` / `emit_v_span`, which append a run of gcells
into a per-program slice of a flat buffer (`buf_idx * MAX_CELLS + cursor`) and
return the advanced cursor - a net index for `init_route`, a probe id for
`polish_emit`. Writes past `MAX_CELLS` are masked off, so an oversized span
truncates rather than corrupting a neighbouring slice; callers size `MAX_CELLS`
to keep that unreachable.

All `@triton.jit` device functions, inlined into their caller.
"""

import triton
import triton.language as tl


@triton.jit
def emit_h_span(H_idx_ptr, H_val_ptr, buf_idx, h_cursor, row, c_lo, c_hi,
                val, gc, MAX_CELLS: tl.constexpr, BLK: tl.constexpr):
    """Emit gcells [c_lo, c_hi) along `row` at `val`; returns the new cursor."""
    span = c_hi - c_lo
    base = buf_idx * MAX_CELLS
    n_chunks: tl.constexpr = MAX_CELLS // BLK
    val_vec = val + tl.zeros((BLK,), dtype=tl.float32)
    arange = tl.arange(0, BLK)
    for chunk in range(0, n_chunks):
        span_offs = chunk * BLK + arange
        mask = (span_offs < span) & ((h_cursor + arange) < MAX_CELLS)
        cells = (row * gc + (c_lo + span_offs)).to(tl.int32)
        sptrs = base + h_cursor + arange
        tl.store(H_idx_ptr + sptrs, cells, mask=mask)
        tl.store(H_val_ptr + sptrs, val_vec, mask=mask)
        h_cursor = h_cursor + tl.sum(mask.to(tl.int32))
    return h_cursor


@triton.jit
def emit_v_span(V_idx_ptr, V_val_ptr, buf_idx, v_cursor, col, r_lo, r_hi,
                val, gc, MAX_CELLS: tl.constexpr, BLK: tl.constexpr):
    """Emit gcells [r_lo, r_hi) along `col` at `val`; returns the new cursor."""
    span = r_hi - r_lo
    base = buf_idx * MAX_CELLS
    n_chunks: tl.constexpr = MAX_CELLS // BLK
    val_vec = val + tl.zeros((BLK,), dtype=tl.float32)
    arange = tl.arange(0, BLK)
    for chunk in range(0, n_chunks):
        span_offs = chunk * BLK + arange
        mask = (span_offs < span) & ((v_cursor + arange) < MAX_CELLS)
        cells = ((r_lo + span_offs) * gc + col).to(tl.int32)
        sptrs = base + v_cursor + arange
        tl.store(V_idx_ptr + sptrs, cells, mask=mask)
        tl.store(V_val_ptr + sptrs, val_vec, mask=mask)
        v_cursor = v_cursor + tl.sum(mask.to(tl.int32))
    return v_cursor


@triton.jit
def route_three_pin(H_idx_ptr, H_val_ptr, V_idx_ptr, V_val_ptr, buf_idx,
                    h_cursor, v_cursor, r0, c0, r1, c1, r2, c2,
                    sign_h, sign_v, gc,
                    MAX_CELLS: tl.constexpr, EMIT_BLK: tl.constexpr):
    """Route a 3-pin net over its three unique gcells; returns the advanced
    (h_cursor, v_cursor).

    Sorts the pins by (col, row) with three conditional swaps, then dispatches
    on geometry: A = both corners bend, B = a T-junction on a shared column,
    C = a shared row, else a route_t fallback that re-sorts by (row, col).
    Mirrors C route_three_pin exactly - the C engine is the scoring oracle, so
    any divergence here makes the GPU proxy disagree with the number the
    placer actually optimizes."""
    rA = r0; cA = c0
    rB = r1; cB = c1
    rC = r2; cC = c2
    AB_swap = (cA > cB) | ((cA == cB) & (rA > rB))
    rA_n = tl.where(AB_swap, rB, rA); cA_n = tl.where(AB_swap, cB, cA)
    rB_n = tl.where(AB_swap, rA, rB); cB_n = tl.where(AB_swap, cA, cB)
    rA = rA_n; cA = cA_n; rB = rB_n; cB = cB_n
    BC_swap = (cB > cC) | ((cB == cC) & (rB > rC))
    rB_n = tl.where(BC_swap, rC, rB); cB_n = tl.where(BC_swap, cC, cB)
    rC_n = tl.where(BC_swap, rB, rC); cC_n = tl.where(BC_swap, cB, cC)
    rB = rB_n; cB = cB_n; rC = rC_n; cC = cC_n
    AB_swap = (cA > cB) | ((cA == cB) & (rA > rB))
    rA_n = tl.where(AB_swap, rB, rA); cA_n = tl.where(AB_swap, cB, cA)
    rB_n = tl.where(AB_swap, rA, rB); cB_n = tl.where(AB_swap, cA, cB)
    rA = rA_n; cA = cA_n; rB = rB_n; cB = cB_n

    y1 = rA; x1 = cA
    y2 = rB; x2 = cB
    y3 = rC; x3 = cC

    miny13 = tl.minimum(y1, y3)
    maxy13 = tl.maximum(y1, y3)

    case_A = (x1 < x2) & (x2 < x3) & (miny13 < y2) & (maxy13 > y2)
    case_B = (x2 == x3) & (x1 < x2) & (y1 < tl.minimum(y2, y3))
    case_C = y2 == y3

    if case_A:
        h_cursor = emit_h_span(
            H_idx_ptr, H_val_ptr, buf_idx, h_cursor,
            y1, x1, x2, sign_h, gc, MAX_CELLS, EMIT_BLK)
        h_cursor = emit_h_span(
            H_idx_ptr, H_val_ptr, buf_idx, h_cursor,
            y2, x2, x3, sign_h, gc, MAX_CELLS, EMIT_BLK)
        r1lo = tl.minimum(y1, y2); r1hi = tl.maximum(y1, y2)
        v_cursor = emit_v_span(
            V_idx_ptr, V_val_ptr, buf_idx, v_cursor,
            x2, r1lo, r1hi, sign_v, gc, MAX_CELLS, EMIT_BLK)
        r2lo = tl.minimum(y2, y3); r2hi = tl.maximum(y2, y3)
        v_cursor = emit_v_span(
            V_idx_ptr, V_val_ptr, buf_idx, v_cursor,
            x3, r2lo, r2hi, sign_v, gc, MAX_CELLS, EMIT_BLK)
    elif case_B:
        h_cursor = emit_h_span(
            H_idx_ptr, H_val_ptr, buf_idx, h_cursor,
            y1, x1, x2, sign_h, gc, MAX_CELLS, EMIT_BLK)
        row_hi = tl.maximum(y2, y3)
        v_cursor = emit_v_span(
            V_idx_ptr, V_val_ptr, buf_idx, v_cursor,
            x2, y1, row_hi, sign_v, gc, MAX_CELLS, EMIT_BLK)
    elif case_C:
        h_cursor = emit_h_span(
            H_idx_ptr, H_val_ptr, buf_idx, h_cursor,
            y1, x1, x2, sign_h, gc, MAX_CELLS, EMIT_BLK)
        h_cursor = emit_h_span(
            H_idx_ptr, H_val_ptr, buf_idx, h_cursor,
            y2, x2, x3, sign_h, gc, MAX_CELLS, EMIT_BLK)
        rlo = tl.minimum(y1, y2); rhi = tl.maximum(y1, y2)
        v_cursor = emit_v_span(
            V_idx_ptr, V_val_ptr, buf_idx, v_cursor,
            x2, rlo, rhi, sign_v, gc, MAX_CELLS, EMIT_BLK)
    else:
        # route_t fallback: re-sort pins by (row, col).
        rA = r0; cA = c0
        rB = r1; cB = c1
        rC = r2; cC = c2
        AB_swap = (rA > rB) | ((rA == rB) & (cA > cB))
        rA_n = tl.where(AB_swap, rB, rA); cA_n = tl.where(AB_swap, cB, cA)
        rB_n = tl.where(AB_swap, rA, rB); cB_n = tl.where(AB_swap, cA, cB)
        rA = rA_n; cA = cA_n; rB = rB_n; cB = cB_n
        BC_swap = (rB > rC) | ((rB == rC) & (cB > cC))
        rB_n = tl.where(BC_swap, rC, rB); cB_n = tl.where(BC_swap, cC, cB)
        rC_n = tl.where(BC_swap, rB, rC); cC_n = tl.where(BC_swap, cB, cC)
        rB = rB_n; cB = cB_n; rC = rC_n; cC = cC_n
        AB_swap = (rA > rB) | ((rA == rB) & (cA > cB))
        rA_n = tl.where(AB_swap, rB, rA); cA_n = tl.where(AB_swap, cB, cA)
        rB_n = tl.where(AB_swap, rA, rB); cB_n = tl.where(AB_swap, cA, cB)
        rA = rA_n; cA = cA_n; rB = rB_n; cB = cB_n

        ty1 = rA; tx1 = cA
        ty2 = rB; tx2 = cB
        ty3 = rC; tx3 = cC
        xmin = tl.minimum(tl.minimum(tx1, tx2), tx3)
        xmax = tl.maximum(tl.maximum(tx1, tx2), tx3)
        h_cursor = emit_h_span(
            H_idx_ptr, H_val_ptr, buf_idx, h_cursor,
            ty2, xmin, xmax, sign_h, gc, MAX_CELLS, EMIT_BLK)
        r1lo = tl.minimum(ty1, ty2); r1hi = tl.maximum(ty1, ty2)
        v_cursor = emit_v_span(
            V_idx_ptr, V_val_ptr, buf_idx, v_cursor,
            tx1, r1lo, r1hi, sign_v, gc, MAX_CELLS, EMIT_BLK)
        r2lo = tl.minimum(ty2, ty3); r2hi = tl.maximum(ty2, ty3)
        v_cursor = emit_v_span(
            V_idx_ptr, V_val_ptr, buf_idx, v_cursor,
            tx3, r2lo, r2hi, sign_v, gc, MAX_CELLS, EMIT_BLK)
    return h_cursor, v_cursor


@triton.jit
def route_two_pin(H_idx_ptr, H_val_ptr, V_idx_ptr, V_val_ptr, buf_idx,
                  h_cursor, v_cursor, d_r, d_c, r1, c1,
                  sign_h, sign_v, gc,
                  MAX_CELLS: tl.constexpr, EMIT_BLK: tl.constexpr):
    """Route a 2-pin net as an L: one H run along the driver row, one V run
    down the sink column. Returns the advanced (h_cursor, v_cursor)."""
    snk_r = r1
    snk_c = c1
    row_min = tl.minimum(d_r, snk_r)
    row_max = tl.maximum(d_r, snk_r)
    col_min = tl.minimum(d_c, snk_c)
    col_max = tl.maximum(d_c, snk_c)
    h_cursor = emit_h_span(
        H_idx_ptr, H_val_ptr, buf_idx, h_cursor,
        d_r, col_min, col_max, sign_h,
        gc, MAX_CELLS, EMIT_BLK)
    v_cursor = emit_v_span(
        V_idx_ptr, V_val_ptr, buf_idx, v_cursor,
        snk_c, row_min, row_max, sign_v,
        gc, MAX_CELLS, EMIT_BLK)
    return h_cursor, v_cursor


@triton.jit
def route_star(H_idx_ptr, H_val_ptr, V_idx_ptr, V_val_ptr, buf_idx,
               h_cursor, v_cursor, d_r, d_c,
               slot_arange, g_slots, r_slots, c_slots, _zero_int,
               sign_h, sign_v, gc,
               SLOTS: tl.constexpr, MAX_CELLS: tl.constexpr,
               EMIT_BLK: tl.constexpr):
    """Route a net of 4+ unique gcells as a star: one L from the driver to
    each other occupied slot. Slots with gk < 0 are empty and skipped.
    Returns the advanced (h_cursor, v_cursor)."""
    for slot in range(1, SLOTS):
        gk = tl.sum(tl.where(slot_arange == slot, g_slots, _zero_int))
        rk = tl.sum(tl.where(slot_arange == slot, r_slots, _zero_int))
        ck = tl.sum(tl.where(slot_arange == slot, c_slots, _zero_int))
        if gk >= 0:
            row_min = tl.minimum(d_r, rk)
            row_max = tl.maximum(d_r, rk)
            col_min = tl.minimum(d_c, ck)
            col_max = tl.maximum(d_c, ck)
            h_cursor = emit_h_span(
                H_idx_ptr, H_val_ptr, buf_idx, h_cursor,
                d_r, col_min, col_max, sign_h,
                gc, MAX_CELLS, EMIT_BLK)
            v_cursor = emit_v_span(
                V_idx_ptr, V_val_ptr, buf_idx, v_cursor,
                ck, row_min, row_max, sign_v,
                gc, MAX_CELLS, EMIT_BLK)
    return h_cursor, v_cursor
