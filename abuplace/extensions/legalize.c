/* Faster spiral-search macro legalization with spatial-grid overlap queries.
 *
 * Improvements over the original:
 *   1. Perimeter-only ring traversal, preserving original candidate order.
 *   2. Duplicate clamped-candidate suppression within each ring.
 *   3. Reciprocal-based grid indexing (mul instead of div in hot path).
 *   4. restrict qualifiers for better compiler optimization.
 *   5. Precomputed min_dim[] for step sizing.
 *   6. memset() initialization for heads[].
 *   7. Reusable scratch buffers across calls (not thread-safe).
 *
 * Behavior:
 *   - Same sort-by-area order.
 *   - Same spiral ring candidate order as the original filtered nested loops.
 *   - Same canvas clamping.
 *   - Same overlap tolerance.
 *
 * Built automatically by placer.py::_ensure_lib on first use, and rebuilt
 * whenever this source is newer than the .so. Compiler flags live in
 * _C_EXTENSIONS there.
 */

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define EPS 0.01
#define MAX_RING 200
#define RING_SEEN_CAP (8 * MAX_RING)

static const double *g_sort_areas = NULL;

/* ------------------------------------------------------------------------- */
/* Reusable scratch buffers (process-global, not thread-safe)                */
/* ------------------------------------------------------------------------- */

typedef struct {
    int    *order;
    int    *heads;
    int    *next_arr;
    double *areas;
    double *min_dim;
    int    *macro_cell;     /* legalize_v2: which cell each macro currently lives in */
    double *targets;        /* legalize_v2: original target positions (2*nh) */
    int    *disp_order;     /* legalize_v2: ordering by displacement-desc (nh) */
    double *disp2;          /* legalize_v2: per-macro squared displacement (nh) */
    int     cap_nh;
    int     cap_ncells;
} legalize_scratch_t;

static legalize_scratch_t g_scratch = {0};

static int ensure_scratch(int nh, int ncells) {
    if (nh > g_scratch.cap_nh) {
        int *new_order      = (int *)realloc(g_scratch.order,    (size_t)nh * sizeof(int));
        int *new_next_arr   = (int *)realloc(g_scratch.next_arr, (size_t)nh * sizeof(int));
        double *new_areas   = (double *)realloc(g_scratch.areas,   (size_t)nh * sizeof(double));
        double *new_min_dim = (double *)realloc(g_scratch.min_dim, (size_t)nh * sizeof(double));
        int *new_mac_cell   = (int *)realloc(g_scratch.macro_cell, (size_t)nh * sizeof(int));
        double *new_targets = (double *)realloc(g_scratch.targets, (size_t)2 * nh * sizeof(double));
        int *new_disp_order = (int *)realloc(g_scratch.disp_order, (size_t)nh * sizeof(int));
        double *new_disp2   = (double *)realloc(g_scratch.disp2,   (size_t)nh * sizeof(double));
        if (!new_order || !new_next_arr || !new_areas || !new_min_dim
                || !new_mac_cell || !new_targets || !new_disp_order || !new_disp2) {
            free(new_order); free(new_next_arr); free(new_areas); free(new_min_dim);
            free(new_mac_cell); free(new_targets); free(new_disp_order); free(new_disp2);
            return 0;
        }
        g_scratch.order = new_order;
        g_scratch.next_arr = new_next_arr;
        g_scratch.areas = new_areas;
        g_scratch.min_dim = new_min_dim;
        g_scratch.macro_cell = new_mac_cell;
        g_scratch.targets = new_targets;
        g_scratch.disp_order = new_disp_order;
        g_scratch.disp2 = new_disp2;
        g_scratch.cap_nh = nh;
    }

    if (ncells > g_scratch.cap_ncells) {
        int *new_heads = (int *)realloc(g_scratch.heads, (size_t)ncells * sizeof(int));
        if (!new_heads) {
            free(new_heads);
            return 0;
        }
        g_scratch.heads = new_heads;
        g_scratch.cap_ncells = ncells;
    }

    return 1;
}

/* ------------------------------------------------------------------------- */
/* Sorting helpers                                                           */
/* ------------------------------------------------------------------------- */

static int cmp_by_area(const void *a, const void *b) {
    const int ia = *(const int *)a;
    const int ib = *(const int *)b;
    const double da = g_sort_areas[ia];
    const double db = g_sort_areas[ib];
    if (da < db) return -1;
    if (da > db) return 1;
    return ia - ib;  /* stable tiebreak on original index */
}

/* ------------------------------------------------------------------------- */
/* Grid helpers                                                              */
/* ------------------------------------------------------------------------- */

static inline int grid_cell(double v, double inv_cell_size, int nc) {
    int c = (int)(v * inv_cell_size);
    if (c < 0) return 0;
    if (c >= nc) return nc - 1;
    return c;
}

static inline int has_overlap(
    double cx,
    double cy,
    double self_hw,
    double self_hh,
    int self_idx,
    const double *restrict pos,
    const double *restrict sizes,
    const int *restrict heads,
    const int *restrict next_arr,
    int ncx,
    int ncy,
    double inv_cell_size
) {
    const int gx = grid_cell(cx, inv_cell_size, ncx);
    const int gy = grid_cell(cy, inv_cell_size, ncy);

    const int x0 = (gx > 0) ? (gx - 1) : 0;
    const int x1 = (gx + 1 < ncx) ? (gx + 1) : (ncx - 1);
    const int y0 = (gy > 0) ? (gy - 1) : 0;
    const int y1 = (gy + 1 < ncy) ? (gy + 1) : (ncy - 1);

    for (int ny = y0; ny <= y1; ++ny) {
        const int row_base = ny * ncx;
        for (int nx = x0; nx <= x1; ++nx) {
            for (int j = heads[row_base + nx]; j != -1; j = next_arr[j]) {
                if (j == self_idx) continue;

                const double sep_x = self_hw + 0.5 * sizes[2 * j] + EPS;
                const double sep_y = self_hh + 0.5 * sizes[2 * j + 1] + EPS;

                double ddx = cx - pos[2 * j];
                if (ddx < 0.0) ddx = -ddx;

                double ddy = cy - pos[2 * j + 1];
                if (ddy < 0.0) ddy = -ddy;

                if (ddx < sep_x && ddy < sep_y) {
                    return 1;
                }
            }
        }
    }

    return 0;
}

static inline int seen_candidate(
    double cx,
    double cy,
    const double *restrict seen_x,
    const double *restrict seen_y,
    int seen_n
) {
    for (int i = 0; i < seen_n; ++i) {
        if (seen_x[i] == cx && seen_y[i] == cy) {
            return 1;
        }
    }
    return 0;
}

static inline void try_candidate(
    int dxm,
    int dym,
    double tx,
    double ty,
    double step,
    double lo_x,
    double hi_x,
    double lo_y,
    double hi_y,
    double self_hw,
    double self_hh,
    int self_idx,
    const double *restrict pos,
    const double *restrict sizes,
    const int *restrict heads,
    const int *restrict next_arr,
    int ncx,
    int ncy,
    double inv_cell_size,
    double *restrict best_d,
    double *restrict bx,
    double *restrict by,
    int *restrict found,
    double *restrict seen_x,
    double *restrict seen_y,
    int *restrict seen_n
) {
    double cxp = tx + (double)dxm * step;
    double cyp = ty + (double)dym * step;

    if (cxp < lo_x) cxp = lo_x;
    else if (cxp > hi_x) cxp = hi_x;

    if (cyp < lo_y) cyp = lo_y;
    else if (cyp > hi_y) cyp = hi_y;

    if (seen_candidate(cxp, cyp, seen_x, seen_y, *seen_n)) {
        return;
    }

    seen_x[*seen_n] = cxp;
    seen_y[*seen_n] = cyp;
    ++(*seen_n);

    if (!has_overlap(
            cxp, cyp, self_hw, self_hh, self_idx,
            pos, sizes, heads, next_arr,
            ncx, ncy, inv_cell_size)) {

        const double ddx = cxp - tx;
        const double ddy = cyp - ty;
        const double d = ddx * ddx + ddy * ddy;

        if (d < *best_d) {
            *best_d = d;
            *bx = cxp;
            *by = cyp;
            *found = 1;
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Main legalization                                                         */
/* ------------------------------------------------------------------------- */

void legalize(
    double *restrict pos,          /* [nh, 2] row-major, in/out */
    int nh,
    const double *restrict sizes,  /* [nh, 2] */
    const int *restrict movable,   /* [nh] nonzero = movable */
    double cw,
    double ch,
    const double *restrict hw,     /* [nh] half-widths */
    const double *restrict hh,     /* [nh] half-heights */
    double step_factor,
    int *restrict had_overlap      /* out: 1 if any movable failed to place, else 0 */
) {
    if (had_overlap) *had_overlap = 0;
    if (nh <= 0) return;

    /* Build order, areas, min_dim, and max_dim in one pass. */
    double max_dim = 0.0;
    for (int i = 0; i < nh; ++i) {
        const double sx = sizes[2 * i];
        const double sy = sizes[2 * i + 1];
        const double area = sx * sy;

        if (sx > max_dim) max_dim = sx;
        if (sy > max_dim) max_dim = sy;
        if (sx < sy) g_scratch.min_dim ? (void)0 : (void)0; /* keep compiler quiet before ensure */
    }

    double cell_size = max_dim + EPS;
    if (cell_size < 1e-6) cell_size = 1.0;
    const double inv_cell_size = 1.0 / cell_size;

    int ncx = (int)(cw * inv_cell_size) + 1;
    int ncy = (int)(ch * inv_cell_size) + 1;
    if (ncx < 1) ncx = 1;
    if (ncy < 1) ncy = 1;
    const int ncells = ncx * ncy;

    if (!ensure_scratch(nh, ncells)) {
        return;
    }

    int *restrict order = g_scratch.order;
    int *restrict heads = g_scratch.heads;
    int *restrict next_arr = g_scratch.next_arr;
    double *restrict areas = g_scratch.areas;
    double *restrict min_dim = g_scratch.min_dim;

    for (int i = 0; i < nh; ++i) {
        const double sx = sizes[2 * i];
        const double sy = sizes[2 * i + 1];
        order[i] = i;
        areas[i] = -(sx * sy);
        min_dim[i] = (sx < sy) ? sx : sy;
    }

    g_sort_areas = areas;
    qsort(order, (size_t)nh, sizeof(int), cmp_by_area);
    g_sort_areas = NULL;

    memset(heads, 0xFF, (size_t)ncells * sizeof(int));

    for (int oi = 0; oi < nh; ++oi) {
        const int idx = order[oi];

        const double self_hw = hw[idx];
        const double self_hh = hh[idx];

        const double tx = pos[2 * idx];
        const double ty = pos[2 * idx + 1];

        if (movable[idx]) {
            if (has_overlap(
                    tx, ty, self_hw, self_hh, idx,
                    pos, sizes, heads, next_arr,
                    ncx, ncy, inv_cell_size)) {

                const double step = min_dim[idx] * step_factor;
                double best_d = 1e30;
                double bx = tx;
                double by = ty;
                int found = 0;

                const double lo_x = self_hw;
                const double hi_x = cw - self_hw;
                const double lo_y = self_hh;
                const double hi_y = ch - self_hh;

                for (int ring = 1; ring < MAX_RING; ++ring) {
                    double seen_x[RING_SEEN_CAP];
                    double seen_y[RING_SEEN_CAP];
                    int seen_n = 0;

                    /* Preserve original candidate order:
                     * for dx = -ring..ring:
                     *   for dy = -ring..ring:
                     *     if perimeter -> test
                     */
                    for (int dxm = -ring; dxm <= ring; ++dxm) {
                        if (dxm == -ring || dxm == ring) {
                            for (int dym = -ring; dym <= ring; ++dym) {
                                try_candidate(
                                    dxm, dym,
                                    tx, ty, step,
                                    lo_x, hi_x, lo_y, hi_y,
                                    self_hw, self_hh, idx,
                                    pos, sizes, heads, next_arr,
                                    ncx, ncy, inv_cell_size,
                                    &best_d, &bx, &by, &found,
                                    seen_x, seen_y, &seen_n
                                );
                            }
                        } else {
                            try_candidate(
                                dxm, -ring,
                                tx, ty, step,
                                lo_x, hi_x, lo_y, hi_y,
                                self_hw, self_hh, idx,
                                pos, sizes, heads, next_arr,
                                ncx, ncy, inv_cell_size,
                                &best_d, &bx, &by, &found,
                                seen_x, seen_y, &seen_n
                            );

                            try_candidate(
                                dxm, ring,
                                tx, ty, step,
                                lo_x, hi_x, lo_y, hi_y,
                                self_hw, self_hh, idx,
                                pos, sizes, heads, next_arr,
                                ncx, ncy, inv_cell_size,
                                &best_d, &bx, &by, &found,
                                seen_x, seen_y, &seen_n
                            );
                        }
                    }

                    if (found) break;
                }

                if (!found && had_overlap) *had_overlap = 1;

                pos[2 * idx] = bx;
                pos[2 * idx + 1] = by;
            }
        }

        /* Insert into grid at final placed position. */
        const int gx = grid_cell(pos[2 * idx],     inv_cell_size, ncx);
        const int gy = grid_cell(pos[2 * idx + 1], inv_cell_size, ncy);
        const int c = gy * ncx + gx;

        next_arr[idx] = heads[c];
        heads[c] = idx;
        if (g_scratch.macro_cell) g_scratch.macro_cell[idx] = c;
    }
}

/* ------------------------------------------------------------------------- */
/* legalize_v2: spiral + min-displacement post-pass + same-size swap         */
/* ------------------------------------------------------------------------- */

/* Remove `idx` from its current cell's linked list. O(bucket size). */
static inline void grid_remove(int idx, int *restrict heads, int *restrict next_arr,
                                 int *restrict macro_cell) {
    int c = macro_cell[idx];
    if (c < 0) return;
    if (heads[c] == idx) {
        heads[c] = next_arr[idx];
    } else {
        int prev = heads[c];
        while (prev != -1 && next_arr[prev] != idx) prev = next_arr[prev];
        if (prev != -1) next_arr[prev] = next_arr[idx];
    }
    next_arr[idx] = -1;
    macro_cell[idx] = -1;
}

/* Insert `idx` into cell c. */
static inline void grid_insert(int idx, int c, int *restrict heads,
                                int *restrict next_arr, int *restrict macro_cell) {
    next_arr[idx] = heads[c];
    heads[c] = idx;
    macro_cell[idx] = c;
}

/* Comparator for displacement-desc ordering. */
static const double *g_sort_disp2 = NULL;
static int cmp_by_disp2_desc(const void *a, const void *b) {
    const int ia = *(const int *)a;
    const int ib = *(const int *)b;
    const double da = g_sort_disp2[ia];
    const double db = g_sort_disp2[ib];
    if (da > db) return -1;
    if (da < db) return 1;
    return ia - ib;
}

/* Post-pass: pull each macro toward its target via bisection. Returns
 * number of commits this pass.
 */
static int postpass_pull(
    double *restrict pos,
    int nh,
    const double *restrict targets,
    const double *restrict sizes,
    const int *restrict movable,
    const double *restrict hw,
    const double *restrict hh,
    double cw, double ch,
    int *restrict heads,
    int *restrict next_arr,
    int *restrict macro_cell,
    int ncx, int ncy,
    double inv_cell_size,
    int *restrict disp_order,
    double *restrict disp2
) {
    /* Compute displacement squared. */
    for (int i = 0; i < nh; ++i) {
        const double dx = pos[2*i] - targets[2*i];
        const double dy = pos[2*i+1] - targets[2*i+1];
        disp2[i] = dx*dx + dy*dy;
        disp_order[i] = i;
    }
    g_sort_disp2 = disp2;
    qsort(disp_order, (size_t)nh, sizeof(int), cmp_by_disp2_desc);
    g_sort_disp2 = NULL;

    static const double fracs[5] = {1.0, 0.5, 0.25, 0.125, 0.0625};
    int commits = 0;

    for (int oi = 0; oi < nh; ++oi) {
        const int idx = disp_order[oi];
        if (!movable[idx]) continue;
        if (disp2[idx] < 1e-18) break;  /* sorted desc; rest are at-target */

        const double cx = pos[2*idx];
        const double cy = pos[2*idx+1];
        const double tx = targets[2*idx];
        const double ty = targets[2*idx+1];
        const double dvx = tx - cx;
        const double dvy = ty - cy;
        const double self_hw = hw[idx];
        const double self_hh = hh[idx];

        /* Remove self from grid so overlap check excludes it cleanly. */
        const int old_c = macro_cell[idx];
        grid_remove(idx, heads, next_arr, macro_cell);

        int committed = 0;
        for (int fi = 0; fi < 5; ++fi) {
            double nx = cx + fracs[fi] * dvx;
            double ny = cy + fracs[fi] * dvy;
            /* Canvas clamp. */
            if (nx < self_hw) nx = self_hw;
            else if (nx > cw - self_hw) nx = cw - self_hw;
            if (ny < self_hh) ny = self_hh;
            else if (ny > ch - self_hh) ny = ch - self_hh;

            /* After clamp must still be a strict improvement. */
            const double new_d2 = (nx - tx) * (nx - tx) + (ny - ty) * (ny - ty);
            if (new_d2 >= disp2[idx] - 1e-18) continue;

            if (!has_overlap(nx, ny, self_hw, self_hh, idx,
                              pos, sizes, heads, next_arr,
                              ncx, ncy, inv_cell_size)) {
                pos[2*idx] = nx;
                pos[2*idx+1] = ny;
                disp2[idx] = new_d2;
                const int new_c = grid_cell(nx, inv_cell_size, ncx)
                                  + grid_cell(ny, inv_cell_size, ncy) * ncx;
                grid_insert(idx, new_c, heads, next_arr, macro_cell);
                committed = 1;
                ++commits;
                break;
            }
        }
        if (!committed) {
            /* Re-insert at original cell. */
            grid_insert(idx, old_c, heads, next_arr, macro_cell);
        }
    }
    return commits;
}

/* Post-pass: same-size swap. For each highly-displaced movable macro i,
 * scan up to swap_top_k same-size neighbors j; swap if total displacement
 * decreases. Same-size guarantees swap is overlap-feasible.
 *
 * Returns number of swap commits.
 */
static int postpass_swap(
    double *restrict pos,
    int nh,
    const double *restrict targets,
    const double *restrict sizes,
    const int *restrict movable,
    int *restrict heads,
    int *restrict next_arr,
    int *restrict macro_cell,
    int ncx, int ncy,
    double inv_cell_size,
    int *restrict disp_order,
    double *restrict disp2,
    int swap_top_k
) {
    /* Compute disp2 + sort desc. */
    for (int i = 0; i < nh; ++i) {
        const double dx = pos[2*i] - targets[2*i];
        const double dy = pos[2*i+1] - targets[2*i+1];
        disp2[i] = dx*dx + dy*dy;
        disp_order[i] = i;
    }
    g_sort_disp2 = disp2;
    qsort(disp_order, (size_t)nh, sizeof(int), cmp_by_disp2_desc);
    g_sort_disp2 = NULL;

    int commits = 0;

    /* For each macro by displacement, scan in expanding cell rings for
     * same-size partners. Bounded by swap_top_k candidates examined.
     */
    for (int oi = 0; oi < nh; ++oi) {
        const int i = disp_order[oi];
        if (!movable[i]) continue;
        if (disp2[i] < 1e-18) break;

        const double sx_i = sizes[2*i];
        const double sy_i = sizes[2*i+1];
        const double pix = pos[2*i];
        const double piy = pos[2*i+1];
        const double tix = targets[2*i];
        const double tiy = targets[2*i+1];

        const int gx = grid_cell(pix, inv_cell_size, ncx);
        const int gy = grid_cell(piy, inv_cell_size, ncy);

        int examined = 0;
        int best_j = -1;
        double best_gain = 1e-18;  /* must strictly reduce */

        /* Expand ring radius until we've examined swap_top_k candidates
         * or hit a hard cap of 8 rings. */
        for (int r = 0; r <= 8 && examined < swap_top_k; ++r) {
            const int x0 = (gx - r > 0) ? (gx - r) : 0;
            const int x1 = (gx + r < ncx) ? (gx + r) : (ncx - 1);
            const int y0 = (gy - r > 0) ? (gy - r) : 0;
            const int y1 = (gy + r < ncy) ? (gy + r) : (ncy - 1);
            for (int cy = y0; cy <= y1; ++cy) {
                /* Only ring perimeter at radius r (skip interior, already done) */
                for (int cx = x0; cx <= x1; ++cx) {
                    if (r > 0 && cy != y0 && cy != y1 && cx != x0 && cx != x1) continue;
                    const int cell = cy * ncx + cx;
                    for (int j = heads[cell]; j != -1; j = next_arr[j]) {
                        if (j == i || !movable[j]) continue;
                        if (sizes[2*j] != sx_i || sizes[2*j+1] != sy_i) continue;
                        ++examined;
                        if (examined > swap_top_k) goto eval_done;

                        const double pjx = pos[2*j];
                        const double pjy = pos[2*j+1];
                        const double tjx = targets[2*j];
                        const double tjy = targets[2*j+1];

                        /* Cost before: (pi - ti)² + (pj - tj)². */
                        const double cost_no = disp2[i]
                            + (pjx - tjx) * (pjx - tjx)
                            + (pjy - tjy) * (pjy - tjy);
                        /* Cost after: i ends at pj (so disp = ‖pj - ti‖²),
                         * j ends at pi (disp = ‖pi - tj‖²). */
                        const double cost_sw =
                            (pjx - tix) * (pjx - tix) + (pjy - tiy) * (pjy - tiy)
                          + (pix - tjx) * (pix - tjx) + (piy - tjy) * (piy - tjy);
                        const double gain = cost_no - cost_sw;
                        if (gain > best_gain) {
                            best_gain = gain;
                            best_j = j;
                        }
                    }
                }
            }
        }
        eval_done: ;

        if (best_j >= 0) {
            /* Commit swap: pos[i] <-> pos[j]; update grid + disp2. */
            const int j = best_j;
            const double pjx = pos[2*j];
            const double pjy = pos[2*j+1];

            grid_remove(i, heads, next_arr, macro_cell);
            grid_remove(j, heads, next_arr, macro_cell);

            pos[2*i]   = pjx;
            pos[2*i+1] = pjy;
            pos[2*j]   = pix;
            pos[2*j+1] = piy;

            const int ic = grid_cell(pjx, inv_cell_size, ncx)
                         + grid_cell(pjy, inv_cell_size, ncy) * ncx;
            const int jc = grid_cell(pix, inv_cell_size, ncx)
                         + grid_cell(piy, inv_cell_size, ncy) * ncx;
            grid_insert(i, ic, heads, next_arr, macro_cell);
            grid_insert(j, jc, heads, next_arr, macro_cell);

            disp2[i] = (pjx - tix) * (pjx - tix) + (pjy - tiy) * (pjy - tiy);
            disp2[j] = (pix - targets[2*j]) * (pix - targets[2*j])
                      + (piy - targets[2*j+1]) * (piy - targets[2*j+1]);
            ++commits;
        }
    }
    return commits;
}

/* Public entry: spiral + post-pass.
 *
 * Args mirror legalize() with these additions:
 *   post_passes:  max pull-toward-target passes (0 = skip post-pass)
 *   swap_passes:  max swap passes (0 = skip swap rescue)
 *   swap_top_k:   per-anchor candidate cap during swap (default 20)
 *
 * The post-pass is a STRICT improvement: each commit reduces displacement.
 * Output is byte-identical to legalize() if post_passes=0 and swap_passes=0.
 */
void legalize_v2(
    double *restrict pos,
    int nh,
    const double *restrict sizes,
    const int *restrict movable,
    double cw,
    double ch,
    const double *restrict hw,
    const double *restrict hh,
    double step_factor,
    int post_passes,
    int swap_passes,
    int swap_top_k,
    int *restrict had_overlap
) {
    /* Phase 1: stash targets BEFORE spiral mutates pos. We need scratch
     * sized first; ensure_scratch is called inside legalize() so we
     * pre-call here just to prep buffers. */
    if (had_overlap) *had_overlap = 0;
    if (nh <= 0) return;

    /* Compute grid params (same logic as legalize). */
    double max_dim = 0.0;
    for (int i = 0; i < nh; ++i) {
        const double sx = sizes[2*i];
        const double sy = sizes[2*i+1];
        if (sx > max_dim) max_dim = sx;
        if (sy > max_dim) max_dim = sy;
    }
    double cell_size = max_dim + EPS;
    if (cell_size < 1e-6) cell_size = 1.0;
    const double inv_cell_size = 1.0 / cell_size;
    int ncx = (int)(cw * inv_cell_size) + 1;
    int ncy = (int)(ch * inv_cell_size) + 1;
    if (ncx < 1) ncx = 1;
    if (ncy < 1) ncy = 1;
    const int ncells = ncx * ncy;
    if (!ensure_scratch(nh, ncells)) return;

    /* Save targets. */
    memcpy(g_scratch.targets, pos, (size_t)2 * nh * sizeof(double));

    /* Initialize macro_cell to -1 (will be filled by spiral). */
    for (int i = 0; i < nh; ++i) g_scratch.macro_cell[i] = -1;

    /* Phase 1: run existing spiral. It populates macro_cell on insert. */
    legalize(pos, nh, sizes, movable, cw, ch, hw, hh, step_factor, had_overlap);
    if (had_overlap && *had_overlap) {
        /* Could not fully legalize - bail post-pass; pos is what spiral got. */
        return;
    }
    if (post_passes <= 0 && swap_passes <= 0) return;

    /* Phase 2: post-pass pull-toward-target. */
    for (int p = 0; p < post_passes; ++p) {
        int commits = postpass_pull(
            pos, nh, g_scratch.targets, sizes, movable, hw, hh,
            cw, ch,
            g_scratch.heads, g_scratch.next_arr, g_scratch.macro_cell,
            ncx, ncy, inv_cell_size,
            g_scratch.disp_order, g_scratch.disp2);
        if (commits == 0) break;
    }

    /* Phase 3: same-size swap rescue. */
    int top_k = (swap_top_k > 0) ? swap_top_k : 20;
    for (int p = 0; p < swap_passes; ++p) {
        int commits = postpass_swap(
            pos, nh, g_scratch.targets, sizes, movable,
            g_scratch.heads, g_scratch.next_arr, g_scratch.macro_cell,
            ncx, ncy, inv_cell_size,
            g_scratch.disp_order, g_scratch.disp2,
            top_k);
        if (commits == 0) break;
        /* After swap, run another pull pass to refine positions further. */
        postpass_pull(pos, nh, g_scratch.targets, sizes, movable, hw, hh,
                       cw, ch,
                       g_scratch.heads, g_scratch.next_arr, g_scratch.macro_cell,
                       ncx, ncy, inv_cell_size,
                       g_scratch.disp_order, g_scratch.disp2);
    }
}