/* External-proxy-faithful congestion relaxer for macro placement.
 *
 * Ports plc_client_os.get_routing() + __smooth_routing_cong() + abu(V||H, 0.05)
 * exactly into C, then runs a greedy accept/reject optimizer that moves soft
 * macros to reduce the real proxy (guaranteed non-regression on internal).
 *
 * Matches the Python reference at
 *   external/MacroPlacement/CodeElements/Plc_client/plc_client_os.py
 *
 *   H/V stripes for 2/3-pin nets, star-split for >3, macro routing allocation
 *   with partial-overlap correction, 1D smoothing (V along col, H along row),
 *   abu(top-5%) on the concatenated V||H array. Normalizes
 *       V /= grid_w * vroutes_per_micron,  H /= grid_h * hroutes_per_micron
 *   before smoothing, then adds macro routing (pre-normalized) after smoothing.
 *
 * Built automatically by placer.py::_ensure_lib on first use, and rebuilt
 * whenever this source is newer than the .so. Compiler flags live in
 * _C_EXTENSIONS there (-O3 -march=native, so the binary is host-tuned).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline double d_max(double a, double b) { return a > b ? a : b; }
static inline double d_min(double a, double b) { return a < b ? a : b; }
static inline double d_abs(double a) { return a < 0 ? -a : a; }
static inline int    i_max(int a, int b)       { return a > b ? a : b; }
static inline int    i_min(int a, int b)       { return a < b ? a : b; }

/* --- State --- */

typedef struct {
    /* Geometry */
    int n, nh;              /* total macros, hard macros */
    int gr, gc, ng;
    double cw, ch, gw, gh;
    double hrpm, vrpm;
    double h_alloc, v_alloc;
    int smooth_range;
    double grid_h_routes;   /* gh * hrpm  - per-bin H routing capacity */
    double grid_v_routes;   /* gw * vrpm */

    /* Inputs (not owned) */
    double *pos;            /* [n, 2] - modified in place */
    const double *sizes;    /* [n, 2] */
    const int    *movable;  /* [n] */

    /* Pins */
    int np;
    const int    *pin_macro;   /* [np] - -1 for port */
    const double *pin_x;       /* [np] - offset if macro, abs if port */
    const double *pin_y;

    /* Cached pin geometry. Ports are static; macro-pin entries are refreshed
     * only for the macros that actually move. */
    double *pin_abs_x;         /* [np] */
    double *pin_abs_y;         /* [np] */
    int    *pin_row;           /* [np] */
    int    *pin_col;           /* [np] */

    /* Macro -> pins CSR for cheap cache refresh on macro moves. */
    int *mp_offsets;           /* [n+1] */
    int *mp_pin_ids;           /* [# macro pins] */

    /* Nets (CSR over sinks; driver is separate) */
    int nn;
    const int    *net_driver;   /* [nn] */
    const int    *net_sinks_off;/* [nn+1] */
    const int    *net_sinks_idx;/* [net_sinks_off[nn]] */
    const double *net_weight;   /* [nn] - used for routing demand AND WL */
    /* Optional per-net WL-only multiplier (DREAMPlace 4.0-style critical-net
     * weighting). When NULL, treated as 1.0 everywhere (identity = no-op).
     * Multiplies net_weight ONLY at WL gradient + HPWL points; routing
     * demand and density are NOT affected. */
    const double *wl_extra_weight; /* [nn] or NULL */

    /* Dynamic per-bin maps (all pre-normalized by grid_{h,v}_routes) */
    double *H_net;   /* [ng] from net routing (pre-smoothing) */
    double *V_net;
    double *H_macro; /* [ng] from hard-macro routing (static after init) */
    double *V_macro;
    double *H_final;  /* [ng] smoothed H_net + H_macro, kept incrementally synced */
    double *V_final;

    /* Macro -> list of nets containing any pin on that macro (CSR) */
    int *mn_offsets; /* [n+1] */
    int *mn_net_ids; /* sum */

    /* Scratch per-net gcell workspace (bounded by max pins per net + 1) */
    int max_pins_per_net;
    int *scratch_rows;
    int *scratch_cols;

    /* Order-preserving gcell dedup scratch for route_net(). */
    unsigned int *seen_gcell;  /* [ng] generation-stamped */
    unsigned int seen_gen;

    /* Density tracking (all macros, soft+hard) */
    double *grid_occupied;  /* [ng] raw area sum; bin density = / grid_area */
    double grid_area;        /* gw * gh */
    /* Top-N extraction scratch (sized to max(ng/10 for density, ng/5 for cong)+1).
     * Preallocated once in cong_relax_v2; reused across all abu_top_n calls. */
    double *abu_scratch;
    /* HPWL tracking */
    double *net_hpwl;        /* [nn] per-net weighted HPWL */
    double total_hpwl;
    double hpwl_norm;        /* (cw + ch) * net_cnt_for_norm */
    double net_cnt_for_norm; /* sum of net weights, matches plc.net_cnt */

    /* Per-net bbox cache (maintained by compute_net_hpwl). Enables a skip
     * path in the per-candidate HPWL recompute: when none of the moving
     * macro's pins on a net are at the cached extremes AND their new abs
     * positions lie strictly inside the cached bbox, the bbox is unchanged
     * -> cached hpwl is bit-identical to a fresh recompute. */
    double *net_xmin;        /* [nn] */
    double *net_xmax;
    double *net_ymin;
    double *net_ymax;
    int    *net_pin_xmin;    /* [nn] pin id at each extreme (ties: first-seen) */
    int    *net_pin_xmax;
    int    *net_pin_ymin;
    int    *net_pin_ymax;

    /* Current proxy cost components */
    double cong_cost;        /* abu top-5% (V_final || H_final) */
    double density_cost;     /* 0.5 * abu top-10% of bin densities */
    double wl_cost;          /* total_hpwl / hpwl_norm */
    double full_cost;        /* wl_cost + 0.5*density_cost + 0.5*cong_cost */
} State;

/* --- Incremental smooth -> H_final/V_final ---
 * Spread pattern (matches compute_smooth_into_final exactly):
 *   V_final[row, ptr] += V_net[row, col] / gcell_cnt_col(col)   ptr ∈ [col-sr, col+sr]∩[0,gc-1]
 *   H_final[ptr, col] += H_net[row, col] / gcell_cnt_row(row)   ptr ∈ [row-sr, row+sr]∩[0,gr-1]
 *
 * V_macro/H_macro are constant after init and are folded into H_final/V_final
 * once at bootstrap; deltas to H_net/V_net drive H_final/V_final by the same
 * spread with uniform weight 1/gcell_cnt.
 *
 * Callers in route_* / route_macro must funnel through these helpers so that
 * H_final/V_final stay in sync with H_net/V_net/H_macro/V_macro. */
static inline void add_H_net(State *s, int r, int c, double v) {
    int gc = s->gc, gr = s->gr, sr = s->smooth_range;
    s->H_net[r * gc + c] += v;
    int lp = r - sr; if (lp < 0) lp = 0;
    int up = r + sr; if (up >= gr) up = gr - 1;
    int cnt = up - lp + 1;
    double val = v / cnt;
    double *hf = s->H_final;
    for (int ptr = lp; ptr <= up; ptr++) hf[ptr * gc + c] += val;
}

static inline void add_V_net(State *s, int r, int c, double v) {
    int gc = s->gc, sr = s->smooth_range;
    s->V_net[r * gc + c] += v;
    int lp = c - sr; if (lp < 0) lp = 0;
    int rp = c + sr; if (rp >= gc) rp = gc - 1;
    int cnt = rp - lp + 1;
    double val = v / cnt;
    double *vf = s->V_final;
    int base = r * gc;
    for (int ptr = lp; ptr <= rp; ptr++) vf[base + ptr] += val;
}

/* Bulk H spread for `for (col in [col_min, col_max)) add_H_net(s, r, col, v)`.
 * The naive form has H_final writes at stride gc inside add_H_net's smooth
 * loop, costing ~5 cache lines per call. Reordering to ptr-outer / col-inner
 * makes the inner loop contiguous (vectorizable) and touches each H_final row
 * once. Mathematically identical to the loop of add_H_net calls - every
 * (col, ptr) pair gets updated exactly once, so addition order is irrelevant. */
static inline void bulk_h_spread(State *s, int r, int col_min, int col_max, double v) {
    if (col_min >= col_max) return;
    int gc = s->gc, gr = s->gr, sr = s->smooth_range;
    int lp = r - sr; if (lp < 0) lp = 0;
    int up = r + sr; if (up >= gr) up = gr - 1;
    int cnt = up - lp + 1;
    double val = v / cnt;

    double *h_net_row = s->H_net + r * gc;
    for (int col = col_min; col < col_max; col++) h_net_row[col] += v;

    for (int ptr = lp; ptr <= up; ptr++) {
        double *h_final_row = s->H_final + ptr * gc;
        for (int col = col_min; col < col_max; col++) h_final_row[col] += val;
    }
}

/* --- Pin -> gcell --- */

static inline void update_pin_cache_entry(State *s, int pin_idx) {
    int m = s->pin_macro[pin_idx];
    double x, y;
    if (m < 0) {
        x = s->pin_x[pin_idx];
        y = s->pin_y[pin_idx];
    } else {
        x = s->pos[m*2 + 0] + s->pin_x[pin_idx];
        y = s->pos[m*2 + 1] + s->pin_y[pin_idx];
    }
    int col = (int)floor(x / s->gw);
    int row = (int)floor(y / s->gh);
    if (col < 0) col = 0; if (col >= s->gc) col = s->gc - 1;
    if (row < 0) row = 0; if (row >= s->gr) row = s->gr - 1;
    s->pin_abs_x[pin_idx] = x;
    s->pin_abs_y[pin_idx] = y;
    s->pin_row[pin_idx] = row;
    s->pin_col[pin_idx] = col;
}

static void build_macro_pin_csr(State *s) {
    for (int i = 0; i <= s->n; i++) s->mp_offsets[i] = 0;
    for (int p = 0; p < s->np; p++) {
        int m = s->pin_macro[p];
        if (m >= 0) s->mp_offsets[m + 1]++;
    }
    for (int i = 1; i <= s->n; i++) s->mp_offsets[i] += s->mp_offsets[i - 1];
    int *cursor = (int *)calloc(s->n > 0 ? s->n : 1, sizeof(int));
    for (int p = 0; p < s->np; p++) {
        int m = s->pin_macro[p];
        if (m < 0) continue;
        int idx = s->mp_offsets[m] + cursor[m]++;
        s->mp_pin_ids[idx] = p;
    }
    free(cursor);
}

static void refresh_all_pin_cache(State *s) {
    for (int p = 0; p < s->np; p++) update_pin_cache_entry(s, p);
}

static void refresh_macro_pin_cache(State *s, int macro_idx) {
    int off = s->mp_offsets[macro_idx];
    int end = s->mp_offsets[macro_idx + 1];
    for (int k = off; k < end; k++) update_pin_cache_entry(s, s->mp_pin_ids[k]);
}

static inline void pin_gcell(const State *s, int pin_idx, int *r, int *c) {
    *r = s->pin_row[pin_idx];
    *c = s->pin_col[pin_idx];
}

static inline void pin_pos(const State *s, int pin_idx, double *x, double *y) {
    *x = s->pin_abs_x[pin_idx];
    *y = s->pin_abs_y[pin_idx];
}

/* --- Routing primitives (H_net / V_net) --- */

static inline void route_two_pin(State *s,
                                 int src_r, int src_c,
                                 int snk_r, int snk_c,
                                 double w, double sign) {
    int row_min = i_min(src_r, snk_r), row_max = i_max(src_r, snk_r);
    int col_min = i_min(src_c, snk_c), col_max = i_max(src_c, snk_c);
    double wn_h = sign * w / s->grid_h_routes;
    double wn_v = sign * w / s->grid_v_routes;
    bulk_h_spread(s, src_r, col_min, col_max, wn_h);
    for (int row = row_min; row < row_max; row++) add_V_net(s, row, snk_c, wn_v);
}

static inline void route_l(State *s,
                           int y1, int x1, int y2, int x2, int y3, int x3,
                           double w, double sign) {
    double wn_h = sign * w / s->grid_h_routes;
    double wn_v = sign * w / s->grid_v_routes;
    bulk_h_spread(s, y1, x1, x2, wn_h);
    bulk_h_spread(s, y2, x2, x3, wn_h);
    int r1lo = i_min(y1, y2), r1hi = i_max(y1, y2);
    for (int row = r1lo; row < r1hi; row++) add_V_net(s, row, x2, wn_v);
    int r2lo = i_min(y2, y3), r2hi = i_max(y2, y3);
    for (int row = r2lo; row < r2hi; row++) add_V_net(s, row, x3, wn_v);
}

static inline void route_t(State *s, int rows[3], int cols[3], double w, double sign) {
    /* Python's node_gcells.sort() sorts tuples lexicographically: (row, col). */
    int idx[3] = {0, 1, 2};
    for (int i = 1; i < 3; i++) {
        int ii = i;
        while (ii > 0) {
            int a = idx[ii - 1], b = idx[ii];
            if (rows[a] > rows[b] || (rows[a] == rows[b] && cols[a] > cols[b])) {
                idx[ii - 1] = b; idx[ii] = a; ii--;
            } else break;
        }
    }
    int y1 = rows[idx[0]], x1 = cols[idx[0]];
    int y2 = rows[idx[1]], x2 = cols[idx[1]];
    int y3 = rows[idx[2]], x3 = cols[idx[2]];
    int xmin = i_min(i_min(x1, x2), x3);
    int xmax = i_max(i_max(x1, x2), x3);
    double wn_h = sign * w / s->grid_h_routes;
    double wn_v = sign * w / s->grid_v_routes;
    bulk_h_spread(s, y2, xmin, xmax, wn_h);
    int r1lo = i_min(y1, y2), r1hi = i_max(y1, y2);
    for (int row = r1lo; row < r1hi; row++) add_V_net(s, row, x1, wn_v);
    int r2lo = i_min(y2, y3), r2hi = i_max(y2, y3);
    for (int row = r2lo; row < r2hi; row++) add_V_net(s, row, x3, wn_v);
}

static inline void route_three_pin(State *s, int rows[3], int cols[3],
                                   double w, double sign) {
    /* Python sort key: (col, row) */
    int idx[3] = {0, 1, 2};
    for (int i = 1; i < 3; i++) {
        int ii = i;
        while (ii > 0) {
            int a = idx[ii - 1], b = idx[ii];
            if (cols[a] > cols[b] || (cols[a] == cols[b] && rows[a] > rows[b])) {
                idx[ii - 1] = b; idx[ii] = a; ii--;
            } else break;
        }
    }
    int y1 = rows[idx[0]], x1 = cols[idx[0]];
    int y2 = rows[idx[1]], x2 = cols[idx[1]];
    int y3 = rows[idx[2]], x3 = cols[idx[2]];
    int miny13 = i_min(y1, y3), maxy13 = i_max(y1, y3);

    if (x1 < x2 && x2 < x3 && miny13 < y2 && maxy13 > y2) {
        route_l(s, y1, x1, y2, x2, y3, x3, w, sign);
    } else if (x2 == x3 && x1 < x2 && y1 < i_min(y2, y3)) {
        double wn_h = sign * w / s->grid_h_routes;
        double wn_v = sign * w / s->grid_v_routes;
        bulk_h_spread(s, y1, x1, x2, wn_h);
        int row_hi = i_max(y2, y3);
        for (int row = y1; row < row_hi; row++) add_V_net(s, row, x2, wn_v);
    } else if (y2 == y3) {
        double wn_h = sign * w / s->grid_h_routes;
        double wn_v = sign * w / s->grid_v_routes;
        bulk_h_spread(s, y1, x1, x2, wn_h);
        bulk_h_spread(s, y2, x2, x3, wn_h);
        int rlo = i_min(y1, y2), rhi = i_max(y1, y2);
        for (int row = rlo; row < rhi; row++) add_V_net(s, row, x2, wn_v);
    } else {
        /* Python __t_routing re-sorts by (row, col) internally */
        int rows_copy[3] = {rows[0], rows[1], rows[2]};
        int cols_copy[3] = {cols[0], cols[1], cols[2]};
        route_t(s, rows_copy, cols_copy, w, sign);
    }
}

/* Route a single net (apply ±1 scale). Collects unique gcells from driver + sinks,
 * then dispatches by count. */
static void route_net(State *s, int net_idx, double sign) {
    int d_pin = s->net_driver[net_idx];
    int s_off = s->net_sinks_off[net_idx];
    int s_end = s->net_sinks_off[net_idx + 1];
    double w = s->net_weight[net_idx];

    int *gids = s->scratch_rows;
    int n_unique = 0;
    unsigned int gen = s->seen_gen + 1u;
    if (gen == 0u) {
        memset(s->seen_gcell, 0, sizeof(unsigned int) * s->ng);
        gen = 1u;
    }
    s->seen_gen = gen;

    int src_r, src_c;
    pin_gcell(s, d_pin, &src_r, &src_c);
    int src_gid = src_r * s->gc + src_c;
    s->seen_gcell[src_gid] = gen;
    gids[n_unique++] = src_gid;

    for (int k = s_off; k < s_end; k++) {
        int p = s->net_sinks_idx[k];
        int gid = s->pin_row[p] * s->gc + s->pin_col[p];
        if (s->seen_gcell[gid] == gen) continue;
        s->seen_gcell[gid] = gen;
        gids[n_unique++] = gid;
    }

    if (n_unique < 2) return;
    if (n_unique == 2) {
        /* two_pin needs source_gcell separate from sink_gcell */
        int other = (gids[0] == src_gid) ? 1 : 0;
        int ogid = gids[other];
        route_two_pin(s, src_r, src_c, ogid / s->gc, ogid % s->gc, w, sign);
    } else if (n_unique == 3) {
        int r3[3] = {gids[0] / s->gc, gids[1] / s->gc, gids[2] / s->gc};
        int c3[3] = {gids[0] % s->gc, gids[1] % s->gc, gids[2] % s->gc};
        route_three_pin(s, r3, c3, w, sign);
    } else {
        /* split: each non-source gcell pairs with source as a 2-pin */
        for (int u = 0; u < n_unique; u++) {
            int gid = gids[u];
            if (gid == src_gid) continue;
            route_two_pin(s, src_r, src_c, gid / s->gc, gid % s->gc, w, sign);
        }
    }
}

/* --- Hard-macro routing (static) --- */

static double overlap_1d(double a_lo, double a_hi, double b_lo, double b_hi) {
    double v = d_min(a_hi, b_hi) - d_max(a_lo, b_lo);
    return v > 0 ? v : 0.0;
}

/* Port of __macro_route_over_grid_cell. Adds into H_macro, V_macro
 * (pre-normalized). Defined later as route_macro_signed; this is just a
 * +1 wrapper that the compiler inlines at -O3. */
static void route_macro_signed(State *s, int macro_idx, double sign);
static inline void route_macro(State *s, int macro_idx) {
    route_macro_signed(s, macro_idx, +1.0);
}

/* --- Smoothing (V along cols, H along rows) ---
 * Spread V_net/H_net into V_final/H_final, then fold in macro contributions.
 * V_final/H_final must be the only smoothed buffers - incremental updates in
 * add_H_net/add_V_net drive them directly to avoid maintaining a separate
 * smoothed map. */

static void compute_smooth_into_final(State *s) {
    int gr = s->gr, gc = s->gc, sr = s->smooth_range, ng = s->ng;
    for (int i = 0; i < ng; i++) {
        s->V_final[i] = s->V_macro[i];
        s->H_final[i] = s->H_macro[i];
    }
    for (int row = 0; row < gr; row++) {
        for (int col = 0; col < gc; col++) {
            int lp = col - sr; if (lp < 0) lp = 0;
            int rp = col + sr; if (rp >= gc) rp = gc - 1;
            int gcell_cnt = rp - lp + 1;
            double val = s->V_net[row * gc + col] / gcell_cnt;
            for (int ptr = lp; ptr <= rp; ptr++)
                s->V_final[row * gc + ptr] += val;
        }
    }
    for (int row = 0; row < gr; row++) {
        for (int col = 0; col < gc; col++) {
            int lp = row - sr; if (lp < 0) lp = 0;
            int up = row + sr; if (up >= gr) up = gr - 1;
            int gcell_cnt = up - lp + 1;
            double val = s->H_net[row * gc + col] / gcell_cnt;
            for (int ptr = lp; ptr <= up; ptr++)
                s->H_final[ptr * gc + col] += val;
        }
    }
}

/* --- abu: top-5% mean of (V||H) ---
 * Port of plc_client_os.abu: sort descending, take floor(len * n) entries,
 * mean them. If cnt == 0, return max. len = na + nb.
 *
 * `scratch` must point to at least `floor((na+nb)*n_frac)` doubles. Caller
 * owns the buffer (preallocated once in cong_relax_v2).
 */
static double abu_top_n(const double *A, int na,
                        const double *B, int nb,
                        double n_frac, double *scratch) {
    int total = na + nb;
    int cnt = (int)floor((double)total * n_frac);
    if (cnt == 0) {
        double m;
        int have = 0;
        if (na > 0) { m = A[0]; have = 1; for (int i = 1; i < na; i++) if (A[i] > m) m = A[i]; }
        if (nb > 0) {
            if (!have) { m = B[0]; have = 1; }
            for (int i = 0; i < nb; i++) if (B[i] > m) m = B[i];
        }
        return have ? m : 0.0;
    }
    double *h = scratch;
    /* Prime: fill from A first, then B if needed. */
    int a_prime = (cnt < na) ? cnt : na;
    for (int i = 0; i < a_prime; i++) h[i] = A[i];
    int b_prime = cnt - a_prime;
    for (int i = 0; i < b_prime; i++) h[a_prime + i] = B[i];
    /* Build min-heap. */
    for (int i = cnt / 2 - 1; i >= 0; i--) {
        int p = i;
        while (1) {
            int l = 2*p + 1, r = 2*p + 2, t = p;
            if (l < cnt && h[l] < h[t]) t = l;
            if (r < cnt && h[r] < h[t]) t = r;
            if (t == p) break;
            double tmp = h[p]; h[p] = h[t]; h[t] = tmp;
            p = t;
        }
    }
    /* Scan remaining A (no branch on source). */
    for (int i = a_prime; i < na; i++) {
        double v = A[i];
        if (v <= h[0]) continue;
        h[0] = v;
        int p = 0;
        while (1) {
            int l = 2*p + 1, r = 2*p + 2, t = p;
            if (l < cnt && h[l] < h[t]) t = l;
            if (r < cnt && h[r] < h[t]) t = r;
            if (t == p) break;
            double tmp = h[p]; h[p] = h[t]; h[t] = tmp;
            p = t;
        }
    }
    /* Scan remaining B. */
    for (int i = b_prime; i < nb; i++) {
        double v = B[i];
        if (v <= h[0]) continue;
        h[0] = v;
        int p = 0;
        while (1) {
            int l = 2*p + 1, r = 2*p + 2, t = p;
            if (l < cnt && h[l] < h[t]) t = l;
            if (r < cnt && h[r] < h[t]) t = r;
            if (t == p) break;
            double tmp = h[p]; h[p] = h[t]; h[t] = tmp;
            p = t;
        }
    }
    double sum = 0.0;
    for (int i = 0; i < cnt; i++) sum += h[i];
    return sum / cnt;
}

/* H_final and V_final are maintained incrementally by add_H_net/add_V_net and
 * route_macro. This just reads the top-5% mean - no smooth/finalize sweep. */
static double compute_cong(State *s) {
    return abu_top_n(s->V_final, s->ng, s->H_final, s->ng, 0.05, s->abu_scratch);
}

/* --- Density tracking ---
 * grid_occupied[i] is raw area sum (un-normalized). Bin density = /grid_area.
 * Each macro contributes xd*yd (area overlap) to bins it touches. Iterates ALL
 * macros (soft + hard) to match plc.get_grid_cells_density.
 */
static void density_macro(State *s, int m, double sign) {
    double cx = s->pos[m*2 + 0];
    double cy = s->pos[m*2 + 1];
    double hw = s->sizes[m*2 + 0] * 0.5;
    double hh = s->sizes[m*2 + 1] * 0.5;
    double mlx = cx - hw, mhx = cx + hw;
    double mly = cy - hh, mhy = cy + hh;
    int ur_row = (int)floor(mhy / s->gh);
    int ur_col = (int)floor(mhx / s->gw);
    int bl_row = (int)floor(mly / s->gh);
    int bl_col = (int)floor(mlx / s->gw);
    if (!(ur_row >= 0 && ur_col >= 0)) return;
    if (bl_row < 0) bl_row = 0;
    if (bl_col < 0) bl_col = 0;
    if (!(bl_row >= 0 && bl_col >= 0)) return;
    if (ur_row > s->gr - 1) ur_row = s->gr - 1;
    if (ur_col > s->gc - 1) ur_col = s->gc - 1;
    for (int r = bl_row; r <= ur_row; r++) {
        double bin_y_lo = r * s->gh, bin_y_hi = bin_y_lo + s->gh;
        double yd = overlap_1d(mly, mhy, bin_y_lo, bin_y_hi);
        for (int c = bl_col; c <= ur_col; c++) {
            double bin_x_lo = c * s->gw, bin_x_hi = bin_x_lo + s->gw;
            double xd = overlap_1d(mlx, mhx, bin_x_lo, bin_x_hi);
            s->grid_occupied[r * s->gc + c] += sign * xd * yd;
        }
    }
}

static double compute_density(State *s) {
    /* abu_top_n only uses ordering and returns the mean; scaling by a positive
     * constant is monotonic, so we can run abu on the raw `grid_occupied` and
     * scale the scalar result. Returns 0.5 * abu to match plc.get_density_cost(). */
    double a = abu_top_n(s->grid_occupied, s->ng, NULL, 0, 0.10, s->abu_scratch);
    return 0.5 * a / s->grid_area;
}

/* --- HPWL tracking --- */

static double compute_net_hpwl(State *s, int net_idx) {
    int d_pin = s->net_driver[net_idx];
    int s_off = s->net_sinks_off[net_idx];
    int s_end = s->net_sinks_off[net_idx + 1];
    double w = s->net_weight[net_idx];
    if (s->wl_extra_weight) w *= s->wl_extra_weight[net_idx];
    double dx, dy; pin_pos(s, d_pin, &dx, &dy);
    double xmin = dx, xmax = dx, ymin = dy, ymax = dy;
    int p_xmin = d_pin, p_xmax = d_pin, p_ymin = d_pin, p_ymax = d_pin;
    for (int k = s_off; k < s_end; k++) {
        int p = s->net_sinks_idx[k];
        double px, py;
        pin_pos(s, p, &px, &py);
        if (px < xmin) { xmin = px; p_xmin = p; }
        if (px > xmax) { xmax = px; p_xmax = p; }
        if (py < ymin) { ymin = py; p_ymin = p; }
        if (py > ymax) { ymax = py; p_ymax = p; }
    }
    s->net_xmin[net_idx] = xmin;
    s->net_xmax[net_idx] = xmax;
    s->net_ymin[net_idx] = ymin;
    s->net_ymax[net_idx] = ymax;
    s->net_pin_xmin[net_idx] = p_xmin;
    s->net_pin_xmax[net_idx] = p_xmax;
    s->net_pin_ymin[net_idx] = p_ymin;
    s->net_pin_ymax[net_idx] = p_ymax;
    return w * ((xmax - xmin) + (ymax - ymin));
}

/* Try to reuse cached per-net bbox when the moving macro's pins don't
 * disturb it. Inputs are the snapshot taken at si entry (NOT s->net_*,
 * which may have been overwritten by a prior candidate in the same si).
 * Returns 1 and writes h on skip; returns 0 (caller must recompute) otherwise. */
static inline int try_skip_net_hpwl(State *s, int ni, int m,
                                    double xmin, double xmax, double ymin, double ymax,
                                    int p_xmin, int p_xmax, int p_ymin, int p_ymax,
                                    double *out_h) {
    int d_pin = s->net_driver[ni];
    if (s->pin_macro[d_pin] == m) {
        if (d_pin == p_xmin || d_pin == p_xmax ||
            d_pin == p_ymin || d_pin == p_ymax) return 0;
        double dx = s->pin_abs_x[d_pin], dy = s->pin_abs_y[d_pin];
        if (dx < xmin || dx > xmax || dy < ymin || dy > ymax) return 0;
    }
    int s_off = s->net_sinks_off[ni], s_end = s->net_sinks_off[ni + 1];
    for (int k = s_off; k < s_end; k++) {
        int p = s->net_sinks_idx[k];
        if (s->pin_macro[p] != m) continue;
        if (p == p_xmin || p == p_xmax || p == p_ymin || p == p_ymax) return 0;
        double px = s->pin_abs_x[p], py = s->pin_abs_y[p];
        if (px < xmin || px > xmax || py < ymin || py > ymax) return 0;
    }
    double w = s->net_weight[ni];
    if (s->wl_extra_weight) w *= s->wl_extra_weight[ni];
    *out_h = w * ((xmax - xmin) + (ymax - ymin));
    return 1;
}

/* --- Init: build state, compute initial routing --- */

static void init_state(State *s) {
    int ng = s->ng;
    refresh_all_pin_cache(s);
    memset(s->H_net,   0, sizeof(double) * ng);
    memset(s->V_net,   0, sizeof(double) * ng);
    memset(s->H_macro, 0, sizeof(double) * ng);
    memset(s->V_macro, 0, sizeof(double) * ng);
    /* H_final/V_final are incrementally maintained by add_H_net/add_V_net and
     * route_macro, so they must start at zero before any routing happens. */
    memset(s->H_final, 0, sizeof(double) * ng);
    memset(s->V_final, 0, sizeof(double) * ng);

    /* 1) Hard macro routing (static) */
    for (int i = 0; i < s->nh; i++) route_macro(s, i);

    /* 2) Net routing - route all nets once */
    for (int i = 0; i < s->nn; i++) route_net(s, i, +1.0);

    /* 3) Build macro -> nets CSR. A macro owns a net if any of its pins (driver or
     *    sink) references that macro. A port-only net doesn't appear in the map. */
    for (int i = 0; i <= s->n; i++) s->mn_offsets[i] = 0;
    /* first pass: count, but dedupe within a single net (don't double-count if a macro
     * has multiple pins on the same net) */
    for (int ni = 0; ni < s->nn; ni++) {
        int s_off = s->net_sinks_off[ni], s_end = s->net_sinks_off[ni + 1];
        /* Collect unique macro indices on this net. Use scratch_rows as temp. */
        int unique_cap = (s_end - s_off) + 1;
        /* Actually just mark via a temporary - use a simple small-N dedup. We use
         * scratch_rows as a set since max_pins_per_net >= all pins here. */
        int cnt = 0;
        int d = s->net_driver[ni];
        int dm = s->pin_macro[d];
        if (dm >= 0) { s->scratch_rows[cnt++] = dm; }
        for (int k = s_off; k < s_end; k++) {
            int p = s->net_sinks_idx[k];
            int pm = s->pin_macro[p];
            if (pm < 0) continue;
            int dup = 0;
            for (int u = 0; u < cnt; u++) if (s->scratch_rows[u] == pm) { dup = 1; break; }
            if (!dup) s->scratch_rows[cnt++] = pm;
        }
        for (int u = 0; u < cnt; u++) s->mn_offsets[s->scratch_rows[u] + 1]++;
        (void)unique_cap;
    }
    for (int i = 1; i <= s->n; i++) s->mn_offsets[i] += s->mn_offsets[i - 1];
    int total = s->mn_offsets[s->n];
    int *cursor = (int *)calloc(s->n > 0 ? s->n : 1, sizeof(int));
    for (int ni = 0; ni < s->nn; ni++) {
        int s_off = s->net_sinks_off[ni], s_end = s->net_sinks_off[ni + 1];
        int cnt = 0;
        int d = s->net_driver[ni];
        int dm = s->pin_macro[d];
        if (dm >= 0) { s->scratch_rows[cnt++] = dm; }
        for (int k = s_off; k < s_end; k++) {
            int p = s->net_sinks_idx[k];
            int pm = s->pin_macro[p];
            if (pm < 0) continue;
            int dup = 0;
            for (int u = 0; u < cnt; u++) if (s->scratch_rows[u] == pm) { dup = 1; break; }
            if (!dup) s->scratch_rows[cnt++] = pm;
        }
        for (int u = 0; u < cnt; u++) {
            int mm = s->scratch_rows[u];
            int idx = s->mn_offsets[mm] + cursor[mm]++;
            s->mn_net_ids[idx] = ni;
        }
    }
    free(cursor);
    (void)total;

    /* Density: deposit ALL macros (soft + hard). */
    memset(s->grid_occupied, 0, sizeof(double) * s->ng);
    for (int i = 0; i < s->n; i++) density_macro(s, i, +1.0);

    /* HPWL: compute per-net and total. */
    s->total_hpwl = 0.0;
    for (int i = 0; i < s->nn; i++) {
        double h = compute_net_hpwl(s, i);
        s->net_hpwl[i] = h;
        s->total_hpwl += h;
    }

    /* Canonicalize H_final/V_final from scratch so the initial abu uses the
     * exact full-smooth values. Incremental updates from add_H_net/add_V_net
     * and route_macro are mathematically equivalent, but this bootstrap
     * eliminates any FP-order drift that could arise during init. */
    compute_smooth_into_final(s);

    s->cong_cost = compute_cong(s);
    s->density_cost = compute_density(s);
    s->wl_cost = (s->hpwl_norm > 0.0) ? (s->total_hpwl / s->hpwl_norm) : 0.0;
    s->full_cost = s->wl_cost + 0.5 * s->density_cost + 0.5 * s->cong_cost;
}

/* Rebuild all position-dependent maps from scratch. Skips CSR (mn_offsets,
 * mn_net_ids) since net topology doesn't depend on positions. Used to flush
 * FP drift accumulated by many incremental add_H_net/move_macro_in/out calls,
 * which otherwise causes pair-swap acceptance to drift away from true cost. */
static void rebuild_maps(State *s) {
    int ng = s->ng;
    memset(s->H_net,   0, sizeof(double) * ng);
    memset(s->V_net,   0, sizeof(double) * ng);
    memset(s->H_macro, 0, sizeof(double) * ng);
    memset(s->V_macro, 0, sizeof(double) * ng);
    memset(s->H_final, 0, sizeof(double) * ng);
    memset(s->V_final, 0, sizeof(double) * ng);

    for (int i = 0; i < s->nh; i++) route_macro(s, i);
    for (int i = 0; i < s->nn; i++) route_net(s, i, +1.0);

    memset(s->grid_occupied, 0, sizeof(double) * ng);
    for (int i = 0; i < s->n; i++) density_macro(s, i, +1.0);

    s->total_hpwl = 0.0;
    for (int i = 0; i < s->nn; i++) {
        double h = compute_net_hpwl(s, i);
        s->net_hpwl[i] = h;
        s->total_hpwl += h;
    }

    compute_smooth_into_final(s);

    s->cong_cost = compute_cong(s);
    s->density_cost = compute_density(s);
    s->wl_cost = (s->hpwl_norm > 0.0) ? (s->total_hpwl / s->hpwl_norm) : 0.0;
    s->full_cost = s->wl_cost + 0.5 * s->density_cost + 0.5 * s->cong_cost;
}

/* --- Move/revert for macro m to (nx, ny) ---
 * Out: subtract m's routing, density, and per-net HPWL contribution.
 * In:  add the same back at the new position.
 * Caller is responsible for setting s->pos[m] between out and in.
 */
static void move_macro_out(State *s, int m) {
    int off = s->mn_offsets[m], end = s->mn_offsets[m + 1];
    for (int k = off; k < end; k++) {
        int ni = s->mn_net_ids[k];
        route_net(s, ni, -1.0);
        s->total_hpwl -= s->net_hpwl[ni];
    }
    density_macro(s, m, -1.0);
}
static void move_macro_in(State *s, int m) {
    int off = s->mn_offsets[m], end = s->mn_offsets[m + 1];
    for (int k = off; k < end; k++) {
        int ni = s->mn_net_ids[k];
        route_net(s, ni, +1.0);
        double h = compute_net_hpwl(s, ni);
        s->net_hpwl[ni] = h;
        s->total_hpwl += h;
    }
    density_macro(s, m, +1.0);
}

/* --- Informed move direction ---
 * Returns a unit direction (dx_out, dy_out) that blends three signals
 * proportional to their weight in the full proxy (1.wl + 0.5.den + 0.5.cong):
 *   - HPWL centroid pull: toward the weighted centroid of all connected pins
 *     that are NOT on this macro.
 *   - Density repulsion: away from higher bin-occupancy in a 2-bin window.
 *   - Congestion repulsion: away from higher H_final+V_final in a 2-bin window.
 * Each signal is unit-normalized before blending, so their magnitudes are
 * comparable regardless of raw scale.
 */
/* Compute per-component unit gradients AND the blended gradient. Outputs 8
 * doubles; any component with no meaningful signal is zeroed.
 *   out[0,1] - blended (hpwl/den/cong, proxy-weighted)
 *   out[2,3] - hpwl-only
 *   out[4,5] - density-only
 *   out[6,7] - congestion-only
 * Splitting lets the candidate generator try moves that target ONE component
 * at a time - useful when the blended direction is near-zero from cancellation
 * but one component still wants a strong move. */
static void compute_dirs(State *s, int m, double out[8]) {
    for (int i = 0; i < 8; i++) out[i] = 0.0;
    double cx = s->pos[m*2+0], cy = s->pos[m*2+1];
    int mc = (int)floor(cx / s->gw);
    int mr = (int)floor(cy / s->gh);
    if (mc < 0) mc = 0; if (mc > s->gc - 1) mc = s->gc - 1;
    if (mr < 0) mr = 0; if (mr > s->gr - 1) mr = s->gr - 1;

    double hpwl_dx = 0.0, hpwl_dy = 0.0, tw = 0.0;
    int off = s->mn_offsets[m], end = s->mn_offsets[m + 1];
    for (int k = off; k < end; k++) {
        int ni = s->mn_net_ids[k];
        double sum_x = 0.0, sum_y = 0.0;
        int count = 0;
        int d_pin = s->net_driver[ni];
        if (s->pin_macro[d_pin] != m) {
            double px, py; pin_pos(s, d_pin, &px, &py);
            sum_x += px; sum_y += py; count++;
        }
        int so = s->net_sinks_off[ni], se = s->net_sinks_off[ni + 1];
        for (int kk = so; kk < se; kk++) {
            int p = s->net_sinks_idx[kk];
            if (s->pin_macro[p] != m) {
                double px, py; pin_pos(s, p, &px, &py);
                sum_x += px; sum_y += py; count++;
            }
        }
        if (count > 0) {
            double cxn = sum_x / count, cyn = sum_y / count;
            double w = s->net_weight[ni];
            if (s->wl_extra_weight) w *= s->wl_extra_weight[ni];
            hpwl_dx += w * (cxn - cx);
            hpwl_dy += w * (cyn - cy);
            tw += w;
        }
    }
    if (tw > 0.0) { hpwl_dx /= tw; hpwl_dy /= tw; }

    const int WIN = 2;
    double dens_dx = 0.0, dens_dy = 0.0;
    {
        int lc = mc - WIN; if (lc < 0) lc = 0;
        int rc = mc + WIN; if (rc > s->gc - 1) rc = s->gc - 1;
        int dr = mr - WIN; if (dr < 0) dr = 0;
        int ur = mr + WIN; if (ur > s->gr - 1) ur = s->gr - 1;
        double lsum = 0.0, rsum = 0.0, dsum = 0.0, usum = 0.0;
        int nl = 0, nr = 0, nd = 0, nu = 0;
        for (int c = lc; c < mc; c++) { lsum += s->grid_occupied[mr * s->gc + c]; nl++; }
        for (int c = mc + 1; c <= rc; c++) { rsum += s->grid_occupied[mr * s->gc + c]; nr++; }
        for (int r = dr; r < mr; r++) { dsum += s->grid_occupied[r * s->gc + mc]; nd++; }
        for (int r = mr + 1; r <= ur; r++) { usum += s->grid_occupied[r * s->gc + mc]; nu++; }
        double l = nl > 0 ? lsum / nl : 0.0;
        double r = nr > 0 ? rsum / nr : 0.0;
        double dd = nd > 0 ? dsum / nd : 0.0;
        double u = nu > 0 ? usum / nu : 0.0;
        dens_dx = l - r;
        dens_dy = dd - u;
    }

    double cong_dx = 0.0, cong_dy = 0.0;
    {
        int lc = mc - WIN; if (lc < 0) lc = 0;
        int rc = mc + WIN; if (rc > s->gc - 1) rc = s->gc - 1;
        int dr = mr - WIN; if (dr < 0) dr = 0;
        int ur = mr + WIN; if (ur > s->gr - 1) ur = s->gr - 1;
        double lsum = 0.0, rsum = 0.0, dsum = 0.0, usum = 0.0;
        int nl = 0, nr = 0, nd = 0, nu = 0;
        for (int c = lc; c < mc; c++) {
            int idx = mr * s->gc + c;
            lsum += s->H_final[idx] + s->V_final[idx]; nl++;
        }
        for (int c = mc + 1; c <= rc; c++) {
            int idx = mr * s->gc + c;
            rsum += s->H_final[idx] + s->V_final[idx]; nr++;
        }
        for (int r = dr; r < mr; r++) {
            int idx = r * s->gc + mc;
            dsum += s->H_final[idx] + s->V_final[idx]; nd++;
        }
        for (int r = mr + 1; r <= ur; r++) {
            int idx = r * s->gc + mc;
            usum += s->H_final[idx] + s->V_final[idx]; nu++;
        }
        double l = nl > 0 ? lsum / nl : 0.0;
        double r = nr > 0 ? rsum / nr : 0.0;
        double dd = nd > 0 ? dsum / nd : 0.0;
        double u = nu > 0 ? usum / nu : 0.0;
        cong_dx = l - r;
        cong_dy = dd - u;
    }

    /* Per-component unit vectors. */
    double h_mag = sqrt(hpwl_dx * hpwl_dx + hpwl_dy * hpwl_dy);
    if (h_mag > 1e-12) { out[2] = hpwl_dx / h_mag; out[3] = hpwl_dy / h_mag; }
    double de_mag = sqrt(dens_dx * dens_dx + dens_dy * dens_dy);
    if (de_mag > 1e-12) { out[4] = dens_dx / de_mag; out[5] = dens_dy / de_mag; }
    double co_mag = sqrt(cong_dx * cong_dx + cong_dy * cong_dy);
    if (co_mag > 1e-12) { out[6] = cong_dx / co_mag; out[7] = cong_dy / co_mag; }

    /* Blended: proxy weights 1.0, 0.5, 0.5 on each unit component. */
    double ux = out[2] * 1.0 + out[4] * 0.5 + out[6] * 0.5;
    double uy = out[3] * 1.0 + out[5] * 0.5 + out[7] * 0.5;
    double fm = sqrt(ux * ux + uy * uy);
    if (fm > 1e-12) { out[0] = ux / fm; out[1] = uy / fm; }
}

/* --- Pair-swap pass ---
 * Translation-only moves can get stuck when two macros would each benefit
 * from being in each other's spot. For each soft macro, find its spatially
 * nearest soft neighbor and evaluate the swap. Accept if it reduces the
 * full proxy.
 *
 * Implementation applies the swap in-place on S (subtract old routing+density
 * of both macros' union-of-nets, swap pos, add new routing+density); then
 * either keeps it or reverts via the inverse ops. Uses net_touched to avoid
 * double-counting nets shared between m1 and m2.
 */
/* --- Hot-region prioritized soft-list ordering ---
 *
 * Per Voudouris/Tsang 2003 "Guided Local Search" + ABCDPlace's hotspot-first
 * traversal: sort the soft-macro list each outer round so that macros sitting
 * on the hottest cong+density bins are processed first. Hot macros are far
 * more likely to have improving moves; processing them earlier lets the
 * step-decay/reheat machinery converge in fewer iters.
 *
 * Heat score per macro = sum over its bbox cells of:
 *     H_final[c] + V_final[c] + α . grid_occupied[c].inv_ba
 * Constant per call; computed once at outer-round start and re-sorts soft_list
 * descending by heat. α weights density vs cong. */
typedef struct {
    double h;
    int    m;
} HeatItem;

static int cmp_heat_desc(const void *a, const void *b) {
    double da = ((const HeatItem *)a)->h;
    double db = ((const HeatItem *)b)->h;
    if (da > db) return -1;
    if (da < db) return  1;
    return 0;
}

static void sort_soft_by_heat(const State *s, int *soft_list, int n_soft,
                              HeatItem *items, double alpha_den) {
    double inv_ba = (s->grid_area > 0.0) ? (1.0 / s->grid_area) : 0.0;
    for (int si = 0; si < n_soft; si++) {
        int m = soft_list[si];
        double cx = s->pos[m * 2 + 0], cy = s->pos[m * 2 + 1];
        double hw = s->sizes[m * 2 + 0] * 0.5;
        double hh = s->sizes[m * 2 + 1] * 0.5;
        int c_lo = (int)floor((cx - hw) / s->gw);
        int c_hi = (int)floor((cx + hw - 1e-12) / s->gw);
        int r_lo = (int)floor((cy - hh) / s->gh);
        int r_hi = (int)floor((cy + hh - 1e-12) / s->gh);
        if (c_lo < 0) c_lo = 0;
        if (r_lo < 0) r_lo = 0;
        if (c_hi >= s->gc) c_hi = s->gc - 1;
        if (r_hi >= s->gr) r_hi = s->gr - 1;
        double h = 0.0;
        for (int r = r_lo; r <= r_hi; r++) {
            int row_base = r * s->gc;
            for (int c = c_lo; c <= c_hi; c++) {
                int idx = row_base + c;
                h += s->H_final[idx] + s->V_final[idx]
                   + alpha_den * s->grid_occupied[idx] * inv_ba;
            }
        }
        items[si].h = h;
        items[si].m = m;
    }
    qsort(items, n_soft, sizeof(HeatItem), cmp_heat_desc);
    for (int si = 0; si < n_soft; si++) {
        soft_list[si] = items[si].m;
    }
}

#define PSW_K_NEAR 3
static int pair_swap_pass(
    State *s,
    int *soft_list, int n_soft,
    double eps,
    int *net_touched,   /* [nn], zeroed entering & on return */
    int *net_buf,       /* [nn] scratch */
    double *saved_hpwl, /* [nn] scratch */
    int *knn_cache      /* [PSW_K_NEAR * n_soft] scratch */
) {
    int accepts = 0;
    const int K_NEAR = PSW_K_NEAR;  /* try swap with K nearest soft neighbors */

    /* Precompute K-nearest for all macros in parallel. Neighbors are based on
     * positions at the START of this pass; they may drift slightly as swaps
     * commit, but spatial neighborhoods are stable enough that stale cache
     * still finds most useful swaps. */
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i1 = 0; i1 < n_soft; i1++) {
        int m1 = soft_list[i1];
        double cx1 = s->pos[m1*2+0], cy1 = s->pos[m1*2+1];
        int  idx_local[PSW_K_NEAR];
        double d2_local[PSW_K_NEAR];
        for (int k = 0; k < K_NEAR; k++) { idx_local[k] = -1; d2_local[k] = 1e300; }
        for (int i2 = 0; i2 < n_soft; i2++) {
            if (i2 == i1) continue;
            int m2x = soft_list[i2];
            double dx = cx1 - s->pos[m2x*2+0];
            double dy = cy1 - s->pos[m2x*2+1];
            double d2 = dx*dx + dy*dy;
            if (d2 >= d2_local[K_NEAR - 1]) continue;
            int pos_ins = K_NEAR - 1;
            while (pos_ins > 0 && d2_local[pos_ins - 1] > d2) {
                d2_local[pos_ins] = d2_local[pos_ins - 1];
                idx_local[pos_ins] = idx_local[pos_ins - 1];
                pos_ins--;
            }
            d2_local[pos_ins] = d2;
            idx_local[pos_ins] = i2;
        }
        for (int k = 0; k < K_NEAR; k++) knn_cache[i1 * K_NEAR + k] = idx_local[k];
    }

    for (int i1 = 0; i1 < n_soft; i1++) {
        int m1 = soft_list[i1];
        double cx1 = s->pos[m1*2+0], cy1 = s->pos[m1*2+1];

        /* Try each cached neighbor in order; first improving swap wins. */
        for (int knn = 0; knn < K_NEAR; knn++) {
        int best_i2 = knn_cache[i1 * K_NEAR + knn];
        if (best_i2 < 0) continue;
        int m2 = soft_list[best_i2];
        double cx2 = s->pos[m2*2+0], cy2 = s->pos[m2*2+1];

        double old_full = s->full_cost;
        double old_cong = s->cong_cost;
        double old_den  = s->density_cost;
        double old_wl   = s->wl_cost;
        double old_total_hpwl = s->total_hpwl;

        /* Union of nets for m1 and m2 (avoid double-counting shared nets). */
        int nnets = 0;
        int o, e;
        o = s->mn_offsets[m1]; e = s->mn_offsets[m1+1];
        for (int k = o; k < e; k++) {
            int ni = s->mn_net_ids[k];
            if (!net_touched[ni]) { net_touched[ni] = 1; net_buf[nnets++] = ni; }
        }
        o = s->mn_offsets[m2]; e = s->mn_offsets[m2+1];
        for (int k = o; k < e; k++) {
            int ni = s->mn_net_ids[k];
            if (!net_touched[ni]) { net_touched[ni] = 1; net_buf[nnets++] = ni; }
        }

        for (int k = 0; k < nnets; k++) saved_hpwl[k] = s->net_hpwl[net_buf[k]];

        for (int k = 0; k < nnets; k++) {
            int ni = net_buf[k];
            route_net(s, ni, -1.0);
            s->total_hpwl -= s->net_hpwl[ni];
        }
        density_macro(s, m1, -1.0);
        density_macro(s, m2, -1.0);

        s->pos[m1*2+0] = cx2; s->pos[m1*2+1] = cy2;
        s->pos[m2*2+0] = cx1; s->pos[m2*2+1] = cy1;
        refresh_macro_pin_cache(s, m1);
        refresh_macro_pin_cache(s, m2);

        density_macro(s, m1, +1.0);
        density_macro(s, m2, +1.0);
        for (int k = 0; k < nnets; k++) {
            int ni = net_buf[k];
            route_net(s, ni, +1.0);
            double h = compute_net_hpwl(s, ni);
            s->net_hpwl[ni] = h;
            s->total_hpwl += h;
        }

        double new_cong = compute_cong(s);
        double new_den  = compute_density(s);
        double new_wl   = (s->hpwl_norm > 0.0) ? (s->total_hpwl / s->hpwl_norm) : 0.0;
        double new_full = new_wl + 0.5 * new_den + 0.5 * new_cong;

        int accepted = 0;
        if (new_full < old_full - eps) {
            s->full_cost    = new_full;
            s->cong_cost    = new_cong;
            s->density_cost = new_den;
            s->wl_cost      = new_wl;
            accepts++;
            accepted = 1;
        } else {
            /* Revert the swap. */
            for (int k = 0; k < nnets; k++) {
                int ni = net_buf[k];
                route_net(s, ni, -1.0);
                s->total_hpwl -= s->net_hpwl[ni];
            }
            density_macro(s, m1, -1.0);
            density_macro(s, m2, -1.0);

            s->pos[m1*2+0] = cx1; s->pos[m1*2+1] = cy1;
            s->pos[m2*2+0] = cx2; s->pos[m2*2+1] = cy2;
            refresh_macro_pin_cache(s, m1);
            refresh_macro_pin_cache(s, m2);

            density_macro(s, m1, +1.0);
            density_macro(s, m2, +1.0);
            for (int k = 0; k < nnets; k++) {
                int ni = net_buf[k];
                route_net(s, ni, +1.0);
                s->net_hpwl[ni] = saved_hpwl[k];
                s->total_hpwl += saved_hpwl[k];
            }
            s->full_cost    = old_full;
            s->cong_cost    = old_cong;
            s->density_cost = old_den;
            s->wl_cost      = old_wl;
            s->total_hpwl   = old_total_hpwl;
        }

        for (int k = 0; k < nnets; k++) net_touched[net_buf[k]] = 0;
        if (accepted) break;  /* move to next m1 after first improving swap */
        }  /* end knn loop */
    }
    return accepts;
}

/* --- Per-thread polish scratch pool (cached across calls) ---
 * cong_relax_v2's per-thread scratch (~30 buffers/thread x max_threads)
 * costs ~277 ms of malloc/free per call on ibm03. The polish loop itself
 * is bit-deterministic w.r.t. these buffers (they're uninitialized scratch,
 * written before read in each iter), so we can keep them allocated across
 * calls. Grows on size increase, never shrinks; cached state lives for
 * the process lifetime. Single-threaded by design - cong_relax_v2 is not
 * called recursively, and the OMP team inside the polish loop doesn't
 * call cong_relax_v2 itself.
 *
 * Saves ~277 ms per cong_relax_v2 call after the first.
 */
typedef struct {
    int n_alloc, ng_alloc, nn_alloc, np_alloc;
    int max_threads_alloc, max_pins_alloc;
    int max_pins_per_macro_alloc, max_nets_per_macro_alloc;
    int tl_abu_cap_alloc;

    double **tl_H_net, **tl_V_net, **tl_H_final, **tl_V_final;
    double **tl_grid_occupied, **tl_net_hpwl, **tl_pos, **tl_abu;
    double **tl_pin_abs_x, **tl_pin_abs_y;
    int **tl_pin_row, **tl_pin_col;
    unsigned int **tl_seen;
    unsigned int *tl_seen_gen;
    int **tl_rows, **tl_cols;
    int **tl_baseline_rows, **tl_baseline_cols;
    double **tl_net_xmin, **tl_net_xmax, **tl_net_ymin, **tl_net_ymax;
    int **tl_net_pin_xmin, **tl_net_pin_xmax;
    int **tl_net_pin_ymin, **tl_net_pin_ymax;
    double **tl_base_xmin, **tl_base_xmax;
    double **tl_base_ymin, **tl_base_ymax;
    int **tl_base_pin_xmin, **tl_base_pin_xmax;
    int **tl_base_pin_ymin, **tl_base_pin_ymax;
} TlScratchPool;

static TlScratchPool _tl_pool = {0};

static void _free_tl_pool_inner(TlScratchPool *p) {
    if (!p->tl_H_net) return;
    int mt = p->max_threads_alloc;
    for (int t = 0; t < mt; t++) {
        free(p->tl_H_net[t]); free(p->tl_V_net[t]);
        free(p->tl_H_final[t]); free(p->tl_V_final[t]);
        free(p->tl_grid_occupied[t]); free(p->tl_net_hpwl[t]);
        free(p->tl_pos[t]); free(p->tl_abu[t]);
        free(p->tl_pin_abs_x[t]); free(p->tl_pin_abs_y[t]);
        free(p->tl_pin_row[t]); free(p->tl_pin_col[t]);
        free(p->tl_seen[t]);
        free(p->tl_rows[t]); free(p->tl_cols[t]);
        free(p->tl_baseline_rows[t]); free(p->tl_baseline_cols[t]);
        free(p->tl_net_xmin[t]); free(p->tl_net_xmax[t]);
        free(p->tl_net_ymin[t]); free(p->tl_net_ymax[t]);
        free(p->tl_net_pin_xmin[t]); free(p->tl_net_pin_xmax[t]);
        free(p->tl_net_pin_ymin[t]); free(p->tl_net_pin_ymax[t]);
        free(p->tl_base_xmin[t]); free(p->tl_base_xmax[t]);
        free(p->tl_base_ymin[t]); free(p->tl_base_ymax[t]);
        free(p->tl_base_pin_xmin[t]); free(p->tl_base_pin_xmax[t]);
        free(p->tl_base_pin_ymin[t]); free(p->tl_base_pin_ymax[t]);
    }
    free(p->tl_H_net); free(p->tl_V_net);
    free(p->tl_H_final); free(p->tl_V_final);
    free(p->tl_grid_occupied); free(p->tl_net_hpwl);
    free(p->tl_pos); free(p->tl_abu);
    free(p->tl_pin_abs_x); free(p->tl_pin_abs_y);
    free(p->tl_pin_row); free(p->tl_pin_col);
    free(p->tl_seen); free(p->tl_seen_gen);
    free(p->tl_rows); free(p->tl_cols);
    free(p->tl_baseline_rows); free(p->tl_baseline_cols);
    free(p->tl_net_xmin); free(p->tl_net_xmax);
    free(p->tl_net_ymin); free(p->tl_net_ymax);
    free(p->tl_net_pin_xmin); free(p->tl_net_pin_xmax);
    free(p->tl_net_pin_ymin); free(p->tl_net_pin_ymax);
    free(p->tl_base_xmin); free(p->tl_base_xmax);
    free(p->tl_base_ymin); free(p->tl_base_ymax);
    free(p->tl_base_pin_xmin); free(p->tl_base_pin_xmax);
    free(p->tl_base_pin_ymin); free(p->tl_base_pin_ymax);
    memset(p, 0, sizeof(*p));
}

static void _ensure_tl_pool(
    int n, int ng, int nn, int np,
    int max_threads, int max_pins,
    int max_pins_per_macro, int max_nets_per_macro,
    int tl_abu_cap
) {
    TlScratchPool *p = &_tl_pool;
    int needs_realloc =
        n > p->n_alloc || ng > p->ng_alloc ||
        nn > p->nn_alloc || np > p->np_alloc ||
        max_threads > p->max_threads_alloc ||
        max_pins > p->max_pins_alloc ||
        max_pins_per_macro > p->max_pins_per_macro_alloc ||
        max_nets_per_macro > p->max_nets_per_macro_alloc ||
        tl_abu_cap > p->tl_abu_cap_alloc;
    if (!needs_realloc) return;

    /* Grow each dim to max(old, new) so smaller subsequent calls still hit. */
    int new_n   = (n   > p->n_alloc)   ? n   : p->n_alloc;
    int new_ng  = (ng  > p->ng_alloc)  ? ng  : p->ng_alloc;
    int new_nn  = (nn  > p->nn_alloc)  ? nn  : p->nn_alloc;
    int new_np  = (np  > p->np_alloc)  ? np  : p->np_alloc;
    int new_mt  = (max_threads > p->max_threads_alloc) ? max_threads : p->max_threads_alloc;
    int new_mp  = (max_pins > p->max_pins_alloc) ? max_pins : p->max_pins_alloc;
    int new_mppm = (max_pins_per_macro > p->max_pins_per_macro_alloc) ? max_pins_per_macro : p->max_pins_per_macro_alloc;
    int new_mnpm = (max_nets_per_macro > p->max_nets_per_macro_alloc) ? max_nets_per_macro : p->max_nets_per_macro_alloc;
    int new_abu = (tl_abu_cap > p->tl_abu_cap_alloc) ? tl_abu_cap : p->tl_abu_cap_alloc;

    _free_tl_pool_inner(p);

    p->n_alloc = new_n;
    p->ng_alloc = new_ng;
    p->nn_alloc = new_nn;
    p->np_alloc = new_np;
    p->max_threads_alloc = new_mt;
    p->max_pins_alloc = new_mp;
    p->max_pins_per_macro_alloc = new_mppm;
    p->max_nets_per_macro_alloc = new_mnpm;
    p->tl_abu_cap_alloc = new_abu;

    p->tl_H_net = (double **)calloc(new_mt, sizeof(double *));
    p->tl_V_net = (double **)calloc(new_mt, sizeof(double *));
    p->tl_H_final = (double **)calloc(new_mt, sizeof(double *));
    p->tl_V_final = (double **)calloc(new_mt, sizeof(double *));
    p->tl_grid_occupied = (double **)calloc(new_mt, sizeof(double *));
    p->tl_net_hpwl = (double **)calloc(new_mt, sizeof(double *));
    p->tl_pos = (double **)calloc(new_mt, sizeof(double *));
    p->tl_abu = (double **)calloc(new_mt, sizeof(double *));
    p->tl_pin_abs_x = (double **)calloc(new_mt, sizeof(double *));
    p->tl_pin_abs_y = (double **)calloc(new_mt, sizeof(double *));
    p->tl_pin_row = (int **)calloc(new_mt, sizeof(int *));
    p->tl_pin_col = (int **)calloc(new_mt, sizeof(int *));
    p->tl_seen = (unsigned int **)calloc(new_mt, sizeof(unsigned int *));
    p->tl_seen_gen = (unsigned int *)calloc(new_mt, sizeof(unsigned int));
    p->tl_rows = (int **)calloc(new_mt, sizeof(int *));
    p->tl_cols = (int **)calloc(new_mt, sizeof(int *));
    p->tl_baseline_rows = (int **)calloc(new_mt, sizeof(int *));
    p->tl_baseline_cols = (int **)calloc(new_mt, sizeof(int *));
    p->tl_net_xmin = (double **)calloc(new_mt, sizeof(double *));
    p->tl_net_xmax = (double **)calloc(new_mt, sizeof(double *));
    p->tl_net_ymin = (double **)calloc(new_mt, sizeof(double *));
    p->tl_net_ymax = (double **)calloc(new_mt, sizeof(double *));
    p->tl_net_pin_xmin = (int **)calloc(new_mt, sizeof(int *));
    p->tl_net_pin_xmax = (int **)calloc(new_mt, sizeof(int *));
    p->tl_net_pin_ymin = (int **)calloc(new_mt, sizeof(int *));
    p->tl_net_pin_ymax = (int **)calloc(new_mt, sizeof(int *));
    p->tl_base_xmin = (double **)calloc(new_mt, sizeof(double *));
    p->tl_base_xmax = (double **)calloc(new_mt, sizeof(double *));
    p->tl_base_ymin = (double **)calloc(new_mt, sizeof(double *));
    p->tl_base_ymax = (double **)calloc(new_mt, sizeof(double *));
    p->tl_base_pin_xmin = (int **)calloc(new_mt, sizeof(int *));
    p->tl_base_pin_xmax = (int **)calloc(new_mt, sizeof(int *));
    p->tl_base_pin_ymin = (int **)calloc(new_mt, sizeof(int *));
    p->tl_base_pin_ymax = (int **)calloc(new_mt, sizeof(int *));

    int sz_d_ng_b   = sizeof(double) * (new_ng > 0 ? new_ng : 1);
    int sz_d_n2_b   = sizeof(double) * (new_n  > 0 ? new_n * 2 : 2);
    int sz_d_nn_b   = sizeof(double) * (new_nn > 0 ? new_nn : 1);
    int sz_d_np_b   = sizeof(double) * (new_np > 0 ? new_np : 1);
    int sz_d_abu_b  = sizeof(double) * (new_abu > 0 ? new_abu : 1);
    int sz_d_mnpm_b = sizeof(double) * (new_mnpm > 0 ? new_mnpm : 1);
    int sz_i_np_b   = sizeof(int) * (new_np > 0 ? new_np : 1);
    int sz_i_mp_b   = sizeof(int) * (new_mp + 1);
    int sz_i_mppm_b = sizeof(int) * (new_mppm > 0 ? new_mppm : 1);
    int sz_i_nn_b   = sizeof(int) * (new_nn > 0 ? new_nn : 1);
    int sz_i_mnpm_b = sizeof(int) * (new_mnpm > 0 ? new_mnpm : 1);

    for (int t = 0; t < new_mt; t++) {
        p->tl_H_net[t]    = (double *)malloc(sz_d_ng_b);
        p->tl_V_net[t]    = (double *)malloc(sz_d_ng_b);
        p->tl_H_final[t]  = (double *)malloc(sz_d_ng_b);
        p->tl_V_final[t]  = (double *)malloc(sz_d_ng_b);
        p->tl_grid_occupied[t] = (double *)malloc(sz_d_ng_b);
        p->tl_net_hpwl[t] = (double *)malloc(sz_d_nn_b);
        p->tl_pos[t]      = (double *)malloc(sz_d_n2_b);
        p->tl_abu[t]      = (double *)malloc(sz_d_abu_b);
        p->tl_pin_abs_x[t] = (double *)malloc(sz_d_np_b);
        p->tl_pin_abs_y[t] = (double *)malloc(sz_d_np_b);
        p->tl_pin_row[t]   = (int *)malloc(sz_i_np_b);
        p->tl_pin_col[t]   = (int *)malloc(sz_i_np_b);
        /* tl_seen MUST be zeroed (gen-counter pattern depends on initial 0). */
        p->tl_seen[t]      = (unsigned int *)calloc(new_ng > 0 ? new_ng : 1, sizeof(unsigned int));
        p->tl_seen_gen[t]  = 1u;
        p->tl_rows[t]      = (int *)malloc(sz_i_mp_b);
        p->tl_cols[t]      = (int *)malloc(sz_i_mp_b);
        p->tl_baseline_rows[t] = (int *)malloc(sz_i_mppm_b);
        p->tl_baseline_cols[t] = (int *)malloc(sz_i_mppm_b);
        p->tl_net_xmin[t]  = (double *)malloc(sz_d_nn_b);
        p->tl_net_xmax[t]  = (double *)malloc(sz_d_nn_b);
        p->tl_net_ymin[t]  = (double *)malloc(sz_d_nn_b);
        p->tl_net_ymax[t]  = (double *)malloc(sz_d_nn_b);
        p->tl_net_pin_xmin[t] = (int *)malloc(sz_i_nn_b);
        p->tl_net_pin_xmax[t] = (int *)malloc(sz_i_nn_b);
        p->tl_net_pin_ymin[t] = (int *)malloc(sz_i_nn_b);
        p->tl_net_pin_ymax[t] = (int *)malloc(sz_i_nn_b);
        p->tl_base_xmin[t] = (double *)malloc(sz_d_mnpm_b);
        p->tl_base_xmax[t] = (double *)malloc(sz_d_mnpm_b);
        p->tl_base_ymin[t] = (double *)malloc(sz_d_mnpm_b);
        p->tl_base_ymax[t] = (double *)malloc(sz_d_mnpm_b);
        p->tl_base_pin_xmin[t] = (int *)malloc(sz_i_mnpm_b);
        p->tl_base_pin_xmax[t] = (int *)malloc(sz_i_mnpm_b);
        p->tl_base_pin_ymin[t] = (int *)malloc(sz_i_mnpm_b);
        p->tl_base_pin_ymax[t] = (int *)malloc(sz_i_mnpm_b);
    }
}


/* --- Main entry point ---
 * Arguments mirror Python ctypes binding in placer.py (_cong_relax_v2_c).
 */
void cong_relax_v2(
    /* positions (in/out) */
    double *pos,
    const double *sizes,
    const int *movable,
    int n, int nh,
    /* canvas/grid */
    double cw, double ch,
    int gr, int gc,
    int smooth_range,
    double hrpm, double vrpm,
    double h_alloc, double v_alloc,
    /* pins */
    const int *pin_macro,
    const double *pin_x,
    const double *pin_y,
    int np,
    /* nets */
    const int *net_driver,
    const int *net_sinks_off,
    const int *net_sinks_idx,
    const double *net_weight,
    int nn,
    /* HPWL normalization */
    double net_cnt_for_norm,  /* sum of net weights (matches plc.net_cnt) */
    /* Optional WL-only multiplier (NULL = identity = 1.0). Applied ONLY
     * to WL gradient + HPWL value, NOT to routing demand or density. */
    const double *wl_extra_weight,
    /* knobs */
    int n_iter,
    double lr_bins,    /* step size expressed in # of grid cells per iter */
    /* optional: outputs */
    double *out_cong_init,   /* may be NULL */
    double *out_cong_final,  /* may be NULL */
    double *out_proxy_init,  /* may be NULL - full proxy at start */
    double *out_proxy_final, /* may be NULL - full proxy at end */
    double *out_wl_final,    /* may be NULL - wirelength component at end */
    double *out_den_final,   /* may be NULL - density component at end */
    /* architectural knob (0/1): when set, sort soft_list once at the start by
     * descending hotspot score so that macros sitting on the highest-congestion
     * cells are optimized first. Coord-descent order matters for final-stage
     * polish - reserve for the final-cc invocation to avoid disturbing the
     * greedy trajectory of mid-cascade cc calls (which cong-c is sensitive to). */
    int hotspot_sort,
    /* Extra polish phases chained after the primary (n_iter, lr_bins) phase.
     * Each extra phase rewinds S.pos to best_pos, rebuilds maps, resets
     * soft_list order and rng, then runs its own (n_iter, lr_bins) outer loop.
     * This is path-equivalent to calling cong_relax_v2 sequentially with the
     * corresponding (n_iter, lr_bins) but avoids per-call init_state +
     * per-thread buffer allocation overhead. Pass 0 / NULL / NULL for a
     * single-phase call. */
    int n_extra_phases,
    const int *extra_n_iters,
    const double *extra_lr_bins
) {
    /* Scalar state setup. */
    State S; memset(&S, 0, sizeof(S));
    S.n = n; S.nh = nh;
    S.gr = gr; S.gc = gc; S.ng = gr * gc;
    S.cw = cw; S.ch = ch;
    S.gw = cw / gc; S.gh = ch / gr;
    S.grid_area = S.gw * S.gh;
    S.hrpm = hrpm; S.vrpm = vrpm;
    S.h_alloc = h_alloc; S.v_alloc = v_alloc;
    S.smooth_range = smooth_range;
    S.grid_h_routes = S.gh * hrpm;
    S.grid_v_routes = S.gw * vrpm;
    S.net_cnt_for_norm = net_cnt_for_norm;
    S.hpwl_norm = (cw + ch) * (net_cnt_for_norm > 0.0 ? net_cnt_for_norm : 1.0);
    S.pos = pos; S.sizes = sizes; S.movable = movable;
    S.np = np;
    S.pin_macro = pin_macro; S.pin_x = pin_x; S.pin_y = pin_y;
    S.nn = nn;
    S.net_driver = net_driver;
    S.net_sinks_off = net_sinks_off;
    S.net_sinks_idx = net_sinks_idx;
    S.net_weight = net_weight;
    S.wl_extra_weight = wl_extra_weight;

    int max_pins = 1;
    for (int i = 0; i < nn; i++) {
        int c = (net_sinks_off[i + 1] - net_sinks_off[i]) + 1;
        if (c > max_pins) max_pins = c;
    }
    S.max_pins_per_net = max_pins;

    S.H_net    = (double *)calloc(S.ng, sizeof(double));
    S.V_net    = (double *)calloc(S.ng, sizeof(double));
    S.H_macro  = (double *)calloc(S.ng, sizeof(double));
    S.V_macro  = (double *)calloc(S.ng, sizeof(double));
    S.H_final  = (double *)calloc(S.ng, sizeof(double));
    S.V_final  = (double *)calloc(S.ng, sizeof(double));
    S.pin_abs_x = (double *)calloc(np > 0 ? np : 1, sizeof(double));
    S.pin_abs_y = (double *)calloc(np > 0 ? np : 1, sizeof(double));
    S.pin_row   = (int *)calloc(np > 0 ? np : 1, sizeof(int));
    S.pin_col   = (int *)calloc(np > 0 ? np : 1, sizeof(int));
    S.mp_offsets = (int *)calloc(n + 1, sizeof(int));
    S.mp_pin_ids = (int *)calloc(np > 0 ? np : 1, sizeof(int));
    S.mn_offsets = (int *)calloc(n + 1, sizeof(int));
    /* upper bound on mn_net_ids: sum over nets of unique macros on net is
     * bounded by sum of (1 driver + sinks_count) = nn + total_sinks. */
    int total_pin_net = nn + (nn > 0 ? net_sinks_off[nn] : 0);
    S.mn_net_ids = (int *)calloc(total_pin_net + 1, sizeof(int));
    S.scratch_rows = (int *)calloc(max_pins + 1, sizeof(int));
    S.scratch_cols = (int *)calloc(max_pins + 1, sizeof(int));
    S.seen_gcell = (unsigned int *)calloc(S.ng > 0 ? S.ng : 1, sizeof(unsigned int));
    S.seen_gen = 1u;
    S.grid_occupied = (double *)calloc(S.ng, sizeof(double));
    S.net_hpwl = (double *)calloc(nn > 0 ? nn : 1, sizeof(double));
    S.net_xmin = (double *)calloc(nn > 0 ? nn : 1, sizeof(double));
    S.net_xmax = (double *)calloc(nn > 0 ? nn : 1, sizeof(double));
    S.net_ymin = (double *)calloc(nn > 0 ? nn : 1, sizeof(double));
    S.net_ymax = (double *)calloc(nn > 0 ? nn : 1, sizeof(double));
    S.net_pin_xmin = (int *)calloc(nn > 0 ? nn : 1, sizeof(int));
    S.net_pin_xmax = (int *)calloc(nn > 0 ? nn : 1, sizeof(int));
    S.net_pin_ymin = (int *)calloc(nn > 0 ? nn : 1, sizeof(int));
    S.net_pin_ymax = (int *)calloc(nn > 0 ? nn : 1, sizeof(int));
    /* abu_scratch sized for the larger of cong (top-5% of V||H = floor(2ng/20)
     * = ng/10) and density (floor(ng/10)). +2 for slack. */
    int abu_cap = S.ng / 10 + 2;
    S.abu_scratch = (double *)calloc(abu_cap, sizeof(double));

    build_macro_pin_csr(&S);
    init_state(&S);

    if (out_cong_init) *out_cong_init = S.cong_cost;
    if (out_proxy_init) *out_proxy_init = S.full_cost;

    if (n_iter <= 0 || nh >= n) {
        if (out_cong_final) *out_cong_final = S.cong_cost;
        if (out_proxy_final) *out_proxy_final = S.full_cost;
        if (out_wl_final) *out_wl_final = S.wl_cost;
        if (out_den_final) *out_den_final = S.density_cost;
        goto cleanup;
    }

    /* Snapshot best by FULL proxy (so we never regress vs the entry state).
     * We also snapshot all position-dependent maps alongside best_pos so that
     * phase-boundary restore can memcpy back into S instead of re-running
     * rebuild_maps (which re-does route_net for every net - the dominant
     * cost in init_state/rebuild_maps). H_macro/V_macro aren't snapshotted
     * because hard macros don't move during cong_relax_v2. The pin cache
     * (pin_abs_x/y, pin_row/col) is cheap to rebuild via refresh_all_pin_cache
     * after restore, so we skip snapshotting it. */
    double *best_pos, *best_H_net, *best_V_net;
    double *best_H_final, *best_V_final;
    double *best_grid_occupied, *best_net_hpwl;

    /* Per-thread scratch state for parallel candidate evaluation.
     * Each thread holds its own copy of the mutable State arrays so it can
     * apply a candidate move + revert without contention with peers. The
     * canonical state in S is only modified after a parallel region completes
     * (commit on best candidate). */
    int max_threads;
    double **tl_H_net, **tl_V_net, **tl_H_final, **tl_V_final;
    double **tl_grid_occupied, **tl_net_hpwl, **tl_pos, **tl_abu;
    double **tl_pin_abs_x, **tl_pin_abs_y;
    int **tl_pin_row, **tl_pin_col;
    unsigned int **tl_seen;
    unsigned int *tl_seen_gen;
    int **tl_rows, **tl_cols;
    /* Per-thread scratch for the route-skip fast path: snapshot of m's pin
     * gcells at baseline (cx, cy). If a candidate's refreshed pin gcells all
     * match, route_net(+1)/route_net(-1) would write identical values to
     * identical cells (route_net is gcell-determined), so we can skip both
     * routing passes and reuse S.cong_cost as the candidate's cong. */
    int **tl_baseline_rows, **tl_baseline_cols;
    /* Per-thread per-net bbox cache (T side) and the per-si snapshot of m's
     * nets at baseline (cx, cy). Snapshot is used by try_skip_net_hpwl so
     * the baseline bbox isn't clobbered by a prior candidate's fallthrough
     * compute_net_hpwl in the same si iteration. */
    double **tl_net_xmin, **tl_net_xmax, **tl_net_ymin, **tl_net_ymax;
    int **tl_net_pin_xmin, **tl_net_pin_xmax, **tl_net_pin_ymin, **tl_net_pin_ymax;
    double **tl_base_xmin, **tl_base_xmax, **tl_base_ymin, **tl_base_ymax;
    int **tl_base_pin_xmin, **tl_base_pin_xmax, **tl_base_pin_ymin, **tl_base_pin_ymax;

    /* Per-candidate result slots (each ci is owned by exactly one thread under
     * schedule(static), so writes from threads don't race; the reduction below
     * walks them in candidate order to bit-match the serial tie-breaking
     * ("first strictly-better gain wins"). We also cache the winning candidate's
     * cong/density/wl scalars so the commit path doesn't have to rerun abu. */
    int CAND_CAP;
    double *cand_gain, *cand_nx, *cand_ny;
    double *cand_new_cong, *cand_new_den, *cand_new_wl;

    /* Soft movable list */
    int *soft_list;

    /* Scratch for in-loop pair-swap passes. */
    int psw_cap = (S.nn > 0 ? S.nn : 1);
    int *sw_net_touched, *sw_net_buf;
    double *sw_saved_hp;
    int *sw_knn_cache;

    best_pos = (double *)malloc(sizeof(double) * n * 2);
    best_H_net = (double *)malloc(sizeof(double) * S.ng);
    best_V_net = (double *)malloc(sizeof(double) * S.ng);
    best_H_final = (double *)malloc(sizeof(double) * S.ng);
    best_V_final = (double *)malloc(sizeof(double) * S.ng);
    best_grid_occupied = (double *)malloc(sizeof(double) * S.ng);
    best_net_hpwl = (double *)malloc(sizeof(double) * (nn > 0 ? nn : 1));

    max_threads = 1;
#ifdef _OPENMP
    max_threads = omp_get_max_threads();
#endif
    int tl_abu_cap = S.ng / 10 + 2;
    int max_pins_per_macro = 1;
    for (int mi = 0; mi < n; mi++) {
        int cnt = S.mp_offsets[mi + 1] - S.mp_offsets[mi];
        if (cnt > max_pins_per_macro) max_pins_per_macro = cnt;
    }
    int max_nets_per_macro = 1;
    for (int mi = 0; mi < n; mi++) {
        int cnt = S.mn_offsets[mi + 1] - S.mn_offsets[mi];
        if (cnt > max_nets_per_macro) max_nets_per_macro = cnt;
    }
    /* Use cached per-thread scratch - saves ~277 ms of malloc/free per call. */
    _ensure_tl_pool(n, S.ng, nn, np, max_threads, max_pins,
                     max_pins_per_macro, max_nets_per_macro, tl_abu_cap);
    tl_H_net = _tl_pool.tl_H_net;
    tl_V_net = _tl_pool.tl_V_net;
    tl_H_final = _tl_pool.tl_H_final;
    tl_V_final = _tl_pool.tl_V_final;
    tl_grid_occupied = _tl_pool.tl_grid_occupied;
    tl_net_hpwl = _tl_pool.tl_net_hpwl;
    tl_pos = _tl_pool.tl_pos;
    tl_abu = _tl_pool.tl_abu;
    tl_pin_abs_x = _tl_pool.tl_pin_abs_x;
    tl_pin_abs_y = _tl_pool.tl_pin_abs_y;
    tl_pin_row = _tl_pool.tl_pin_row;
    tl_pin_col = _tl_pool.tl_pin_col;
    tl_seen = _tl_pool.tl_seen;
    tl_seen_gen = _tl_pool.tl_seen_gen;
    tl_rows = _tl_pool.tl_rows;
    tl_cols = _tl_pool.tl_cols;
    tl_baseline_rows = _tl_pool.tl_baseline_rows;
    tl_baseline_cols = _tl_pool.tl_baseline_cols;
    tl_net_xmin = _tl_pool.tl_net_xmin;
    tl_net_xmax = _tl_pool.tl_net_xmax;
    tl_net_ymin = _tl_pool.tl_net_ymin;
    tl_net_ymax = _tl_pool.tl_net_ymax;
    tl_net_pin_xmin = _tl_pool.tl_net_pin_xmin;
    tl_net_pin_xmax = _tl_pool.tl_net_pin_xmax;
    tl_net_pin_ymin = _tl_pool.tl_net_pin_ymin;
    tl_net_pin_ymax = _tl_pool.tl_net_pin_ymax;
    tl_base_xmin = _tl_pool.tl_base_xmin;
    tl_base_xmax = _tl_pool.tl_base_xmax;
    tl_base_ymin = _tl_pool.tl_base_ymin;
    tl_base_ymax = _tl_pool.tl_base_ymax;
    tl_base_pin_xmin = _tl_pool.tl_base_pin_xmin;
    tl_base_pin_xmax = _tl_pool.tl_base_pin_xmax;
    tl_base_pin_ymin = _tl_pool.tl_base_pin_ymin;
    tl_base_pin_ymax = _tl_pool.tl_base_pin_ymax;

    CAND_CAP = 24;
    cand_gain     = (double *)malloc(sizeof(double) * CAND_CAP);
    cand_nx       = (double *)malloc(sizeof(double) * CAND_CAP);
    cand_ny       = (double *)malloc(sizeof(double) * CAND_CAP);
    cand_new_cong = (double *)malloc(sizeof(double) * CAND_CAP);
    cand_new_den  = (double *)malloc(sizeof(double) * CAND_CAP);
    cand_new_wl   = (double *)malloc(sizeof(double) * CAND_CAP);

    soft_list = (int *)malloc(sizeof(int) * n);
    sw_net_touched = (int *)calloc(psw_cap, sizeof(int));
    sw_net_buf     = (int *)malloc(sizeof(int) * psw_cap);
    sw_saved_hp    = (double *)malloc(sizeof(double) * psw_cap);
    /* sized by n (upper bound on n_soft). */
    sw_knn_cache   = (int *)malloc(sizeof(int) * PSW_K_NEAR * (n > 0 ? n : 1));

    /* Hot-region scratch (Tier-1 polish improvements):
     *   heat_items: (heat, macro_id) pairs, sorted descending each outer round.
     *   cold_mask: per-soft index, 1 means "skip in iters >= 1 of this outer
     *              round" (had 0 accepts in iter 0 -> cold). Reset each outer.
     *   per_macro_acc_iter0: accept count per soft index in iter 0 of an
     *                        outer round; used to populate cold_mask after
     *                        iter 0 ends.
     *   accepts_history: small ring of recent per-iter accept counts for the
     *                    convergence early-stop (stop phase if the last 3
     *                    iters all had <5% acceptance).
     */
    HeatItem *heat_items = (HeatItem *)malloc(
        sizeof(HeatItem) * (n > 0 ? n : 1));
    int *cold_mask = (int *)calloc((n > 0 ? n : 1), sizeof(int));
    int *per_macro_acc_iter0 = (int *)calloc((n > 0 ? n : 1), sizeof(int));
    int accepts_history[3] = {0, 0, 0};

    memcpy(best_pos, pos, sizeof(double) * n * 2);
    memcpy(best_H_net, S.H_net, sizeof(double) * S.ng);
    memcpy(best_V_net, S.V_net, sizeof(double) * S.ng);
    memcpy(best_H_final, S.H_final, sizeof(double) * S.ng);
    memcpy(best_V_final, S.V_final, sizeof(double) * S.ng);
    memcpy(best_grid_occupied, S.grid_occupied, sizeof(double) * S.ng);
    memcpy(best_net_hpwl, S.net_hpwl, sizeof(double) * (nn > 0 ? nn : 1));
    double best_proxy = S.full_cost;
    double best_cong = S.cong_cost;
    double best_wl = S.wl_cost;
    double best_den = S.density_cost;
    double best_total_hpwl = S.total_hpwl;

    int n_soft = 0;
    for (int i = nh; i < n; i++) if (movable[i]) soft_list[n_soft++] = i;
    (void)hotspot_sort;  /* reserved for future ordering experiments */

    /* Hard-macro spatial index: hard macros are static through cong_relax_v2,
     * so the per-candidate overlap test (currently O(nh)=301 per check) can be
     * reduced to O(few) via a coarse 2D grid bucketed by hard centroid. The
     * test logic and FP comparisons are unchanged - only the iteration set
     * shrinks. Bucket pitch = 2*max_hard_half_size in each axis so a candidate
     * needs to scan only ~ceil(2*hw / pitch + 1) buckets per axis. */
    double max_hard_hw = 0.0, max_hard_hh = 0.0;
    for (int j = 0; j < nh; j++) {
        double whj = sizes[j*2 + 0] * 0.5;
        double hhj = sizes[j*2 + 1] * 0.5;
        if (whj > max_hard_hw) max_hard_hw = whj;
        if (hhj > max_hard_hh) max_hard_hh = hhj;
    }
    double hb_bsx = (max_hard_hw > 0.0) ? (2.0 * max_hard_hw) : cw;
    double hb_bsy = (max_hard_hh > 0.0) ? (2.0 * max_hard_hh) : ch;
    int hb_bnx = (int)(cw / hb_bsx) + 1;
    int hb_bny = (int)(ch / hb_bsy) + 1;
    if (hb_bnx < 1) hb_bnx = 1;
    if (hb_bny < 1) hb_bny = 1;
    int hb_nbins = hb_bnx * hb_bny;
    int *hb_offsets = (int *)calloc(hb_nbins + 1, sizeof(int));
    int *hb_macros  = (int *)malloc(sizeof(int) * (nh > 0 ? nh : 1));
    {
        int *hb_cursor = (int *)calloc(hb_nbins, sizeof(int));
        for (int j = 0; j < nh; j++) {
            double jx = pos[j*2 + 0], jy = pos[j*2 + 1];
            int bx = (int)(jx / hb_bsx); if (bx < 0) bx = 0; if (bx >= hb_bnx) bx = hb_bnx - 1;
            int by = (int)(jy / hb_bsy); if (by < 0) by = 0; if (by >= hb_bny) by = hb_bny - 1;
            hb_offsets[by * hb_bnx + bx + 1]++;
        }
        for (int b = 1; b <= hb_nbins; b++) hb_offsets[b] += hb_offsets[b - 1];
        for (int j = 0; j < nh; j++) {
            double jx = pos[j*2 + 0], jy = pos[j*2 + 1];
            int bx = (int)(jx / hb_bsx); if (bx < 0) bx = 0; if (bx >= hb_bnx) bx = hb_bnx - 1;
            int by = (int)(jy / hb_bsy); if (by < 0) by = 0; if (by >= hb_bny) by = hb_bny - 1;
            int b = by * hb_bnx + bx;
            hb_macros[hb_offsets[b] + hb_cursor[b]++] = j;
        }
        free(hb_cursor);
    }

    double step = lr_bins;   /* in bins */
    double eps = 1e-9;
    /* Reheat: when the step schedule decays below 0.25, reset it to the
     * starting value and continue - the expanded candidate set explores
     * different basins after each reheat because the blended/component
     * directions shift as S moves. Capped to keep total work bounded. */
    int reheats = 0;
    const int max_reheats = 3;

    /* xorshift32 seed for shuffle on reheat (keeps deterministic order
     * in the warm phase, then perturbs order after each anneal-in to break
     * out of repeated basins). */
    unsigned int rng = 0x9E3779B1u;
    (void)rng;

    /* Outer alternation: translation loop, then pair-swap pass, repeat.
     * Pair-swap exposes new translation moves (frees blocked neighbors)
     * and translation reorganizes around the swap, so each round can find
     * improvements the other couldn't. */
    const int n_outer = 2;
    const int max_sw_passes_inner = 6;
    int n_phases = 1 + (n_extra_phases > 0 ? n_extra_phases : 0);

    /* Hoisted control vars: shared across the parallel region so that the
     * single-thread updates inside `omp single` blocks are visible to all
     * threads after the implicit barrier. */
    int accepts = 0;
    int do_break = 0;
    int sh_best_ci = -1;
    double sh_best_nx = 0.0, sh_best_ny = 0.0;
    int sw_acc_shared = 0;

    /* One parallel region for the entire optimization. Each thread maintains
     * a State T pinned to its private tl_* buffers; T is synced to S exactly
     * once at start (and again only on phase-rewind / pair-swap boundaries),
     * not per-soft-macro as before. Per-si accepts replay on every thread's
     * T so it stays bit-identical to S.
     *
     * Right-sized team: with OMP_NUM_THREADS=16 and typical nc~8-16, half the
     * team idles on the candidate omp-for. Reducing to 8 keeps every thread
     * busy on the common case and frees L2 capacity per thread (each State T
     * holds 5 ng-sized buffers ≈ 50KB; 16 threads thrash L2). */
#ifdef _OPENMP
    int desired_threads = max_threads;
    if (desired_threads > 8) desired_threads = 8;
    #pragma omp parallel num_threads(desired_threads)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        State T = S;
        T.H_net = tl_H_net[tid];
        T.V_net = tl_V_net[tid];
        T.H_final = tl_H_final[tid];
        T.V_final = tl_V_final[tid];
        T.grid_occupied = tl_grid_occupied[tid];
        T.net_hpwl = tl_net_hpwl[tid];
        T.pos = tl_pos[tid];
        T.abu_scratch = tl_abu[tid];
        T.pin_abs_x = tl_pin_abs_x[tid];
        T.pin_abs_y = tl_pin_abs_y[tid];
        T.pin_row = tl_pin_row[tid];
        T.pin_col = tl_pin_col[tid];
        T.seen_gcell = tl_seen[tid];
        T.seen_gen = tl_seen_gen[tid];
        T.scratch_rows = tl_rows[tid];
        T.scratch_cols = tl_cols[tid];
        T.net_xmin = tl_net_xmin[tid];
        T.net_xmax = tl_net_xmax[tid];
        T.net_ymin = tl_net_ymin[tid];
        T.net_ymax = tl_net_ymax[tid];
        T.net_pin_xmin = tl_net_pin_xmin[tid];
        T.net_pin_xmax = tl_net_pin_xmax[tid];
        T.net_pin_ymin = tl_net_pin_ymin[tid];
        T.net_pin_ymax = tl_net_pin_ymax[tid];

#define SYNC_T_FROM_S()                                                        \
    do {                                                                       \
        memcpy(T.H_net, S.H_net, sizeof(double) * S.ng);                       \
        memcpy(T.V_net, S.V_net, sizeof(double) * S.ng);                       \
        memcpy(T.H_final, S.H_final, sizeof(double) * S.ng);                   \
        memcpy(T.V_final, S.V_final, sizeof(double) * S.ng);                   \
        memcpy(T.grid_occupied, S.grid_occupied, sizeof(double) * S.ng);       \
        memcpy(T.net_hpwl, S.net_hpwl,                                         \
               sizeof(double) * (S.nn > 0 ? S.nn : 1));                        \
        memcpy(T.pos, S.pos, sizeof(double) * S.n * 2);                        \
        memcpy(T.pin_abs_x, S.pin_abs_x,                                       \
               sizeof(double) * (S.np > 0 ? S.np : 1));                        \
        memcpy(T.pin_abs_y, S.pin_abs_y,                                       \
               sizeof(double) * (S.np > 0 ? S.np : 1));                        \
        memcpy(T.pin_row, S.pin_row,                                           \
               sizeof(int) * (S.np > 0 ? S.np : 1));                           \
        memcpy(T.pin_col, S.pin_col,                                           \
               sizeof(int) * (S.np > 0 ? S.np : 1));                           \
        memcpy(T.net_xmin, S.net_xmin,                                         \
               sizeof(double) * (S.nn > 0 ? S.nn : 1));                        \
        memcpy(T.net_xmax, S.net_xmax,                                         \
               sizeof(double) * (S.nn > 0 ? S.nn : 1));                        \
        memcpy(T.net_ymin, S.net_ymin,                                         \
               sizeof(double) * (S.nn > 0 ? S.nn : 1));                        \
        memcpy(T.net_ymax, S.net_ymax,                                         \
               sizeof(double) * (S.nn > 0 ? S.nn : 1));                        \
        memcpy(T.net_pin_xmin, S.net_pin_xmin,                                 \
               sizeof(int) * (S.nn > 0 ? S.nn : 1));                           \
        memcpy(T.net_pin_xmax, S.net_pin_xmax,                                 \
               sizeof(int) * (S.nn > 0 ? S.nn : 1));                           \
        memcpy(T.net_pin_ymin, S.net_pin_ymin,                                 \
               sizeof(int) * (S.nn > 0 ? S.nn : 1));                           \
        memcpy(T.net_pin_ymax, S.net_pin_ymax,                                 \
               sizeof(int) * (S.nn > 0 ? S.nn : 1));                           \
        T.total_hpwl = S.total_hpwl;                                           \
        T.cong_cost = S.cong_cost;                                             \
        T.density_cost = S.density_cost;                                       \
        T.wl_cost = S.wl_cost;                                                 \
        T.full_cost = S.full_cost;                                             \
    } while (0)

        SYNC_T_FROM_S();

    for (int ph = 0; ph < n_phases; ph++) {
        int cur_n_iter = (ph == 0) ? n_iter : extra_n_iters[ph - 1];
        double cur_lr_bins = (ph == 0) ? lr_bins : extra_lr_bins[ph - 1];

        /* Track best_proxy at phase start so end-of-phase snapshot can be
         * skipped if the phase made no improvement. */
        double phase_start_proxy = best_proxy;

        if (ph > 0) {
            /* Rewind to best-of-prior-phase. We snapshotted all position-
             * dependent maps alongside best_pos on every update, so restore is
             * a handful of memcpys instead of rebuild_maps (which re-runs
             * route_net for every net). The pin cache is cheap to rebuild via
             * refresh_all_pin_cache. Also reset soft_list order and rng to
             * match a fresh call's initial state, path-equivalent to a
             * separate cong_relax_v2 call. */
#ifdef _OPENMP
            #pragma omp single
#endif
            {
                memcpy(pos, best_pos, sizeof(double) * n * 2);
                memcpy(S.H_net, best_H_net, sizeof(double) * S.ng);
                memcpy(S.V_net, best_V_net, sizeof(double) * S.ng);
                memcpy(S.H_final, best_H_final, sizeof(double) * S.ng);
                memcpy(S.V_final, best_V_final, sizeof(double) * S.ng);
                memcpy(S.grid_occupied, best_grid_occupied, sizeof(double) * S.ng);
                memcpy(S.net_hpwl, best_net_hpwl, sizeof(double) * (nn > 0 ? nn : 1));
                S.total_hpwl = best_total_hpwl;
                S.cong_cost = best_cong;
                S.wl_cost = best_wl;
                S.density_cost = best_den;
                S.full_cost = best_proxy;
                refresh_all_pin_cache(&S);
                n_soft = 0;
                for (int i = nh; i < n; i++) if (movable[i]) soft_list[n_soft++] = i;
                rng = 0x9E3779B1u;
            } /* implicit barrier */
            SYNC_T_FROM_S();
        }

    for (int outer = 0; outer < n_outer; outer++) {
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            step = cur_lr_bins;
            reheats = 0;
            /* Tier-1 polish: hot-region sort + cold-mask reset.
             * hotspot_sort=1 means "sort soft_list descending by current
             * cong+density heat" so hot macros are processed first each iter.
             * cold_mask is reset each outer round; macros that produce 0
             * accepts in iter 0 are marked cold and skipped in iters >= 1.
             * accepts_history is the rolling window for phase early-stop. */
            /* Hot-sort only fires for long phases (n_iter >= 10) - short
             * worker phases (n_iter=6) regress slightly when reordered
             * because the shorter iter budget can't recover from the
             * trajectory shift. Init polish phase 0 (n_iter=12) is the
             * primary beneficiary. */
            if (hotspot_sort && cur_n_iter >= 10) {
                /* alpha_den balances density vs cong contributions; matches
                 * the proxy weighting (both *0.5* in the proxy formula, and
                 * inv_ba scales density to a comparable magnitude). */
                sort_soft_by_heat(&S, soft_list, n_soft, heat_items, 1.0);
            }
            for (int si = 0; si < n_soft; si++) {
                cold_mask[si] = 0;
                per_macro_acc_iter0[si] = 0;
            }
            accepts_history[0] = 0;
            accepts_history[1] = 0;
            accepts_history[2] = 0;
        }
        /* implicit barrier - soft_list/cold_mask visible to all threads */

    for (int it = 0; it < cur_n_iter; it++) {
#ifdef _OPENMP
        #pragma omp single
#endif
        { accepts = 0; }
        /* implicit barrier */

        for (int si = 0; si < n_soft; si++) {
            /* Tier-1 polish: cold-macro pruning. After iter 0 of this outer
             * round, any macro whose iter-0 produced 0 accepts is marked
             * cold and skipped. Saves ~30-40% of iterations on later passes
             * with neutral-or-positive proxy. */
            if (hotspot_sort && it > 0 && cold_mask[si]) continue;

            /* Replicated per-thread compute - cheap scalar/CSR reads on stable
             * S, no race because S is read-only between accepts. */
            int m = soft_list[si];
            double cx = S.pos[m*2 + 0];
            double cy = S.pos[m*2 + 1];
            double hw = S.sizes[m*2 + 0] * 0.5;
            double hh = S.sizes[m*2 + 1] * 0.5;

            double cands[24][2] = {
                { +step, 0.0 },    { -step, 0.0 },
                { 0.0, +step },    { 0.0, -step },
                { +step, +step },  { -step, +step },
                { +step, -step },  { -step, -step },
            };
            int nc = 8;
            double dirs[8]; compute_dirs(&S, m, dirs);
            const int CMAX = 24;
            const double blended_mults[4] = { 1.0, 2.0, 4.0, 8.0 };
            const double comp_mults[2]    = { 1.0, 2.0 };
            for (int di = 0; di < 4 && nc < CMAX; di++) {
                double sx = dirs[2*di], sy = dirs[2*di + 1];
                if (sx * sx + sy * sy <= 1e-12) continue;
                const double *mults = (di == 0) ? blended_mults : comp_mults;
                int nm = (di == 0) ? 4 : 2;
                for (int mi = 0; mi < nm && nc < CMAX; mi++) {
                    cands[nc][0] = step * mults[mi] * sx;
                    cands[nc][1] = step * mults[mi] * sy;
                    nc++;
                }
            }

            /* Per-thread fast path: take m out of T lazily on the first valid
             * candidate (`my_did_eval` flag), then run `set_pos + in + eval +
             * out` per candidate (one routing pair instead of two). After the
             * omp-for, only threads that actually evaluated put m back at
             * (cx, cy). Threads with no valid candidates pay zero overhead. */
            int my_did_eval = 0;

            /* Snapshot m's pin gcells at baseline (cx, cy). T.pin_row/col
             * currently reflects m at cx,cy (invariant maintained by init,
             * SYNC_T_FROM_S, and each prior si's replay/post-loop restore).
             * If a candidate's refreshed gcells all match, route_net(+1) at
             * new pos would write bit-identical deltas to identical cells as
             * it would at baseline, so the route_in/out pair is a no-op in
             * effect on H_net/V_net/H_final/V_final - skip it entirely. */
            int *baseline_rows = tl_baseline_rows[tid];
            int *baseline_cols = tl_baseline_cols[tid];
            int mp_off_m = S.mp_offsets[m];
            int mp_end_m = S.mp_offsets[m + 1];
            int mp_npins_m = mp_end_m - mp_off_m;
            for (int k = 0; k < mp_npins_m; k++) {
                int p = S.mp_pin_ids[mp_off_m + k];
                baseline_rows[k] = T.pin_row[p];
                baseline_cols[k] = T.pin_col[p];
            }
            /* Snapshot baseline per-net bbox for m's nets. Needed because the
             * slow-path's compute_net_hpwl call inside the candidate loop
             * overwrites T.net_xmin/xmax/etc., so subsequent candidates can't
             * read T's cached bbox to validate a skip. */
            double *base_xmin = tl_base_xmin[tid];
            double *base_xmax = tl_base_xmax[tid];
            double *base_ymin = tl_base_ymin[tid];
            double *base_ymax = tl_base_ymax[tid];
            int *base_pin_xmin = tl_base_pin_xmin[tid];
            int *base_pin_xmax = tl_base_pin_xmax[tid];
            int *base_pin_ymin = tl_base_pin_ymin[tid];
            int *base_pin_ymax = tl_base_pin_ymax[tid];
            int mn_off_m_b = T.mn_offsets[m];
            int mn_end_m_b = T.mn_offsets[m + 1];
            int mn_cnt_m_b = mn_end_m_b - mn_off_m_b;
            for (int k = 0; k < mn_cnt_m_b; k++) {
                int ni = T.mn_net_ids[mn_off_m_b + k];
                base_xmin[k] = T.net_xmin[ni];
                base_xmax[k] = T.net_xmax[ni];
                base_ymin[k] = T.net_ymin[ni];
                base_ymax[k] = T.net_ymax[ni];
                base_pin_xmin[k] = T.net_pin_xmin[ni];
                base_pin_xmax[k] = T.net_pin_xmax[ni];
                base_pin_ymin[k] = T.net_pin_ymin[ni];
                base_pin_ymax[k] = T.net_pin_ymax[ni];
            }
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int ci = 0; ci < nc; ci++) {
                cand_gain[ci] = -1e300;  /* default - overwrite on valid eval */

                double nx = cx + cands[ci][0] * S.gw;
                double ny = cy + cands[ci][1] * S.gh;
                if (nx < hw) nx = hw;
                if (nx > S.cw - hw) nx = S.cw - hw;
                if (ny < hh) ny = hh;
                if (ny > S.ch - hh) ny = S.ch - hh;
                if (d_abs(nx - cx) < 1e-9 && d_abs(ny - cy) < 1e-9) continue;

                bool bad = false;
                {
                    double qlo_x = nx - hw - max_hard_hw;
                    double qhi_x = nx + hw + max_hard_hw;
                    double qlo_y = ny - hh - max_hard_hh;
                    double qhi_y = ny + hh + max_hard_hh;
                    int bx_lo = (int)(qlo_x / hb_bsx); if (bx_lo < 0) bx_lo = 0;
                    int bx_hi = (int)(qhi_x / hb_bsx); if (bx_hi >= hb_bnx) bx_hi = hb_bnx - 1;
                    int by_lo = (int)(qlo_y / hb_bsy); if (by_lo < 0) by_lo = 0;
                    int by_hi = (int)(qhi_y / hb_bsy); if (by_hi >= hb_bny) by_hi = hb_bny - 1;
                    for (int by = by_lo; by <= by_hi && !bad; by++) {
                        for (int bx = bx_lo; bx <= bx_hi && !bad; bx++) {
                            int b = by * hb_bnx + bx;
                            int boff = hb_offsets[b], bend = hb_offsets[b + 1];
                            for (int kk = boff; kk < bend; kk++) {
                                int j = hb_macros[kk];
                                double jx = S.pos[j*2+0], jy = S.pos[j*2+1];
                                double sepx = hw + S.sizes[j*2+0] * 0.5;
                                double sepy = hh + S.sizes[j*2+1] * 0.5;
                                if (d_abs(nx - jx) < sepx - 1e-6 &&
                                    d_abs(ny - jy) < sepy - 1e-6) { bad = true; break; }
                            }
                        }
                    }
                }
                if (bad) continue;

                if (!my_did_eval) {
                    move_macro_out(&T, m);
                    my_did_eval = 1;
                }

                T.pos[m*2+0] = nx; T.pos[m*2+1] = ny;
                refresh_macro_pin_cache(&T, m);

                /* Fast path: all of m's pins landed in the same gcells as
                 * baseline. Skip both route_net passes (they'd cancel bit-
                 * identically anyway). Still need hpwl (pin abs pos changed)
                 * and density (overlap area depends on center pos). */
                int all_match = 1;
                for (int k = 0; k < mp_npins_m; k++) {
                    int p = S.mp_pin_ids[mp_off_m + k];
                    if (T.pin_row[p] != baseline_rows[k] ||
                        T.pin_col[p] != baseline_cols[k]) {
                        all_match = 0;
                        break;
                    }
                }

                double new_cong, new_density, new_wl;
                if (all_match) {
                    int mn_off_m = T.mn_offsets[m];
                    int mn_end_m = T.mn_offsets[m + 1];
                    for (int k = mn_off_m; k < mn_end_m; k++) {
                        int ni = T.mn_net_ids[k];
                        int kb = k - mn_off_m;
                        double h;
                        if (!try_skip_net_hpwl(&T, ni, m,
                                base_xmin[kb], base_xmax[kb],
                                base_ymin[kb], base_ymax[kb],
                                base_pin_xmin[kb], base_pin_xmax[kb],
                                base_pin_ymin[kb], base_pin_ymax[kb],
                                &h)) {
                            h = compute_net_hpwl(&T, ni);
                        }
                        T.net_hpwl[ni] = h;
                        T.total_hpwl += h;
                    }
                    density_macro(&T, m, +1.0);
                    new_cong    = S.cong_cost;
                    new_density = compute_density(&T);
                    new_wl      = (T.hpwl_norm > 0.0)
                        ? (T.total_hpwl / T.hpwl_norm) : 0.0;
                    /* Undo to restore T to "m absent" baseline for next ci. */
                    for (int k = mn_off_m; k < mn_end_m; k++) {
                        int ni = T.mn_net_ids[k];
                        T.total_hpwl -= T.net_hpwl[ni];
                    }
                    density_macro(&T, m, -1.0);
                } else {
                    /* Inlined move_macro_in but using try_skip_net_hpwl so we
                     * can reuse cached bbox when m's pins land inside it. */
                    int mn_off_m = T.mn_offsets[m];
                    int mn_end_m = T.mn_offsets[m + 1];
                    for (int k = mn_off_m; k < mn_end_m; k++) {
                        int ni = T.mn_net_ids[k];
                        route_net(&T, ni, +1.0);
                        int kb = k - mn_off_m;
                        double h;
                        if (!try_skip_net_hpwl(&T, ni, m,
                                base_xmin[kb], base_xmax[kb],
                                base_ymin[kb], base_ymax[kb],
                                base_pin_xmin[kb], base_pin_xmax[kb],
                                base_pin_ymin[kb], base_pin_ymax[kb],
                                &h)) {
                            h = compute_net_hpwl(&T, ni);
                        }
                        T.net_hpwl[ni] = h;
                        T.total_hpwl += h;
                    }
                    density_macro(&T, m, +1.0);
                    new_cong    = compute_cong(&T);
                    new_density = compute_density(&T);
                    new_wl      = (T.hpwl_norm > 0.0)
                        ? (T.total_hpwl / T.hpwl_norm) : 0.0;
                    /* Take m back out at the just-evaluated position (route_net
                     * cancels exactly cell-by-cell). T returns to the "m absent"
                     * baseline established by the lazy initial move_macro_out. */
                    move_macro_out(&T, m);
                }

                double new_full = new_wl + 0.5 * new_density + 0.5 * new_cong;
                cand_gain[ci]     = S.full_cost - new_full;
                cand_nx[ci]       = nx;
                cand_ny[ci]       = ny;
                cand_new_cong[ci] = new_cong;
                cand_new_den[ci]  = new_density;
                cand_new_wl[ci]   = new_wl;
            }
            /* implicit barrier at end of omp for - no race with the omp single
             * that follows, even though threads now restore in parallel. */
            if (my_did_eval) {
                T.pos[m*2+0] = cx; T.pos[m*2+1] = cy;
                refresh_macro_pin_cache(&T, m);
                move_macro_in(&T, m);
            }

#ifdef _OPENMP
            #pragma omp single
#endif
            {
                /* Walk candidates in order - first strictly-better wins. */
                sh_best_ci = -1;
                double best_gain = 0.0;
                sh_best_nx = cx; sh_best_ny = cy;
                for (int ci = 0; ci < nc; ci++) {
                    if (cand_gain[ci] > best_gain + eps) {
                        best_gain = cand_gain[ci];
                        sh_best_nx = cand_nx[ci];
                        sh_best_ny = cand_ny[ci];
                        sh_best_ci = ci;
                    }
                }

                if (sh_best_ci >= 0) {
                    move_macro_out(&S, m);
                    S.pos[m*2+0] = sh_best_nx; S.pos[m*2+1] = sh_best_ny;
                    refresh_macro_pin_cache(&S, m);
                    move_macro_in(&S, m);
                    S.cong_cost    = cand_new_cong[sh_best_ci];
                    S.density_cost = cand_new_den[sh_best_ci];
                    S.wl_cost      = cand_new_wl[sh_best_ci];
                    S.full_cost    = S.wl_cost + 0.5 * S.density_cost + 0.5 * S.cong_cost;
                    /* Within a phase, S.full_cost is monotonically decreasing
                     * (each accept requires cand_gain > eps), so per-accept
                     * scalar tracking is sufficient - the bulk best_*
                     * snapshot of (H_net, V_net, H_final, V_final,
                     * grid_occupied, net_hpwl, pos) is deferred to phase end.
                     * Saves a ~116KB memcpy on each of ~1k accepts/place,
                     * inside an omp single block that stalls all threads. */
                    if (S.full_cost < best_proxy) {
                        best_proxy = S.full_cost;
                        best_cong = S.cong_cost;
                        best_wl = S.wl_cost;
                        best_den = S.density_cost;
                        best_total_hpwl = S.total_hpwl;
                    }
                    accepts++;
                    /* Tier-1 polish: track per-macro accepts in iter 0 for
                     * cold-mask population at end of iter 0. */
                    if (hotspot_sort && it == 0) {
                        per_macro_acc_iter0[si]++;
                    }
                }
            } /* implicit barrier - sh_best_* now visible to all */

            /* Replay accepted move on every thread's T so it stays in sync
             * with S without re-memcpying state. The move/refresh sequence
             * is deterministic, so post-replay T is bit-identical to S. */
            if (sh_best_ci >= 0) {
                move_macro_out(&T, m);
                T.pos[m*2+0] = sh_best_nx; T.pos[m*2+1] = sh_best_ny;
                refresh_macro_pin_cache(&T, m);
                move_macro_in(&T, m);
                T.cong_cost    = cand_new_cong[sh_best_ci];
                T.density_cost = cand_new_den[sh_best_ci];
                T.wl_cost      = cand_new_wl[sh_best_ci];
                T.full_cost    = T.wl_cost + 0.5 * T.density_cost + 0.5 * T.cong_cost;
            }
        }  /* end si */

#ifdef _OPENMP
        #pragma omp single
#endif
        {
            do_break = 0;
            /* Tier-1 polish: cold-mask population, gated to long phases.
             * Only mark macros cold for short-schedule phases (n_iter < 10)
             * regresses ibm08's worker output: macros that "wake up" in
             * iter 2-5 after neighbor moves get unfairly skipped. Restrict
             * to phases with enough iter-budget for the prune to amortize. */
            if (hotspot_sort && cur_n_iter >= 10 && it == 0) {
                int cold = 0;
                for (int si = 0; si < n_soft; si++) {
                    if (per_macro_acc_iter0[si] == 0) {
                        cold_mask[si] = 1;
                        cold++;
                    }
                }
                /* If essentially everyone was cold, don't prune - the
                 * outer is too early in its cooldown; let phase decay run. */
                if (cold > n_soft - 4) {
                    for (int si = 0; si < n_soft; si++) cold_mask[si] = 0;
                }
            }
            /* Roll the accepts history (oldest dropped). */
            accepts_history[2] = accepts_history[1];
            accepts_history[1] = accepts_history[0];
            accepts_history[0] = accepts;

            if (accepts == 0) {
                step *= 0.5;
                if (step < 0.25) {
                    if (reheats < max_reheats) {
                        step = cur_lr_bins;
                        reheats++;
                        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                        unsigned int seed = rng + (unsigned int)reheats * 0x9E3779B1u;
                        for (int i = n_soft - 1; i > 0; i--) {
                            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                            int j = (int)(seed % (unsigned int)(i + 1));
                            int tmp = soft_list[i];
                            soft_list[i] = soft_list[j];
                            soft_list[j] = tmp;
                        }
                    } else {
                        do_break = 1;
                    }
                }
            } else if (hotspot_sort && cur_n_iter >= 10 && it >= 3) {
                /* Tier-1 polish: phase early-stop on convergence stall.
                 * Same gate as cold-pruning - only fire on long phases. If
                 * accept rate (accepts / n_soft) has been < 5% for 3 con-
                 * secutive iters, break out early. */
                int thresh = (n_soft + 19) / 20;  /* 5% of n_soft, rounded up */
                if (thresh < 1) thresh = 1;
                if (accepts_history[0] < thresh
                    && accepts_history[1] < thresh
                    && accepts_history[2] < thresh) {
                    do_break = 1;
                }
            }
        } /* implicit barrier - do_break/step/reheats/soft_list visible */
        if (do_break) break;

    }  /* end it */

        /* Pair-swap refinement at end of each outer round. */
        for (int sw_it = 0; sw_it < max_sw_passes_inner; sw_it++) {
#ifdef _OPENMP
            #pragma omp single
#endif
            {
                sw_acc_shared = pair_swap_pass(&S, soft_list, n_soft, eps,
                                               sw_net_touched, sw_net_buf,
                                               sw_saved_hp, sw_knn_cache);
                /* pair_swap_pass also strictly improves (line 1155 gate),
                 * so the best_* arrays are updated in the deferred end-of-
                 * phase snapshot below. Track scalars here for the rewind
                 * conditional. */
                if (S.full_cost < best_proxy) {
                    best_proxy = S.full_cost;
                    best_cong = S.cong_cost;
                    best_wl = S.wl_cost;
                    best_den = S.density_cost;
                    best_total_hpwl = S.total_hpwl;
                }
            } /* implicit barrier */
            /* pair_swap_pass mutated S extensively; resync each thread's T. */
            SYNC_T_FROM_S();
            if (sw_acc_shared == 0) break;
        }
    }  /* end outer loop */

        /* Deferred phase-end snapshot. Replaces ~1k per-accept memcpys
         * (each ~116KB inside an omp single block) with one bulk copy when
         * the phase actually improved. Bit-exact equivalent because S's
         * full_cost decreases monotonically inside a phase. */
#ifdef _OPENMP
        #pragma omp single
#endif
        {
            if (best_proxy < phase_start_proxy - eps) {
                memcpy(best_pos, pos, sizeof(double) * n * 2);
                memcpy(best_H_net, S.H_net, sizeof(double) * S.ng);
                memcpy(best_V_net, S.V_net, sizeof(double) * S.ng);
                memcpy(best_H_final, S.H_final, sizeof(double) * S.ng);
                memcpy(best_V_final, S.V_final, sizeof(double) * S.ng);
                memcpy(best_grid_occupied, S.grid_occupied,
                       sizeof(double) * S.ng);
                memcpy(best_net_hpwl, S.net_hpwl,
                       sizeof(double) * (nn > 0 ? nn : 1));
            }
        } /* implicit barrier */
    }  /* end phase loop */

        tl_seen_gen[tid] = T.seen_gen;
#undef SYNC_T_FROM_S
    } /* end omp parallel */

    /* Commit best */
    memcpy(pos, best_pos, sizeof(double) * n * 2);
    if (out_cong_final) *out_cong_final = best_cong;
    if (out_proxy_final) *out_proxy_final = best_proxy;
    if (out_wl_final) *out_wl_final = best_wl;
    if (out_den_final) *out_den_final = best_den;

    free(best_pos);
    free(best_H_net); free(best_V_net);
    free(best_H_final); free(best_V_final);
    free(best_grid_occupied); free(best_net_hpwl);
    free(soft_list);
    free(hb_offsets); free(hb_macros);
    /* tl_* per-thread scratch is owned by _tl_pool - kept allocated across
     * cong_relax_v2 calls (saves ~277 ms/call). Process exit reclaims. */
    free(cand_gain); free(cand_nx); free(cand_ny);
    free(cand_new_cong); free(cand_new_den); free(cand_new_wl);
    free(sw_net_touched); free(sw_net_buf); free(sw_saved_hp); free(sw_knn_cache);
    free(heat_items); free(cold_mask); free(per_macro_acc_iter0);

cleanup:
    free(S.H_net); free(S.V_net);
    free(S.H_macro); free(S.V_macro);
    free(S.H_final); free(S.V_final);
    free(S.pin_abs_x); free(S.pin_abs_y);
    free(S.pin_row); free(S.pin_col);
    free(S.mp_offsets); free(S.mp_pin_ids);
    free(S.mn_offsets); free(S.mn_net_ids);
    free(S.scratch_rows); free(S.scratch_cols);
    free(S.seen_gcell);
    free(S.grid_occupied); free(S.net_hpwl);
    free(S.net_xmin); free(S.net_xmax);
    free(S.net_ymin); free(S.net_ymax);
    free(S.net_pin_xmin); free(S.net_pin_xmax);
    free(S.net_pin_ymin); free(S.net_pin_ymax);
    free(S.abu_scratch);
}


/* ---
 *  Incremental-scoring public C API
 * ---
 *
 * Allocates a State once and exposes apply/revert/commit primitives so
 * Python callers (hard-shake, pair-swap) can probe many candidate moves
 * without paying the ~5-15 ms init_state rebuild on every call.
 *
 * Per-probe cost drops from full re-route of every net to: route the
 * affected nets once (sub) + once (add), plus a top-N pass over the
 * 4096-cell grid for cong/density. Typical ~0.5-2 ms vs 5-15 ms.
 */

/* Signed variant of route_macro: applies +/- a hard-macro's blockage
 * to H_macro/V_macro and H_final/V_final. Mathematically equivalent to
 * route_macro when sign=+1.0. The partial_v / partial_h corrections
 * propagate the sign correctly because they're double-counted in the
 * main loop and removed once. */
static void route_macro_signed(State *s, int macro_idx, double sign) {
    double cx = s->pos[macro_idx * 2 + 0];
    double cy = s->pos[macro_idx * 2 + 1];
    double hw = s->sizes[macro_idx * 2 + 0] * 0.5;
    double hh = s->sizes[macro_idx * 2 + 1] * 0.5;
    double mlx = cx - hw, mhx = cx + hw;
    double mly = cy - hh, mhy = cy + hh;

    int ur_row = (int)floor(mhy / s->gh);
    int ur_col = (int)floor(mhx / s->gw);
    int bl_row = (int)floor(mly / s->gh);
    int bl_col = (int)floor(mlx / s->gw);

    if (!(ur_row >= 0 && ur_col >= 0)) return;
    if (bl_row < 0) bl_row = 0;
    if (bl_col < 0) bl_col = 0;
    if (!(bl_row >= 0 && bl_col >= 0)) return;
    if (ur_row > s->gr - 1) ur_row = s->gr - 1;
    if (ur_col > s->gc - 1) ur_col = s->gc - 1;

    bool partial_v = false, partial_h = false;
    double v_scale = s->v_alloc / s->grid_v_routes;
    double h_scale = s->h_alloc / s->grid_h_routes;

    for (int r = bl_row; r <= ur_row; r++) {
        double bin_y_lo = r * s->gh, bin_y_hi = bin_y_lo + s->gh;
        double yd = overlap_1d(mly, mhy, bin_y_lo, bin_y_hi);
        for (int c = bl_col; c <= ur_col; c++) {
            double bin_x_lo = c * s->gw, bin_x_hi = bin_x_lo + s->gw;
            double xd = overlap_1d(mlx, mhx, bin_x_lo, bin_x_hi);

            if (ur_row != bl_row) {
                if ((r == bl_row || r == ur_row) && d_abs(yd - s->gh) > 1e-5)
                    partial_v = true;
            }
            if (ur_col != bl_col) {
                if ((c == bl_col || c == ur_col) && d_abs(xd - s->gw) > 1e-5)
                    partial_h = true;
            }
            double v_add = sign * xd * v_scale;
            double h_add = sign * yd * h_scale;
            int idx = r * s->gc + c;
            s->V_macro[idx] += v_add; s->V_final[idx] += v_add;
            s->H_macro[idx] += h_add; s->H_final[idx] += h_add;
        }
    }
    if (partial_v) {
        int r = ur_row;
        for (int c = bl_col; c <= ur_col; c++) {
            double bin_x_lo = c * s->gw, bin_x_hi = bin_x_lo + s->gw;
            double xd = overlap_1d(mlx, mhx, bin_x_lo, bin_x_hi);
            double v_add = sign * xd * v_scale;
            int idx = r * s->gc + c;
            s->V_macro[idx] -= v_add; s->V_final[idx] -= v_add;
        }
    }
    if (partial_h) {
        int c = ur_col;
        for (int r = bl_row; r <= ur_row; r++) {
            double bin_y_lo = r * s->gh, bin_y_hi = bin_y_lo + s->gh;
            double yd = overlap_1d(mly, mhy, bin_y_lo, bin_y_hi);
            double h_add = sign * yd * h_scale;
            int idx = r * s->gc + c;
            s->H_macro[idx] -= h_add; s->H_final[idx] -= h_add;
        }
    }
}

/* Out: subtract macro m from all grids (blockage if hard, nets, density,
 * hpwl). Mirrors move_macro_out + route_macro_signed(-1) for hard. */
static void move_macro_out_full(State *s, int m) {
    if (m < s->nh) {
        route_macro_signed(s, m, -1.0);
    }
    move_macro_out(s, m);
}

/* In: add macro m to all grids at its current pos[m]. */
static void move_macro_in_full(State *s, int m) {
    if (m < s->nh) {
        route_macro_signed(s, m, +1.0);
    }
    move_macro_in(s, m);
}

/* Recompute the final scalar costs from current grids. Caller must have
 * applied any pending route_net / density_macro / route_macro_signed
 * deltas. */
static void recompute_costs(State *s) {
    s->cong_cost = compute_cong(s);
    s->density_cost = compute_density(s);
    s->wl_cost = (s->hpwl_norm > 0.0) ? (s->total_hpwl / s->hpwl_norm) : 0.0;
    s->full_cost = s->wl_cost + 0.5 * s->density_cost + 0.5 * s->cong_cost;
}

/* Handle wrapper. State is heap-allocated so the State pointer is stable
 * across Python calls. Pending arrays support one-level revert: an apply()
 * stores the previous positions and modified macros; revert() restores them.
 * Multiple consecutive applies overwrite the pending list - caller is
 * responsible for committing or reverting between applies. */
typedef struct {
    State *S;
    /* Pending revert info - a flat copy of pre-apply positions for the
     * macros we touched in the most recent apply(). cap covers worst case
     * (anyone might pass a full macro list). */
    int pending_n;
    int pending_cap;
    int *pending_macros;
    double *pending_old_x;
    double *pending_old_y;
    /* Owned copy of pos so the caller can free their numpy buffer. */
    double *owned_pos;
    /* Sticky copies of caller's input pointers - required for sizes,
     * pin_x/y, net arrays which State holds as const* pointers. We copy
     * into our own buffers for lifetime safety. */
    double *owned_sizes;
    int    *owned_movable;
    int    *owned_pin_macro;
    double *owned_pin_x;
    double *owned_pin_y;
    int    *owned_net_driver;
    int    *owned_net_sinks_off;
    int    *owned_net_sinks_idx;
    double *owned_net_weight;
    double *owned_wl_extra_weight; /* may be NULL = no critical-net weighting */
} CongHandle;

static void cong_handle_alloc_state_buffers(State *S, int n, int nh, int gr, int gc,
                                            int np, int nn, const int *net_sinks_off) {
    int ng = gr * gc;
    S->H_net    = (double *)calloc(ng, sizeof(double));
    S->V_net    = (double *)calloc(ng, sizeof(double));
    S->H_macro  = (double *)calloc(ng, sizeof(double));
    S->V_macro  = (double *)calloc(ng, sizeof(double));
    S->H_final  = (double *)calloc(ng, sizeof(double));
    S->V_final  = (double *)calloc(ng, sizeof(double));
    S->pin_abs_x = (double *)calloc(np > 0 ? np : 1, sizeof(double));
    S->pin_abs_y = (double *)calloc(np > 0 ? np : 1, sizeof(double));
    S->pin_row   = (int *)calloc(np > 0 ? np : 1, sizeof(int));
    S->pin_col   = (int *)calloc(np > 0 ? np : 1, sizeof(int));
    S->mp_offsets = (int *)calloc(n + 1, sizeof(int));
    S->mp_pin_ids = (int *)calloc(np > 0 ? np : 1, sizeof(int));
    S->mn_offsets = (int *)calloc(n + 1, sizeof(int));
    int total_pin_net = nn + (nn > 0 ? net_sinks_off[nn] : 0);
    S->mn_net_ids = (int *)calloc(total_pin_net + 1, sizeof(int));
    int max_pins = 1;
    for (int i = 0; i < nn; i++) {
        int c = (net_sinks_off[i + 1] - net_sinks_off[i]) + 1;
        if (c > max_pins) max_pins = c;
    }
    S->max_pins_per_net = max_pins;
    S->scratch_rows = (int *)calloc(max_pins + 1, sizeof(int));
    S->scratch_cols = (int *)calloc(max_pins + 1, sizeof(int));
    S->seen_gcell = (unsigned int *)calloc(ng > 0 ? ng : 1, sizeof(unsigned int));
    S->seen_gen = 1u;
    S->grid_occupied = (double *)calloc(ng, sizeof(double));
    S->net_hpwl = (double *)calloc(nn > 0 ? nn : 1, sizeof(double));
    S->net_xmin = (double *)calloc(nn > 0 ? nn : 1, sizeof(double));
    S->net_xmax = (double *)calloc(nn > 0 ? nn : 1, sizeof(double));
    S->net_ymin = (double *)calloc(nn > 0 ? nn : 1, sizeof(double));
    S->net_ymax = (double *)calloc(nn > 0 ? nn : 1, sizeof(double));
    S->net_pin_xmin = (int *)calloc(nn > 0 ? nn : 1, sizeof(int));
    S->net_pin_xmax = (int *)calloc(nn > 0 ? nn : 1, sizeof(int));
    S->net_pin_ymin = (int *)calloc(nn > 0 ? nn : 1, sizeof(int));
    S->net_pin_ymax = (int *)calloc(nn > 0 ? nn : 1, sizeof(int));
    int abu_cap = ng / 10 + 2;
    S->abu_scratch = (double *)calloc(abu_cap, sizeof(double));
    (void)nh;
}

void* cong_state_create(
    double *pos, double *sizes, int *movable,
    int n, int nh,
    double cw, double ch,
    int gr, int gc,
    int smooth_range,
    double hrpm, double vrpm,
    double h_alloc, double v_alloc,
    int *pin_macro, double *pin_x, double *pin_y, int np_pins,
    int *net_driver, int *net_sinks_off, int *net_sinks_idx,
    double *net_weight, int nn,
    double net_cnt_for_norm,
    /* Optional WL-only multiplier; NULL = identity. */
    double *wl_extra_weight
) {
    CongHandle *H = (CongHandle *)calloc(1, sizeof(CongHandle));
    H->S = (State *)calloc(1, sizeof(State));
    State *S = H->S;
    S->n = n; S->nh = nh;
    S->gr = gr; S->gc = gc; S->ng = gr * gc;
    S->cw = cw; S->ch = ch;
    S->gw = cw / gc; S->gh = ch / gr;
    S->grid_area = S->gw * S->gh;
    S->hrpm = hrpm; S->vrpm = vrpm;
    S->h_alloc = h_alloc; S->v_alloc = v_alloc;
    S->smooth_range = smooth_range;
    S->grid_h_routes = S->gh * hrpm;
    S->grid_v_routes = S->gw * vrpm;
    S->net_cnt_for_norm = net_cnt_for_norm;
    S->hpwl_norm = (cw + ch) * (net_cnt_for_norm > 0.0 ? net_cnt_for_norm : 1.0);

    /* Take ownership of pos, sizes, movable, pin/net arrays so the Python
     * caller's buffers can be freed. */
    H->owned_pos = (double *)malloc(sizeof(double) * n * 2);
    memcpy(H->owned_pos, pos, sizeof(double) * n * 2);
    H->owned_sizes = (double *)malloc(sizeof(double) * n * 2);
    memcpy(H->owned_sizes, sizes, sizeof(double) * n * 2);
    H->owned_movable = (int *)malloc(sizeof(int) * n);
    memcpy(H->owned_movable, movable, sizeof(int) * n);

    int np = np_pins;
    H->owned_pin_macro = (int *)malloc(sizeof(int) * (np > 0 ? np : 1));
    H->owned_pin_x = (double *)malloc(sizeof(double) * (np > 0 ? np : 1));
    H->owned_pin_y = (double *)malloc(sizeof(double) * (np > 0 ? np : 1));
    if (np > 0) {
        memcpy(H->owned_pin_macro, pin_macro, sizeof(int) * np);
        memcpy(H->owned_pin_x, pin_x, sizeof(double) * np);
        memcpy(H->owned_pin_y, pin_y, sizeof(double) * np);
    }

    H->owned_net_driver = (int *)malloc(sizeof(int) * (nn > 0 ? nn : 1));
    H->owned_net_sinks_off = (int *)malloc(sizeof(int) * (nn + 1));
    int sinks_total = nn > 0 ? net_sinks_off[nn] : 0;
    H->owned_net_sinks_idx = (int *)malloc(sizeof(int) * (sinks_total > 0 ? sinks_total : 1));
    H->owned_net_weight = (double *)malloc(sizeof(double) * (nn > 0 ? nn : 1));
    if (wl_extra_weight) {
        H->owned_wl_extra_weight = (double *)malloc(sizeof(double) * (nn > 0 ? nn : 1));
    } else {
        H->owned_wl_extra_weight = NULL;
    }
    if (nn > 0) {
        memcpy(H->owned_net_driver, net_driver, sizeof(int) * nn);
        memcpy(H->owned_net_sinks_off, net_sinks_off, sizeof(int) * (nn + 1));
        if (sinks_total > 0)
            memcpy(H->owned_net_sinks_idx, net_sinks_idx, sizeof(int) * sinks_total);
        memcpy(H->owned_net_weight, net_weight, sizeof(double) * nn);
        if (wl_extra_weight) {
            memcpy(H->owned_wl_extra_weight, wl_extra_weight, sizeof(double) * nn);
        }
    } else {
        H->owned_net_sinks_off[0] = 0;
    }

    S->pos = H->owned_pos;
    S->sizes = H->owned_sizes;
    S->movable = H->owned_movable;
    S->np = np;
    S->pin_macro = H->owned_pin_macro;
    S->pin_x = H->owned_pin_x;
    S->pin_y = H->owned_pin_y;
    S->nn = nn;
    S->net_driver = H->owned_net_driver;
    S->net_sinks_off = H->owned_net_sinks_off;
    S->net_sinks_idx = H->owned_net_sinks_idx;
    S->net_weight = H->owned_net_weight;
    S->wl_extra_weight = H->owned_wl_extra_weight;

    cong_handle_alloc_state_buffers(S, n, nh, gr, gc, np, nn, H->owned_net_sinks_off);
    build_macro_pin_csr(S);
    init_state(S);

    /* Pending revert buffers (worst case = n macros). */
    H->pending_cap = n > 0 ? n : 1;
    H->pending_n = 0;
    H->pending_macros = (int *)malloc(sizeof(int) * H->pending_cap);
    H->pending_old_x = (double *)malloc(sizeof(double) * H->pending_cap);
    H->pending_old_y = (double *)malloc(sizeof(double) * H->pending_cap);

    return (void *)H;
}

void cong_state_destroy(void *handle) {
    if (!handle) return;
    CongHandle *H = (CongHandle *)handle;
    State *S = H->S;
    if (S) {
        free(S->H_net); free(S->V_net);
        free(S->H_macro); free(S->V_macro);
        free(S->H_final); free(S->V_final);
        free(S->pin_abs_x); free(S->pin_abs_y);
        free(S->pin_row); free(S->pin_col);
        free(S->mp_offsets); free(S->mp_pin_ids);
        free(S->mn_offsets); free(S->mn_net_ids);
        free(S->scratch_rows); free(S->scratch_cols);
        free(S->seen_gcell);
        free(S->grid_occupied); free(S->net_hpwl);
        free(S->net_xmin); free(S->net_xmax);
        free(S->net_ymin); free(S->net_ymax);
        free(S->net_pin_xmin); free(S->net_pin_xmax);
        free(S->net_pin_ymin); free(S->net_pin_ymax);
        free(S->abu_scratch);
        free(S);
    }
    free(H->owned_pos);
    free(H->owned_sizes);
    free(H->owned_movable);
    free(H->owned_pin_macro);
    free(H->owned_pin_x);
    free(H->owned_pin_y);
    free(H->owned_net_driver);
    free(H->owned_net_sinks_off);
    free(H->owned_net_sinks_idx);
    free(H->owned_net_weight);
    free(H->owned_wl_extra_weight);
    free(H->pending_macros);
    free(H->pending_old_x);
    free(H->pending_old_y);
    free(H);
}

/* Read current cost components - no state mutation. */
double cong_state_score_current(
    void *handle,
    double *out_wl, double *out_den, double *out_cong
) {
    if (!handle) return 0.0;
    CongHandle *H = (CongHandle *)handle;
    State *S = H->S;
    if (out_wl) *out_wl = S->wl_cost;
    if (out_den) *out_den = S->density_cost;
    if (out_cong) *out_cong = S->cong_cost;
    return S->full_cost;
}

/* Apply position changes to n_changes macros. Saves old positions so the
 * call can be reverted. Returns new full proxy (with breakdown via outs).
 *
 * Multiple consecutive applies REPLACE the pending list - i.e. only the
 * most recent apply can be reverted. Use revert() or commit() between
 * applies if you need transactional behavior.
 *
 * NB: macros must be unique in the input list. The caller is responsible
 * for de-duping. */
double cong_state_apply(
    void *handle,
    int n_changes,
    int *macros,
    double *new_x, double *new_y,
    double *out_wl, double *out_den, double *out_cong
) {
    if (!handle || n_changes <= 0) {
        if (handle) {
            return cong_state_score_current(handle, out_wl, out_den, out_cong);
        }
        return 0.0;
    }
    CongHandle *H = (CongHandle *)handle;
    State *S = H->S;

    /* 1) Save old positions, then move every macro out. */
    H->pending_n = n_changes;
    if (n_changes > H->pending_cap) {
        H->pending_macros = (int *)realloc(H->pending_macros, sizeof(int) * n_changes);
        H->pending_old_x = (double *)realloc(H->pending_old_x, sizeof(double) * n_changes);
        H->pending_old_y = (double *)realloc(H->pending_old_y, sizeof(double) * n_changes);
        H->pending_cap = n_changes;
    }
    for (int k = 0; k < n_changes; k++) {
        int m = macros[k];
        H->pending_macros[k] = m;
        H->pending_old_x[k] = S->pos[m * 2 + 0];
        H->pending_old_y[k] = S->pos[m * 2 + 1];
        move_macro_out_full(S, m);
    }
    /* 2) Update positions. */
    for (int k = 0; k < n_changes; k++) {
        int m = macros[k];
        S->pos[m * 2 + 0] = new_x[k];
        S->pos[m * 2 + 1] = new_y[k];
        refresh_macro_pin_cache(S, m);
    }
    /* 3) Move every macro in. */
    for (int k = 0; k < n_changes; k++) {
        int m = macros[k];
        move_macro_in_full(S, m);
    }
    /* 4) Recompute scalars. */
    recompute_costs(S);
    if (out_wl) *out_wl = S->wl_cost;
    if (out_den) *out_den = S->density_cost;
    if (out_cong) *out_cong = S->cong_cost;
    return S->full_cost;
}

/* Undo the most recent apply by re-applying the saved old positions. */
double cong_state_revert(
    void *handle,
    double *out_wl, double *out_den, double *out_cong
) {
    if (!handle) return 0.0;
    CongHandle *H = (CongHandle *)handle;
    State *S = H->S;
    if (H->pending_n <= 0) {
        return cong_state_score_current(handle, out_wl, out_den, out_cong);
    }
    int n = H->pending_n;
    for (int k = 0; k < n; k++) {
        move_macro_out_full(S, H->pending_macros[k]);
    }
    for (int k = 0; k < n; k++) {
        int m = H->pending_macros[k];
        S->pos[m * 2 + 0] = H->pending_old_x[k];
        S->pos[m * 2 + 1] = H->pending_old_y[k];
        refresh_macro_pin_cache(S, m);
    }
    for (int k = 0; k < n; k++) {
        move_macro_in_full(S, H->pending_macros[k]);
    }
    recompute_costs(S);
    H->pending_n = 0;
    if (out_wl) *out_wl = S->wl_cost;
    if (out_den) *out_den = S->density_cost;
    if (out_cong) *out_cong = S->cong_cost;
    return S->full_cost;
}

/* Commit the pending apply (clear the revert log). State is unchanged. */
void cong_state_commit(void *handle) {
    if (!handle) return;
    CongHandle *H = (CongHandle *)handle;
    H->pending_n = 0;
}

/* Copy the current pos array out (n_macros * 2 doubles). */
void cong_state_get_pos(void *handle, double *out_pos) {
    if (!handle || !out_pos) return;
    CongHandle *H = (CongHandle *)handle;
    memcpy(out_pos, H->S->pos, sizeof(double) * H->S->n * 2);
}

/* Force a full rebuild of the maps (FP drift mitigation after many
 * incremental updates). Same effect as init_state's last steps without
 * rebuilding the static CSR. */
void cong_state_rebuild(void *handle) {
    if (!handle) return;
    CongHandle *H = (CongHandle *)handle;
    rebuild_maps(H->S);
    H->pending_n = 0;
}
