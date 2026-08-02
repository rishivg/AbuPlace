#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline double d_max(double a, double b) { return a > b ? a : b; }
static inline double d_min(double a, double b) { return a < b ? a : b; }
static inline double d_abs(double a) { return a < 0 ? -a : a; }
static inline int    i_max(int a, int b)       { return a > b ? a : b; }
static inline int    i_min(int a, int b)       { return a < b ? a : b; }

typedef struct {
    /* geometry */
    int n, nh;
    double cw, ch;
    int gr, gc, ng;
    double gw, gh;
    double inv_ba;
    int k_den, k_cong;
    int sr;

    const double *sizes;     /* [n*2] */
    const int *movable;      /* [n] */

    /* Pins. pin_macro[p]: macro idx (>=0) or -1 for port.
     * pin_x/pin_y: offset (relative to macro center) for macro pins; absolute
     * coordinates for port pins. pin_abs holds the per-iteration absolute
     * position cache (pin_abs[p*2+d] = pos[m*2+d]+offset, refreshed each step).
     */
    int np;
    const int    *pin_macro;
    const double *pin_x;
    const double *pin_y;
    double       *pin_abs;     /* [np*2] */

    /* Pin-net CSR. net_pin_idx[net_pin_off[k] .. net_pin_off[k+1]) = pins of
     * net k (driver first, then sinks). net_weight[k] scales WL contribution. */
    int nn;
    int total_pin_refs;
    const int    *net_pin_off;
    const int    *net_pin_idx;
    const double *net_weight;
    /* Optional WL-only multiplier, applied alongside net_weight at WL
     * forward + WL backward. NULL means identity (no critical-net weighting). */
    const double *wl_extra_weight;
    double *deg_scale;       /* [nn] sqrt(clamp(len-1, 1)) */

    /* mov_idx: movable macros in [0..n). */
    int n_mov;
    int *mov_idx;            /* [n_mov] */
    int *mov_rev;            /* [n]  mov_rev[i] = slot in mov_idx or -1 */

    /* Positions (full, [n*2]). */
    double *pos;

    /* Adam state on movable macros: m[n_mov*2], v[n_mov*2]. */
    double *adm_m;
    double *adm_v;
    int adm_t;
    double beta1, beta2, eps;
    double beta1_pow;
    double beta2_pow;

    /* Per-iteration scratch. */
    double *grad;            /* [n_mov*2] */

    /* WA-HPWL per-net intermediates (two dims). */
    double *wa_hi;           /* [nn*2] */
    double *wa_lo;           /* [nn*2] */
    double *sum_e_hi;        /* [nn*2] */
    double *sum_e_lo;        /* [nn*2] */
    double *mx_hi;           /* [nn*2] shift for numerical stability */
    double *mx_lo;           /* [nn*2] */
    double *inv_gamma_net;   /* [nn] */
    double  gamma_base;      /* gamma * scalar */
    double  gamma_avg;       /* avg total_span per net */
    double *total_span;      /* [nn] - hi0-lo0+hi1-lo1 */

    /* Per-pin-ref exp caches, indexed by ref p and dim d: arr[p*2 + d]. */
    double *e_hi_cache;      /* [total_pin_refs*2] */
    double *e_lo_cache;      /* [total_pin_refs*2] */

    /* Density & cong maps. */
    double *den_map;         /* [ng] */
    double *den_map_grad;    /* [ng] */
    double *cong_dm;         /* [ng] */
    double *cong_sm;         /* [ng] */
    double *cong_sm_grad;    /* [ng] */
    double *cong_dm_grad;    /* [ng] */

    /* Shared reusable scratch. */
    int max_threads;
    double *tls_grad;        /* [max_threads * n_mov * 2] */
    double *tls_den_map;     /* [max_threads * ng] */
    double *tls_cong_dm;     /* [max_threads * ng] */
    double *smooth_tmp;      /* [ng] */
    int *topk_idx_den;       /* [k_den] */
    int *topk_idx_cong;      /* [k_cong] */
    double *topk_heap;       /* [max(k_den, k_cong)] */
} RS;

static inline int grad_slot(const RS *s, int macro_idx) {
    return s->mov_rev[macro_idx];
}

static inline void add_grad_mov(const RS *s, double *g_mov, int macro_idx, int dim, double val) {
    int mv = s->mov_rev[macro_idx];
    if (mv >= 0) {
        g_mov[mv * 2 + dim] += val;
    }
}

static void zero_tls_grad(RS *s) {
    if (s->n_mov > 0 && s->max_threads > 0) {
        memset(s->tls_grad, 0, sizeof(double) * (size_t)s->max_threads * (size_t)s->n_mov * 2u);
    }
}

static void reduce_tls_grad(const RS *s, double *g_mov) {
    int N2 = s->n_mov * 2;
    for (int t = 0; t < s->max_threads; t++) {
        const double *src = s->tls_grad + (size_t)t * (size_t)N2;
        for (int i = 0; i < N2; i++) {
            g_mov[i] += src[i];
        }
    }
}

static void zero_tls_map(double *buf, int max_threads, int ng) {
    if (max_threads > 0 && ng > 0) {
        memset(buf, 0, sizeof(double) * (size_t)max_threads * (size_t)ng);
    }
}

static void reduce_tls_map(const double *tls, int max_threads, double *out, int ng) {
    memset(out, 0, sizeof(double) * (size_t)ng);
    for (int t = 0; t < max_threads; t++) {
        const double *src = tls + (size_t)t * (size_t)ng;
        for (int i = 0; i < ng; i++) {
            out[i] += src[i];
        }
    }
}

/* Partial-select top-k indices in arr[0..ng) by descending value.
 * Uses a fixed-size linear min-buffer. Caller provides scratch heap_val[k]. */
static void topk_indices(const double *arr, int ng, int k, int *out_idx, double *heap_val) {
    if (k > ng) k = ng;
    for (int i = 0; i < k; i++) {
        heap_val[i] = -1e300;
        out_idx[i] = -1;
    }
    double cur_min = -1e300;
    int min_pos = 0;
    int filled = 0;
    for (int i = 0; i < ng; i++) {
        double v = arr[i];
        if (filled < k) {
            heap_val[filled] = v;
            out_idx[filled] = i;
            filled++;
            if (filled == k) {
                cur_min = heap_val[0];
                min_pos = 0;
                for (int j = 1; j < k; j++) {
                    if (heap_val[j] < cur_min) {
                        cur_min = heap_val[j];
                        min_pos = j;
                    }
                }
            }
            continue;
        }
        if (v > cur_min) {
            heap_val[min_pos] = v;
            out_idx[min_pos] = i;
            cur_min = heap_val[0];
            min_pos = 0;
            for (int j = 1; j < k; j++) {
                if (heap_val[j] < cur_min) {
                    cur_min = heap_val[j];
                    min_pos = j;
                }
            }
        }
    }
}

/* Refresh pin_abs cache. Macro pins: pos[m] + offset. Ports: absolute. */
static void refresh_pin_abs(RS *s) {
#pragma omp parallel for schedule(static)
    for (int p = 0; p < s->np; p++) {
        int m = s->pin_macro[p];
        if (m >= 0) {
            s->pin_abs[p * 2 + 0] = s->pos[m * 2 + 0] + s->pin_x[p];
            s->pin_abs[p * 2 + 1] = s->pos[m * 2 + 1] + s->pin_y[p];
        } else {
            s->pin_abs[p * 2 + 0] = s->pin_x[p];
            s->pin_abs[p * 2 + 1] = s->pin_y[p];
        }
    }
}

static void wl_forward_pass1(RS *s) {
    int nn = s->nn;
#pragma omp parallel for schedule(static)
    for (int k = 0; k < nn; k++) {
        int o = s->net_pin_off[k], e = s->net_pin_off[k + 1];
        double mx0 = -1e30, mx1 = -1e30, mn0 = 1e30, mn1 = 1e30;
        for (int p = o; p < e; p++) {
            int pin = s->net_pin_idx[p];
            double x = s->pin_abs[pin * 2 + 0];
            double y = s->pin_abs[pin * 2 + 1];
            if (x > mx0) mx0 = x;
            if (y > mx1) mx1 = y;
            if (x < mn0) mn0 = x;
            if (y < mn1) mn1 = y;
        }
        s->mx_hi[k * 2 + 0] = mx0;
        s->mx_lo[k * 2 + 0] = mn0;
        s->mx_hi[k * 2 + 1] = mx1;
        s->mx_lo[k * 2 + 1] = mn1;
        s->total_span[k] = (mx0 - mn0) + (mx1 - mn1);
    }

    double sum = 0.0;
    for (int k = 0; k < nn; k++) sum += s->total_span[k];
    double avg = (nn > 0) ? sum / (double)nn : 1.0;
    if (avg < 1e-6) avg = 1e-6;
    s->gamma_avg = avg;

#pragma omp parallel for schedule(static)
    for (int k = 0; k < nn; k++) {
        double gamma_net = s->gamma_base * (0.85 + 0.15 * s->total_span[k] / s->gamma_avg);
        s->inv_gamma_net[k] = 1.0 / gamma_net;
    }
}

/* Per-net contribution buffer reused across calls. Sized once at first call
 * (or via init); we just rely on s->nn being constant per RS instance. */
static double *_wl_per_net = NULL;
static int _wl_per_net_n = 0;

static double wl_forward_pass2(RS *s) {
    int nn = s->nn;
    if (_wl_per_net_n < nn) {
        free(_wl_per_net);
        _wl_per_net = (double *)malloc(sizeof(double) * (nn > 0 ? nn : 1));
        _wl_per_net_n = nn;
    }
    /* PARALLEL WRITE to per-net slot (no reduction) - disjoint indices, no race. */
#pragma omp parallel for schedule(static)
    for (int k = 0; k < nn; k++) {
        int o = s->net_pin_off[k], e = s->net_pin_off[k + 1];
        double inv_gn = s->inv_gamma_net[k];
        double w = s->net_weight[k];
        if (s->wl_extra_weight) w *= s->wl_extra_weight[k];
        double net_contrib = 0.0;
        for (int d = 0; d < 2; d++) {
            double mx = s->mx_hi[k * 2 + d];
            double mn = s->mx_lo[k * 2 + d];
            double num_hi = 0.0, den_hi = 0.0;
            double num_lo = 0.0, den_lo = 0.0;
            for (int p = o; p < e; p++) {
                int pin = s->net_pin_idx[p];
                double c = s->pin_abs[pin * 2 + d];
                double eh = exp((c - mx) * inv_gn);
                double el = exp(-(c - mn) * inv_gn);
                s->e_hi_cache[p * 2 + d] = eh;
                s->e_lo_cache[p * 2 + d] = el;
                num_hi += c * eh;
                den_hi += eh;
                num_lo += c * el;
                den_lo += el;
            }
            double wa_hi = num_hi / (den_hi + 1e-10);
            double wa_lo = num_lo / (den_lo + 1e-10);
            s->wa_hi[k * 2 + d] = wa_hi;
            s->wa_lo[k * 2 + d] = wa_lo;
            s->sum_e_hi[k * 2 + d] = den_hi;
            s->sum_e_lo[k * 2 + d] = den_lo;
            net_contrib += w * (wa_hi - wa_lo);
        }
        _wl_per_net[k] = net_contrib;
    }
    /* SERIAL sum in fixed (k=0..nn-1) order - bit-deterministic. */
    double wl_total = 0.0;
    for (int k = 0; k < nn; k++) wl_total += _wl_per_net[k];
    return wl_total;
}

static void wl_backward(RS *s, double *g_mov, double g_scale) {
    zero_tls_grad(s);
#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *gp = s->tls_grad + (size_t)tid * (size_t)s->n_mov * 2u;
#pragma omp for schedule(static)
        for (int k = 0; k < s->nn; k++) {
            int o = s->net_pin_off[k], e = s->net_pin_off[k + 1];
            double inv_gn = s->inv_gamma_net[k];
            double w_k = s->net_weight[k];
            if (s->wl_extra_weight) w_k *= s->wl_extra_weight[k];
            double wg = g_scale * w_k;
            for (int d = 0; d < 2; d++) {
                double wa_hi = s->wa_hi[k * 2 + d];
                double wa_lo = s->wa_lo[k * 2 + d];
                double S_hi = s->sum_e_hi[k * 2 + d] + 1e-10;
                double S_lo = s->sum_e_lo[k * 2 + d] + 1e-10;
                double inv_S_hi = 1.0 / S_hi;
                double inv_S_lo = 1.0 / S_lo;
                for (int p = o; p < e; p++) {
                    int pin = s->net_pin_idx[p];
                    int m = s->pin_macro[pin];
                    if (m < 0) continue;
                    int mv = grad_slot(s, m);
                    if (mv < 0) continue;
                    double c = s->pin_abs[pin * 2 + d];
                    double eh = s->e_hi_cache[p * 2 + d];
                    double el = s->e_lo_cache[p * 2 + d];
                    double d_hi = eh * inv_S_hi * (1.0 + (c - wa_hi) * inv_gn);
                    double d_lo = el * inv_S_lo * (1.0 - (c - wa_lo) * inv_gn);
                    gp[mv * 2 + d] += wg * (d_hi - d_lo);
                }
            }
        }
    }
    reduce_tls_grad(s, g_mov);
}

static void density_forward(RS *s, double *out_den) {
    zero_tls_map(s->tls_den_map, s->max_threads, s->ng);

#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *map = s->tls_den_map + (size_t)tid * (size_t)s->ng;
#pragma omp for schedule(static)
        for (int i = 0; i < s->n; i++) {
            double px = s->pos[i * 2 + 0], py = s->pos[i * 2 + 1];
            double hw = s->sizes[i * 2 + 0] * 0.5;
            double hh = s->sizes[i * 2 + 1] * 0.5;
            double lx = px - hw, rx = px + hw;
            double ly = py - hh, ry = py + hh;
            int c_lo = (int)floor(lx / s->gw); if (c_lo < 0) c_lo = 0;
            int c_hi = (int)floor((rx - 1e-12) / s->gw); if (c_hi >= s->gc) c_hi = s->gc - 1;
            int r_lo = (int)floor(ly / s->gh); if (r_lo < 0) r_lo = 0;
            int r_hi = (int)floor((ry - 1e-12) / s->gh); if (r_hi >= s->gr) r_hi = s->gr - 1;
            for (int r = r_lo; r <= r_hi; r++) {
                double bl_y = r * s->gh, tl_y = bl_y + s->gh;
                double oy_v = d_min(ry, tl_y) - d_max(ly, bl_y);
                if (oy_v <= 0.0) continue;
                for (int c = c_lo; c <= c_hi; c++) {
                    double bl_x = c * s->gw, tl_x = bl_x + s->gw;
                    double ox_v = d_min(rx, tl_x) - d_max(lx, bl_x);
                    if (ox_v <= 0.0) continue;
                    map[r * s->gc + c] += ox_v * oy_v;
                }
            }
        }
    }

    reduce_tls_map(s->tls_den_map, s->max_threads, s->den_map, s->ng);
    for (int i = 0; i < s->ng; i++) s->den_map[i] *= s->inv_ba;

    memset(s->den_map_grad, 0, sizeof(double) * (size_t)s->ng);
    topk_indices(s->den_map, s->ng, s->k_den, s->topk_idx_den, s->topk_heap);
    double sum = 0.0;
    double w = 1.0 / (double)s->k_den;
    for (int i = 0; i < s->k_den; i++) {
        int idx = s->topk_idx_den[i];
        if (idx < 0) continue;
        sum += s->den_map[idx];
        s->den_map_grad[idx] = w;
    }
    *out_den = sum * w;
}

static void density_backward(RS *s, double g_scale) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < s->n; i++) {
        int mv = grad_slot(s, i);
        if (mv < 0) continue;
        double px = s->pos[i * 2 + 0], py = s->pos[i * 2 + 1];
        double hw = s->sizes[i * 2 + 0] * 0.5;
        double hh = s->sizes[i * 2 + 1] * 0.5;
        double lx = px - hw, rx = px + hw;
        double ly = py - hh, ry = py + hh;
        int c_lo = (int)floor(lx / s->gw); if (c_lo < 0) c_lo = 0;
        int c_hi = (int)floor((rx - 1e-12) / s->gw); if (c_hi >= s->gc) c_hi = s->gc - 1;
        int r_lo = (int)floor(ly / s->gh); if (r_lo < 0) r_lo = 0;
        int r_hi = (int)floor((ry - 1e-12) / s->gh); if (r_hi >= s->gr) r_hi = s->gr - 1;
        double grad_x = 0.0, grad_y = 0.0;
        for (int r = r_lo; r <= r_hi; r++) {
            double bl_y = r * s->gh, tl_y = bl_y + s->gh;
            double top_y = d_min(ry, tl_y);
            double bot_y = d_max(ly, bl_y);
            double oy_v = top_y - bot_y;
            if (oy_v <= 0.0) continue;
            double doy = ((ry < tl_y) ? 1.0 : 0.0) - ((ly > bl_y) ? 1.0 : 0.0);
            for (int c = c_lo; c <= c_hi; c++) {
                double bl_x = c * s->gw, tl_x = bl_x + s->gw;
                double top_x = d_min(rx, tl_x);
                double bot_x = d_max(lx, bl_x);
                double ox_v = top_x - bot_x;
                if (ox_v <= 0.0) continue;
                double dox = ((rx < tl_x) ? 1.0 : 0.0) - ((lx > bl_x) ? 1.0 : 0.0);
                double g_cell = s->den_map_grad[r * s->gc + c] * s->inv_ba;
                if (g_cell == 0.0) continue;
                grad_x += g_cell * oy_v * dox;
                grad_y += g_cell * ox_v * doy;
            }
        }
        s->grad[mv * 2 + 0] += g_scale * grad_x;
        s->grad[mv * 2 + 1] += g_scale * grad_y;
    }
}

static inline double box_overlap_1d(double a_lo, double a_hi, double b_lo, double b_hi) {
    double lo = d_max(a_lo, b_lo), hi = d_min(a_hi, b_hi);
    return hi > lo ? hi - lo : 0.0;
}

static void cong_forward(RS *s, double *out_cong) {
    zero_tls_map(s->tls_cong_dm, s->max_threads, s->ng);

#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *map = s->tls_cong_dm + (size_t)tid * (size_t)s->ng;

#pragma omp for schedule(static)
        for (int k = 0; k < s->nn; k++) {
            double lo_x = s->wa_lo[k * 2 + 0], hi_x = s->wa_hi[k * 2 + 0];
            double lo_y = s->wa_lo[k * 2 + 1], hi_y = s->wa_hi[k * 2 + 1];
            double bw = hi_x - lo_x; if (bw < 0.1) bw = 0.1;
            double bh = hi_y - lo_y; if (bh < 0.1) bh = 0.1;
            double dw = s->deg_scale[k] / (bw * bh);

            int c_lo = (int)floor(lo_x / s->gw); if (c_lo < 0) c_lo = 0;
            int c_hi = (int)floor((hi_x - 1e-12) / s->gw); if (c_hi >= s->gc) c_hi = s->gc - 1;
            int r_lo = (int)floor(lo_y / s->gh); if (r_lo < 0) r_lo = 0;
            int r_hi = (int)floor((hi_y - 1e-12) / s->gh); if (r_hi >= s->gr) r_hi = s->gr - 1;
            for (int r = r_lo; r <= r_hi; r++) {
                double bl_y = r * s->gh, tl_y = bl_y + s->gh;
                double cy = box_overlap_1d(lo_y, hi_y, bl_y, tl_y);
                if (cy <= 0.0) continue;
                for (int c = c_lo; c <= c_hi; c++) {
                    double bl_x = c * s->gw, tl_x = bl_x + s->gw;
                    double cx = box_overlap_1d(lo_x, hi_x, bl_x, tl_x);
                    if (cx <= 0.0) continue;
                    map[r * s->gc + c] += dw * cy * cx;
                }
            }
        }

        double hd_scale = s->inv_ba * 0.9;
#pragma omp for schedule(static)
        for (int i = 0; i < s->nh; i++) {
            double px = s->pos[i * 2 + 0], py = s->pos[i * 2 + 1];
            double hw = s->sizes[i * 2 + 0] * 0.5;
            double hh = s->sizes[i * 2 + 1] * 0.5;
            double lx = px - hw, rx = px + hw;
            double ly = py - hh, ry = py + hh;
            int c_lo = (int)floor(lx / s->gw); if (c_lo < 0) c_lo = 0;
            int c_hi = (int)floor((rx - 1e-12) / s->gw); if (c_hi >= s->gc) c_hi = s->gc - 1;
            int r_lo = (int)floor(ly / s->gh); if (r_lo < 0) r_lo = 0;
            int r_hi = (int)floor((ry - 1e-12) / s->gh); if (r_hi >= s->gr) r_hi = s->gr - 1;
            for (int r = r_lo; r <= r_hi; r++) {
                double bl_y = r * s->gh, tl_y = bl_y + s->gh;
                double oy_v = box_overlap_1d(ly, ry, bl_y, tl_y);
                if (oy_v <= 0.0) continue;
                for (int c = c_lo; c <= c_hi; c++) {
                    double bl_x = c * s->gw, tl_x = bl_x + s->gw;
                    double ox_v = box_overlap_1d(lx, rx, bl_x, tl_x);
                    if (ox_v <= 0.0) continue;
                    map[r * s->gc + c] += hd_scale * ox_v * oy_v;
                }
            }
        }
    }

    reduce_tls_map(s->tls_cong_dm, s->max_threads, s->cong_dm, s->ng);

    int gr = s->gr, gc = s->gc, sr = s->sr;
    int ks = 2 * sr + 1;
    double inv_ks = 1.0 / (double)ks;
    memset(s->cong_sm, 0, sizeof(double) * (size_t)s->ng);

    for (int r = 0; r < gr; r++) {
        for (int c = 0; c < gc; c++) {
            double sum = 0.0;
            for (int k = -sr; k <= sr; k++) {
                int cc = c + k;
                if (cc < 0) cc = 0;
                if (cc >= gc) cc = gc - 1;
                sum += s->cong_dm[r * gc + cc];
            }
            s->smooth_tmp[r * gc + c] = sum * inv_ks;
        }
    }
    for (int i = 0; i < s->ng; i++) s->cong_sm[i] += s->smooth_tmp[i];

    for (int r = 0; r < gr; r++) {
        for (int c = 0; c < gc; c++) {
            double sum = 0.0;
            for (int k = -sr; k <= sr; k++) {
                int rr = r + k;
                if (rr < 0) rr = 0;
                if (rr >= gr) rr = gr - 1;
                sum += s->cong_dm[rr * gc + c];
            }
            s->smooth_tmp[r * gc + c] = sum * inv_ks;
        }
    }
    for (int i = 0; i < s->ng; i++) s->cong_sm[i] += s->smooth_tmp[i];

    memset(s->cong_sm_grad, 0, sizeof(double) * (size_t)s->ng);
    topk_indices(s->cong_sm, s->ng, s->k_cong, s->topk_idx_cong, s->topk_heap);
    double sum = 0.0;
    double w = 1.0 / (double)s->k_cong;
    for (int i = 0; i < s->k_cong; i++) {
        int idx = s->topk_idx_cong[i];
        if (idx < 0) continue;
        sum += s->cong_sm[idx];
        s->cong_sm_grad[idx] = w;
    }
    *out_cong = sum * w;
}

static void cong_backward_unsmooth(RS *s) {
    int gr = s->gr, gc = s->gc, sr = s->sr;
    int ks = 2 * sr + 1;
    double inv_ks = 1.0 / (double)ks;
    memset(s->cong_dm_grad, 0, sizeof(double) * (size_t)s->ng);
    for (int r = 0; r < gr; r++) {
        for (int c = 0; c < gc; c++) {
            double sg = s->cong_sm_grad[r * gc + c];
            if (sg == 0.0) continue;
            double w = sg * inv_ks;
            for (int k = -sr; k <= sr; k++) {
                int cc = c - k;
                if (cc < 0) cc = 0;
                if (cc >= gc) cc = gc - 1;
                s->cong_dm_grad[r * gc + cc] += w;
            }
        }
    }
    for (int r = 0; r < gr; r++) {
        for (int c = 0; c < gc; c++) {
            double sg = s->cong_sm_grad[r * gc + c];
            if (sg == 0.0) continue;
            double w = sg * inv_ks;
            for (int k = -sr; k <= sr; k++) {
                int rr = r - k;
                if (rr < 0) rr = 0;
                if (rr >= gr) rr = gr - 1;
                s->cong_dm_grad[rr * gc + c] += w;
            }
        }
    }
}

static void cong_backward(RS *s, double *g_mov, double g_scale) {
    cong_backward_unsmooth(s);
    zero_tls_grad(s);

#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *gp = s->tls_grad + (size_t)tid * (size_t)s->n_mov * 2u;

#pragma omp for schedule(static)
        for (int k = 0; k < s->nn; k++) {
            double lo_x = s->wa_lo[k * 2 + 0], hi_x = s->wa_hi[k * 2 + 0];
            double lo_y = s->wa_lo[k * 2 + 1], hi_y = s->wa_hi[k * 2 + 1];
            double raw_bw = hi_x - lo_x;
            double raw_bh = hi_y - lo_y;
            double bw = raw_bw < 0.1 ? 0.1 : raw_bw;
            double bh = raw_bh < 0.1 ? 0.1 : raw_bh;
            double dbw_draw = (raw_bw < 0.1) ? 0.0 : 1.0;
            double dbh_draw = (raw_bh < 0.1) ? 0.0 : 1.0;
            double dw = s->deg_scale[k] / (bw * bh);

            int c_lo = (int)floor(lo_x / s->gw); if (c_lo < 0) c_lo = 0;
            int c_hi = (int)floor((hi_x - 1e-12) / s->gw); if (c_hi >= s->gc) c_hi = s->gc - 1;
            int r_lo = (int)floor(lo_y / s->gh); if (r_lo < 0) r_lo = 0;
            int r_hi = (int)floor((hi_y - 1e-12) / s->gh); if (r_hi >= s->gr) r_hi = s->gr - 1;

            double g_dw = 0.0;
            double g_hi_x = 0.0, g_lo_x = 0.0;
            double g_hi_y = 0.0, g_lo_y = 0.0;
            for (int r = r_lo; r <= r_hi; r++) {
                double bl_y = r * s->gh, tl_y = bl_y + s->gh;
                double top_y = d_min(hi_y, tl_y), bot_y = d_max(lo_y, bl_y);
                double cy = top_y - bot_y;
                if (cy <= 0.0) continue;
                double dcy_dhi = (hi_y < tl_y) ? 1.0 : 0.0;
                double dcy_dlo = (lo_y > bl_y) ? -1.0 : 0.0;
                for (int c = c_lo; c <= c_hi; c++) {
                    double bl_x = c * s->gw, tl_x = bl_x + s->gw;
                    double top_x = d_min(hi_x, tl_x), bot_x = d_max(lo_x, bl_x);
                    double cx = top_x - bot_x;
                    if (cx <= 0.0) continue;
                    double g_cell = s->cong_dm_grad[r * s->gc + c];
                    if (g_cell == 0.0) continue;
                    double dcx_dhi = (hi_x < tl_x) ? 1.0 : 0.0;
                    double dcx_dlo = (lo_x > bl_x) ? -1.0 : 0.0;
                    g_dw   += g_cell * cy * cx;
                    g_hi_x += g_cell * dw * cy * dcx_dhi;
                    g_lo_x += g_cell * dw * cy * dcx_dlo;
                    g_hi_y += g_cell * dw * cx * dcy_dhi;
                    g_lo_y += g_cell * dw * cx * dcy_dlo;
                }
            }
            double d_dw_bw = -dw / bw;
            double d_dw_bh = -dw / bh;
            g_hi_x += g_dw * d_dw_bw * dbw_draw;
            g_lo_x += g_dw * d_dw_bw * (-dbw_draw);
            g_hi_y += g_dw * d_dw_bh * dbh_draw;
            g_lo_y += g_dw * d_dw_bh * (-dbh_draw);

            int o = s->net_pin_off[k], e = s->net_pin_off[k + 1];
            double inv_gn = s->inv_gamma_net[k];
            for (int d = 0; d < 2; d++) {
                double wa_hi_v = s->wa_hi[k * 2 + d];
                double wa_lo_v = s->wa_lo[k * 2 + d];
                double S_hi = s->sum_e_hi[k * 2 + d] + 1e-10;
                double S_lo = s->sum_e_lo[k * 2 + d] + 1e-10;
                double inv_S_hi = 1.0 / S_hi;
                double inv_S_lo = 1.0 / S_lo;
                double up_hi = (d == 0) ? g_hi_x : g_hi_y;
                double up_lo = (d == 0) ? g_lo_x : g_lo_y;
                for (int p = o; p < e; p++) {
                    int pin = s->net_pin_idx[p];
                    int m = s->pin_macro[pin];
                    if (m < 0) continue;
                    int mv = grad_slot(s, m);
                    if (mv < 0) continue;
                    double c = s->pin_abs[pin * 2 + d];
                    double eh = s->e_hi_cache[p * 2 + d];
                    double el = s->e_lo_cache[p * 2 + d];
                    double d_hi = eh * inv_S_hi * (1.0 + (c - wa_hi_v) * inv_gn);
                    double d_lo = el * inv_S_lo * (1.0 - (c - wa_lo_v) * inv_gn);
                    gp[mv * 2 + d] += g_scale * (up_hi * d_hi + up_lo * d_lo);
                }
            }
        }
    }
    reduce_tls_grad(s, g_mov);

    double hd_scale = s->inv_ba * 0.9;
#pragma omp parallel for schedule(static)
    for (int i = 0; i < s->nh; i++) {
        int mv = grad_slot(s, i);
        if (mv < 0) continue;
        double px = s->pos[i * 2 + 0], py = s->pos[i * 2 + 1];
        double hw = s->sizes[i * 2 + 0] * 0.5;
        double hh = s->sizes[i * 2 + 1] * 0.5;
        double lx = px - hw, rx = px + hw;
        double ly = py - hh, ry = py + hh;
        int c_lo = (int)floor(lx / s->gw); if (c_lo < 0) c_lo = 0;
        int c_hi = (int)floor((rx - 1e-12) / s->gw); if (c_hi >= s->gc) c_hi = s->gc - 1;
        int r_lo = (int)floor(ly / s->gh); if (r_lo < 0) r_lo = 0;
        int r_hi = (int)floor((ry - 1e-12) / s->gh); if (r_hi >= s->gr) r_hi = s->gr - 1;
        double grad_x = 0.0, grad_y = 0.0;
        for (int r = r_lo; r <= r_hi; r++) {
            double bl_y = r * s->gh, tl_y = bl_y + s->gh;
            double top_y = d_min(ry, tl_y), bot_y = d_max(ly, bl_y);
            double oy_v = top_y - bot_y;
            if (oy_v <= 0.0) continue;
            double doy = ((ry < tl_y) ? 1.0 : 0.0) - ((ly > bl_y) ? 1.0 : 0.0);
            for (int c = c_lo; c <= c_hi; c++) {
                double bl_x = c * s->gw, tl_x = bl_x + s->gw;
                double top_x = d_min(rx, tl_x), bot_x = d_max(lx, bl_x);
                double ox_v = top_x - bot_x;
                if (ox_v <= 0.0) continue;
                double dox = ((rx < tl_x) ? 1.0 : 0.0) - ((lx > bl_x) ? 1.0 : 0.0);
                double g_cell = s->cong_dm_grad[r * s->gc + c];
                if (g_cell == 0.0) continue;
                grad_x += g_cell * hd_scale * oy_v * dox;
                grad_y += g_cell * hd_scale * ox_v * doy;
            }
        }
        s->grad[mv * 2 + 0] += g_scale * grad_x;
        s->grad[mv * 2 + 1] += g_scale * grad_y;
    }
}

static double rep_forward_and_grad(RS *s, double rm, double *g_mov, double g_scale) {
    zero_tls_grad(s);
    /* Per-thread rep accumulator; combined serially in thread-id order
     * for bit-determinism. (OpenMP's `reduction(+:rep)` combine order is
     * implementation-defined -> non-deterministic FP.) */
    double *tl_rep = (double *)calloc((size_t)(s->max_threads > 0 ? s->max_threads : 1),
                                       sizeof(double));
    /* schedule(static) - deterministic per-iter thread assignment.
     * (was schedule(dynamic, 8); dynamic schedule order is non-deterministic.) */
#pragma omp parallel
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        double *gp = s->tls_grad + (size_t)tid * (size_t)s->n_mov * 2u;
        double local_rep = 0.0;
#pragma omp for schedule(static)
        for (int i = 0; i < s->nh; i++) {
            for (int j = i + 1; j < s->nh; j++) {
                double dxv = s->pos[i * 2 + 0] - s->pos[j * 2 + 0];
                double dyv = s->pos[i * 2 + 1] - s->pos[j * 2 + 1];
                double adx = d_abs(dxv), ady = d_abs(dyv);
                double sx = (s->sizes[i * 2 + 0] + s->sizes[j * 2 + 0]) * 0.5;
                double sy = (s->sizes[i * 2 + 1] + s->sizes[j * 2 + 1]) * 0.5;
                double ax = sx + rm - adx;
                double ay = sy + rm - ady;
                if (ax <= 0.0 || ay <= 0.0) continue;

                local_rep += 2.0 * ax * ay;

                double sdx = (dxv >= 0.0) ? 1.0 : -1.0;
                double sdy = (dyv >= 0.0) ? 1.0 : -1.0;
                double gi_x = -2.0 * sdx * ay;
                double gi_y = -2.0 * sdy * ax;
                double gj_x =  2.0 * sdx * ay;
                double gj_y =  2.0 * sdy * ax;

                add_grad_mov(s, gp, i, 0, g_scale * gi_x);
                add_grad_mov(s, gp, i, 1, g_scale * gi_y);
                add_grad_mov(s, gp, j, 0, g_scale * gj_x);
                add_grad_mov(s, gp, j, 1, g_scale * gj_y);
            }
        }
        tl_rep[tid] = local_rep;
    }
    /* Serial sum in thread-id order - bit-deterministic. */
    double rep = 0.0;
    for (int t = 0; t < s->max_threads; t++) rep += tl_rep[t];
    free(tl_rep);
    reduce_tls_grad(s, g_mov);
    return rep;
}

static void adam_step(RS *s, const double *g_mov, double lr,
                      const double *lo_bounds, const double *hi_bounds) {
    s->adm_t += 1;
    s->beta1_pow *= s->beta1;
    s->beta2_pow *= s->beta2;
    double bc1 = 1.0 - s->beta1_pow;
    double bc2 = 1.0 - s->beta2_pow;

    double gnorm2 = 0.0;
    for (int i = 0; i < s->n_mov; i++) {
        double gx = g_mov[i * 2 + 0], gy = g_mov[i * 2 + 1];
        gnorm2 += gx * gx + gy * gy;
    }
    double gnorm = sqrt(gnorm2);
    double clip_scale = 1.0;
    if (gnorm > 100.0) clip_scale = 100.0 / (gnorm + 1e-12);

    for (int i = 0; i < s->n_mov; i++) {
        int m = s->mov_idx[i];
        for (int d = 0; d < 2; d++) {
            double gi = g_mov[i * 2 + d] * clip_scale;
            double mi = s->beta1 * s->adm_m[i * 2 + d] + (1.0 - s->beta1) * gi;
            double vi = s->beta2 * s->adm_v[i * 2 + d] + (1.0 - s->beta2) * gi * gi;
            s->adm_m[i * 2 + d] = mi;
            s->adm_v[i * 2 + d] = vi;
            double m_hat = mi / bc1;
            double v_hat = vi / bc2;
            double step = lr * m_hat / (sqrt(v_hat) + s->eps);
            double new_p = s->pos[m * 2 + d] - step;
            if (new_p < lo_bounds[i * 2 + d]) new_p = lo_bounds[i * 2 + d];
            if (new_p > hi_bounds[i * 2 + d]) new_p = hi_bounds[i * 2 + d];
            s->pos[m * 2 + d] = new_p;
        }
    }
}

void refine_adam(
    double *pos,
    int n, int nh,
    const double *sizes,
    const int *movable,
    double cw, double ch,
    int gr, int gc,
    int smooth_range,
    int np_pins,
    const int    *pin_macro,
    const double *pin_x,
    const double *pin_y,
    const int    *net_pin_off,
    const int    *net_pin_idx,
    const double *net_weight,
    int nn,
    double lr, int n_iters,
    double cong_mult, double den_mult, double rep_margin, double gamma_mult,
    double *out_proxy_best,
    /* Optional WL-only multiplier (NULL = identity = no critical-net weighting). */
    const double *wl_extra_weight
) {
    RS S;
    memset(&S, 0, sizeof(S));
    S.n = n;
    S.nh = nh;
    S.cw = cw;
    S.ch = ch;
    S.gr = gr;
    S.gc = gc;
    S.ng = gr * gc;
    S.gw = cw / (double)gc;
    S.gh = ch / (double)gr;
    S.inv_ba = 1.0 / (S.gw * S.gh);
    S.k_den = i_max(1, (int)(0.10 * S.ng));
    S.k_cong = i_max(1, (int)(0.05 * S.ng));
    S.sr = smooth_range;
    S.sizes = sizes;
    S.movable = movable;
    S.pos = pos;
    S.np = np_pins;
    S.pin_macro = pin_macro;
    S.pin_x = pin_x;
    S.pin_y = pin_y;
    S.nn = nn;
    S.total_pin_refs = (nn > 0) ? net_pin_off[nn] : 0;
    S.net_pin_off = net_pin_off;
    S.net_pin_idx = net_pin_idx;
    S.net_weight = net_weight;
    S.wl_extra_weight = wl_extra_weight;
    S.gamma_base = (S.gw + S.gh) * gamma_mult;
    S.beta1 = 0.9;
    S.beta2 = 0.999;
    S.eps = 1e-8;
    S.beta1_pow = 1.0;
    S.beta2_pow = 1.0;
#ifdef _OPENMP
    S.max_threads = omp_get_max_threads();
#else
    S.max_threads = 1;
#endif

    S.deg_scale = (double *)malloc(sizeof(double) * (size_t)(nn > 0 ? nn : 1));
    for (int k = 0; k < nn; k++) {
        int len = net_pin_off[k + 1] - net_pin_off[k];
        double v = (double)(len - 1);
        if (v < 1.0) v = 1.0;
        S.deg_scale[k] = sqrt(v);
    }

    S.pin_abs = (double *)calloc((size_t)(np_pins > 0 ? np_pins : 1) * 2u, sizeof(double));

    S.mov_idx = (int *)malloc(sizeof(int) * (size_t)(n > 0 ? n : 1));
    S.mov_rev = (int *)malloc(sizeof(int) * (size_t)(n > 0 ? n : 1));
    for (int i = 0; i < n; i++) S.mov_rev[i] = -1;
    S.n_mov = 0;
    for (int i = 0; i < n; i++) {
        if (movable[i]) {
            S.mov_rev[i] = S.n_mov;
            S.mov_idx[S.n_mov++] = i;
        }
    }

    S.grad = (double *)calloc((size_t)(S.n_mov > 0 ? S.n_mov : 1) * 2u, sizeof(double));

    int nnx = nn > 0 ? nn : 1;
    int ne2 = (S.total_pin_refs > 0 ? S.total_pin_refs : 1) * 2;
    S.wa_hi = (double *)calloc((size_t)nnx * 2u, sizeof(double));
    S.wa_lo = (double *)calloc((size_t)nnx * 2u, sizeof(double));
    S.sum_e_hi = (double *)calloc((size_t)nnx * 2u, sizeof(double));
    S.sum_e_lo = (double *)calloc((size_t)nnx * 2u, sizeof(double));
    S.mx_hi = (double *)calloc((size_t)nnx * 2u, sizeof(double));
    S.mx_lo = (double *)calloc((size_t)nnx * 2u, sizeof(double));
    S.inv_gamma_net = (double *)calloc((size_t)nnx, sizeof(double));
    S.total_span = (double *)calloc((size_t)nnx, sizeof(double));
    S.e_hi_cache = (double *)calloc((size_t)ne2, sizeof(double));
    S.e_lo_cache = (double *)calloc((size_t)ne2, sizeof(double));

    S.den_map = (double *)calloc((size_t)S.ng, sizeof(double));
    S.den_map_grad = (double *)calloc((size_t)S.ng, sizeof(double));
    S.cong_dm = (double *)calloc((size_t)S.ng, sizeof(double));
    S.cong_sm = (double *)calloc((size_t)S.ng, sizeof(double));
    S.cong_sm_grad = (double *)calloc((size_t)S.ng, sizeof(double));
    S.cong_dm_grad = (double *)calloc((size_t)S.ng, sizeof(double));

    S.adm_m = (double *)calloc((size_t)(S.n_mov > 0 ? S.n_mov : 1) * 2u, sizeof(double));
    S.adm_v = (double *)calloc((size_t)(S.n_mov > 0 ? S.n_mov : 1) * 2u, sizeof(double));
    S.adm_t = 0;

    S.tls_grad = (double *)calloc((size_t)S.max_threads * (size_t)(S.n_mov > 0 ? S.n_mov : 1) * 2u, sizeof(double));
    S.tls_den_map = (double *)calloc((size_t)S.max_threads * (size_t)S.ng, sizeof(double));
    S.tls_cong_dm = (double *)calloc((size_t)S.max_threads * (size_t)S.ng, sizeof(double));
    S.smooth_tmp = (double *)calloc((size_t)S.ng, sizeof(double));
    S.topk_idx_den = (int *)malloc(sizeof(int) * (size_t)(S.k_den > 0 ? S.k_den : 1));
    S.topk_idx_cong = (int *)malloc(sizeof(int) * (size_t)(S.k_cong > 0 ? S.k_cong : 1));
    int topk_scratch_n = i_max(S.k_den, S.k_cong);
    S.topk_heap = (double *)malloc(sizeof(double) * (size_t)(topk_scratch_n > 0 ? topk_scratch_n : 1));

    double *lo_bounds = (double *)malloc(sizeof(double) * (size_t)(S.n_mov > 0 ? S.n_mov : 1) * 2u);
    double *hi_bounds = (double *)malloc(sizeof(double) * (size_t)(S.n_mov > 0 ? S.n_mov : 1) * 2u);
    for (int i = 0; i < S.n_mov; i++) {
        int m = S.mov_idx[i];
        double hw = sizes[m * 2 + 0] * 0.5;
        double hh = sizes[m * 2 + 1] * 0.5;
        lo_bounds[i * 2 + 0] = hw;
        hi_bounds[i * 2 + 0] = cw - hw;
        lo_bounds[i * 2 + 1] = hh;
        hi_bounds[i * 2 + 1] = ch - hh;
    }

    double *best_pos = (double *)malloc(sizeof(double) * (size_t)n * 2u);
    memcpy(best_pos, pos, sizeof(double) * (size_t)n * 2u);
    double best_eval = 1e300;
    double wl_s = 1.0, den_s = 0.01, cong_s = 0.01;

    for (int step = 0; step < n_iters; step++) {
        double frac = (n_iters > 1) ? (double)step / (double)(n_iters - 1) : 0.0;
        memset(S.grad, 0, sizeof(double) * (size_t)S.n_mov * 2u);

        refresh_pin_abs(&S);
        wl_forward_pass1(&S);
        double wl = wl_forward_pass2(&S);
        double den = 0.0, cong = 0.0;
        density_forward(&S, &den);
        if (cong_mult > 0.0) cong_forward(&S, &cong);

        if (step == 0) {
            wl_s = (wl > 1.0) ? wl : 1.0;
            den_s = (den > 0.01) ? den : 0.01;
            cong_s = (cong > 0.01) ? cong : 0.01;
        }

        double w_wl = 1.0 / wl_s;
        double w_den = den_mult * (0.25 + 2.0 * frac) / den_s;
        double w_cong = (cong_mult > 0.0)
            ? cong_mult * (0.5 + 2.0 * frac) / cong_s
            : 0.0;
        double w_rep = (8.0 + 12.0 * frac) / 0.01;
        double rm = rep_margin * frac;

        wl_backward(&S, S.grad, w_wl);
        density_backward(&S, w_den);
        if (cong_mult > 0.0) cong_backward(&S, S.grad, w_cong);
        double rep = rep_forward_and_grad(&S, rm, S.grad, w_rep);

        double loss = w_wl * wl + w_den * den + w_cong * cong + w_rep * rep;
        if (!(loss == loss) || loss > 1e30 || loss < -1e30) break;

        double ev = wl / wl_s + 0.5 * den / den_s
                  + 0.5 * cong / (cong_s > 0.01 ? cong_s : 0.01);
        if (ev < best_eval) {
            best_eval = ev;
            memcpy(best_pos, pos, sizeof(double) * (size_t)n * 2u);
        }

        double step_lr = lr * 0.1
                       + 0.5 * (lr - lr * 0.1)
                         * (1.0 + cos(M_PI * (double)step / (double)n_iters));
        adam_step(&S, S.grad, step_lr, lo_bounds, hi_bounds);
    }

    memcpy(pos, best_pos, sizeof(double) * (size_t)n * 2u);
    if (out_proxy_best) *out_proxy_best = best_eval;

    free(best_pos);
    free(lo_bounds);
    free(hi_bounds);
    free(S.topk_heap);
    free(S.topk_idx_den);
    free(S.topk_idx_cong);
    free(S.smooth_tmp);
    free(S.tls_cong_dm);
    free(S.tls_den_map);
    free(S.tls_grad);
    free(S.adm_m);
    free(S.adm_v);
    free(S.mov_idx);
    free(S.mov_rev);
    free(S.grad);
    free(S.wa_hi);
    free(S.wa_lo);
    free(S.sum_e_hi);
    free(S.sum_e_lo);
    free(S.mx_hi);
    free(S.mx_lo);
    free(S.inv_gamma_net);
    free(S.total_span);
    free(S.e_hi_cache);
    free(S.e_lo_cache);
    free(S.den_map);
    free(S.den_map_grad);
    free(S.cong_dm);
    free(S.cong_sm);
    free(S.cong_sm_grad);
    free(S.cong_dm_grad);
    free(S.deg_scale);
    free(S.pin_abs);
}