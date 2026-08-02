"""Xplace GPU placer entry point: LEF/DEF gen -> Xplace GP -> refinement
cascade -> proxy eval (proxy = wirelength + 0.5*density + 0.5*congestion)."""

import copy
import ctypes
import gc
import logging
import os
import pickle
import shutil
import subprocess
import struct
import sys
import threading
import time
from dataclasses import dataclass

# Reduce CUDA allocator fragmentation - best-of-N runs several full pipelines
# back-to-back in one process; on an 8GB GPU that fragments and OOMs without
# this. Must be set before torch initializes CUDA. No effect on results.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

# Make `abuplace.*` importable when this file is loaded directly,
# e.g. via importlib in the eval harness.
_PLACER_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_PLACER_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from macro_place.benchmark import Benchmark


_EXT_DIR = os.path.join(_PLACER_DIR, "extensions")

# Persistent subprocess worker driving parallel cong_relax_v2 polishes.
_ILS_POLISH_WORKER = os.path.join(_PLACER_DIR, "_ils_polish_worker.py")

_DBL_PTR = ctypes.POINTER(ctypes.c_double)
_INT_PTR = ctypes.POINTER(ctypes.c_int)


def _find_cc():
    """Locate a C compiler for the extension build: $CC first, then the usual
    names on PATH. Returns an absolute path, or None if nothing is available."""
    cands = [c for c in (os.environ.get("CC"), "cc", "gcc", "clang") if c]
    for cand in cands:
        if os.path.sep in cand:
            if os.path.exists(cand):
                return cand
            continue
        p = shutil.which(cand)
        if p:
            return p
    return None


# Argument groups shared by more than one C entry point. Naming them keeps the
# two declarations that use each group from drifting apart.

# The placement problem itself: geometry, grid, pins and nets. Identical for
# cong_relax_v2 and cong_state_create.
_CONG_PROBLEM_ARGS = [
    _DBL_PTR, _DBL_PTR, _INT_PTR,      # pos, sizes, movable
    ctypes.c_int, ctypes.c_int,        # n, nh
    ctypes.c_double, ctypes.c_double,  # cw, ch
    ctypes.c_int, ctypes.c_int,        # gr, gc
    ctypes.c_int,                      # smooth_range
    ctypes.c_double, ctypes.c_double,  # hrpm, vrpm
    ctypes.c_double, ctypes.c_double,  # h_alloc, v_alloc
    _INT_PTR, _DBL_PTR, _DBL_PTR,      # pin_macro, pin_x, pin_y
    ctypes.c_int,                      # np
    _INT_PTR, _INT_PTR, _INT_PTR,      # net_driver, net_sinks_off, net_sinks_idx
    _DBL_PTR,                          # net_weight
    ctypes.c_int,                      # nn
    ctypes.c_double,                   # net_cnt_for_norm
    _DBL_PTR,                          # wl_extra_weight (NULL = identity)
]

# Geometry the spiral legalizer needs. Identical for legalize and legalize_v2.
_LEGALIZE_ARGS = [
    _DBL_PTR,                          # pos
    ctypes.c_int,                      # nh
    _DBL_PTR,                          # sizes
    _INT_PTR,                          # movable
    ctypes.c_double, ctypes.c_double,  # cw, ch
    _DBL_PTR, _DBL_PTR,                # hw, hh
    ctypes.c_double,                   # step_factor
]

# The (wl, den, cong) out-parameter triple every cong_state_* scorer writes.
_CONG_STATE_OUT_ARGS = [_DBL_PTR, _DBL_PTR, _DBL_PTR]

# Per-extension build config: src/so basenames, extra cflags, and fn name ->
# (restype, argtypes).
_C_EXTENSIONS = {
    "leg": {
        "src": "legalize.c", "so": "liblegalize.so",
        "cflags": ["-flto", "-funroll-loops", "-fno-plt"],
        "fns": {
            "legalize": (None, [
                *_LEGALIZE_ARGS,
                _INT_PTR,                          # had_overlap (out)
            ]),
            "legalize_v2": (None, [
                *_LEGALIZE_ARGS,
                ctypes.c_int,                      # post_passes
                ctypes.c_int,                      # swap_passes
                ctypes.c_int,                      # swap_top_k
                _INT_PTR,                          # had_overlap (out)
            ]),
        },
    },
    "cong": {
        "src": "congestion.c", "so": "libcongestion.so",
        "cflags": ["-fopenmp", "-flto", "-funroll-loops", "-fno-plt"],
        "fns": {
            "cong_relax_v2": (None, [
                *_CONG_PROBLEM_ARGS,
                ctypes.c_int, ctypes.c_double,     # n_iter, lr_bins
                _DBL_PTR, _DBL_PTR,                # out_cong_init/final
                _DBL_PTR, _DBL_PTR,                # out_proxy_init/final
                _DBL_PTR, _DBL_PTR,                # out_wl_final, out_den_final
                ctypes.c_int,                      # hotspot_sort
                ctypes.c_int,                      # n_extra_phases
                _INT_PTR, _DBL_PTR,                # extra_n_iters, extra_lr_bins
            ]),
            # Incremental scoring API around a state handle.
            "cong_state_create": (ctypes.c_void_p, [
                *_CONG_PROBLEM_ARGS,
            ]),
            "cong_state_destroy": (None, [ctypes.c_void_p]),
            "cong_state_score_current": (ctypes.c_double, [
                ctypes.c_void_p,
                *_CONG_STATE_OUT_ARGS,             # out_wl, out_den, out_cong
            ]),
            "cong_state_apply": (ctypes.c_double, [
                ctypes.c_void_p,
                ctypes.c_int,                      # n_changes
                _INT_PTR, _DBL_PTR, _DBL_PTR,      # macros, new_x, new_y
                *_CONG_STATE_OUT_ARGS,             # out_wl, out_den, out_cong
            ]),
            "cong_state_revert": (ctypes.c_double, [
                ctypes.c_void_p,
                *_CONG_STATE_OUT_ARGS,             # out_wl, out_den, out_cong
            ]),
            "cong_state_commit": (None, [ctypes.c_void_p]),
            "cong_state_get_pos": (None, [ctypes.c_void_p, _DBL_PTR]),
            "cong_state_rebuild": (None, [ctypes.c_void_p]),
        },
    },
    "refine": {
        "src": "refine.c", "so": "librefine.so",
        "cflags": ["-ffast-math", "-fopenmp",
                   "-flto", "-funroll-loops", "-fno-plt"],
        "fns": {
            "refine_adam": (None, [
                _DBL_PTR,                          # pos
                ctypes.c_int, ctypes.c_int,        # n, nh
                _DBL_PTR, _INT_PTR,                # sizes, movable
                ctypes.c_double, ctypes.c_double,  # cw, ch
                ctypes.c_int, ctypes.c_int,        # gr, gc
                ctypes.c_int,                      # smooth_range
                ctypes.c_int,                      # np_pins
                _INT_PTR, _DBL_PTR, _DBL_PTR,      # pin_macro, pin_x, pin_y
                _INT_PTR, _INT_PTR, _DBL_PTR,      # net_pin_off, net_pin_idx, net_weight
                ctypes.c_int,                      # nn
                ctypes.c_double, ctypes.c_int,     # lr, n_iters
                ctypes.c_double, ctypes.c_double,  # cong_mult, den_mult
                ctypes.c_double, ctypes.c_double,  # rep_margin, gamma_mult
                _DBL_PTR,                          # out_proxy_best
                _DBL_PTR,                          # wl_extra_weight (NULL = identity)
            ]),
        },
    },
}

_loaded_libs = {}


def _ensure_lib(key):
    """Build (if stale) and dlopen the named C extension; cache and return."""
    if key in _loaded_libs:
        return _loaded_libs[key]
    cfg = _C_EXTENSIONS[key]
    src = os.path.join(_EXT_DIR, cfg["src"])
    so = os.path.join(_EXT_DIR, cfg["so"])
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        cc = _find_cc()
        if cc is None:
            raise RuntimeError(f"No C compiler found for {cfg['so']} build")
        subprocess.run(
            [cc, "-O3", "-march=native", "-fPIC", "-shared",
             *cfg["cflags"], "-o", so, src, "-lm"],
            check=True)
    lib = ctypes.CDLL(so)
    for fn_name, (restype, argtypes) in cfg["fns"].items():
        fn = getattr(lib, fn_name)
        fn.restype = restype
        fn.argtypes = argtypes
    _loaded_libs[key] = lib
    return lib


def _refine_adam_c(pos_np, n, nh, sizes, movable, cw, ch, gr, gc,
                   smooth_range,
                   pin_macro, pin_x, pin_y,
                   net_pin_off, net_pin_idx, net_weight,
                   lr, n_iters, cong_mult, den_mult, rep_margin, gamma_mult,
                   wl_extra_weight=None):
    """C Adam refinement on pin-true WL + density + congestion + repulsion;
    wl_extra_weight (optional ndarray[nn]) scales only the WL
    forward/backward, leaving routing/density untouched."""
    lib = _ensure_lib("refine")
    pos_c = np.ascontiguousarray(pos_np, dtype=np.float64).copy()
    sizes_c = np.ascontiguousarray(sizes, dtype=np.float64)
    mov_c = np.ascontiguousarray(movable, dtype=np.int32)
    pm_c = np.ascontiguousarray(pin_macro, dtype=np.int32)
    px_c = np.ascontiguousarray(pin_x, dtype=np.float64)
    py_c = np.ascontiguousarray(pin_y, dtype=np.float64)
    npo_c = np.ascontiguousarray(net_pin_off, dtype=np.int32)
    npi_c = np.ascontiguousarray(net_pin_idx, dtype=np.int32)
    nw_c = np.ascontiguousarray(net_weight, dtype=np.float64)
    nn = npo_c.shape[0] - 1
    np_pins = pm_c.shape[0]
    if wl_extra_weight is not None:
        wxw_c = np.ascontiguousarray(wl_extra_weight, dtype=np.float64)
        wxw_ptr = wxw_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        wxw_c = None
        wxw_ptr = None
    out_proxy = ctypes.c_double(0.0)
    lib.refine_adam(
        pos_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(n), ctypes.c_int(nh),
        sizes_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mov_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_double(cw), ctypes.c_double(ch),
        ctypes.c_int(gr), ctypes.c_int(gc),
        ctypes.c_int(smooth_range),
        ctypes.c_int(np_pins),
        pm_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        px_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        py_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        npo_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        npi_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        nw_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(nn),
        ctypes.c_double(lr), ctypes.c_int(n_iters),
        ctypes.c_double(cong_mult), ctypes.c_double(den_mult),
        ctypes.c_double(rep_margin), ctypes.c_double(gamma_mult),
        ctypes.byref(out_proxy),
        wxw_ptr,
    )
    return pos_c


def _cong_relax_v2_c(pos, sizes, movable, n, nh, cw, ch, gr, gc,
                     smooth_range, hrpm, vrpm, h_alloc, v_alloc,
                     pin_macro, pin_x, pin_y,
                     net_driver, net_sinks_off, net_sinks_idx, net_weight,
                     net_cnt_for_norm, n_iter, lr_bins, hotspot_sort=1,
                     extra_phases=None, wl_extra_weight=None):
    """C cong_relax_v2 wrapper that gates accepts on the full external proxy
    (wl + 0.5*den + 0.5*cong); extra_phases appends (n_iter, lr_bins) phases
    sharing one init_state; returns (pos, cong_init, cong_final, proxy_init,
    proxy_final, wl_final, den_final)."""
    lib = _ensure_lib("cong")
    pos_c = np.ascontiguousarray(pos, dtype=np.float64).copy()
    sizes_c = np.ascontiguousarray(sizes, dtype=np.float64)
    mov_c = np.ascontiguousarray(movable, dtype=np.int32)
    pm_c = np.ascontiguousarray(pin_macro, dtype=np.int32)
    px_c = np.ascontiguousarray(pin_x, dtype=np.float64)
    py_c = np.ascontiguousarray(pin_y, dtype=np.float64)
    nd_c = np.ascontiguousarray(net_driver, dtype=np.int32)
    nso_c = np.ascontiguousarray(net_sinks_off, dtype=np.int32)
    nsi_c = np.ascontiguousarray(net_sinks_idx, dtype=np.int32)
    nw_c = np.ascontiguousarray(net_weight, dtype=np.float64)
    np_pins = pm_c.shape[0]
    nn = nd_c.shape[0]
    if wl_extra_weight is not None:
        wxw_c = np.ascontiguousarray(wl_extra_weight, dtype=np.float64)
        wxw_ptr = wxw_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        wxw_c = None
        wxw_ptr = None
    out_cong_init = ctypes.c_double(0.0)
    out_cong_final = ctypes.c_double(0.0)
    out_proxy_init = ctypes.c_double(0.0)
    out_proxy_final = ctypes.c_double(0.0)
    out_wl_final = ctypes.c_double(0.0)
    out_den_final = ctypes.c_double(0.0)
    if extra_phases:
        extra_iters_np = np.ascontiguousarray(
            [int(p[0]) for p in extra_phases], dtype=np.int32)
        extra_lrs_np = np.ascontiguousarray(
            [float(p[1]) for p in extra_phases], dtype=np.float64)
        extra_iters_ptr = extra_iters_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        extra_lrs_ptr = extra_lrs_np.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        n_extra = len(extra_phases)
    else:
        # Keep refs alive - C reads through the pointers below.
        extra_iters_np = None
        extra_lrs_np = None
        extra_iters_ptr = None
        extra_lrs_ptr = None
        n_extra = 0
    lib.cong_relax_v2(
        pos_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sizes_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mov_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_int(n), ctypes.c_int(nh),
        ctypes.c_double(cw), ctypes.c_double(ch),
        ctypes.c_int(gr), ctypes.c_int(gc),
        ctypes.c_int(smooth_range),
        ctypes.c_double(hrpm), ctypes.c_double(vrpm),
        ctypes.c_double(h_alloc), ctypes.c_double(v_alloc),
        pm_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        px_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        py_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(np_pins),
        nd_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        nso_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        nsi_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        nw_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(nn),
        ctypes.c_double(net_cnt_for_norm),
        wxw_ptr,
        ctypes.c_int(n_iter),
        ctypes.c_double(lr_bins),
        ctypes.byref(out_cong_init),
        ctypes.byref(out_cong_final),
        ctypes.byref(out_proxy_init),
        ctypes.byref(out_proxy_final),
        ctypes.byref(out_wl_final),
        ctypes.byref(out_den_final),
        ctypes.c_int(int(hotspot_sort)),
        ctypes.c_int(n_extra),
        extra_iters_ptr,
        extra_lrs_ptr,
    )
    return (pos_c, out_cong_init.value, out_cong_final.value,
            out_proxy_init.value, out_proxy_final.value,
            out_wl_final.value, out_den_final.value)


class CongState:
    """Live cong_relax_v2 state with incremental score/apply/commit/revert; use
    as a context manager so the C handle is freed; per-probe cost drops from
    a 5-15ms full rebuild to 0.5-2ms incremental routing."""

    def __init__(self, pos, sizes, movable, n, nh, cw, ch, octx, pn,
                 wl_extra_weight=None):
        lib = _ensure_lib("cong")
        # Keep refs alive until cong_state_create returns (C copies internally
        # afterward).
        self._pos = np.ascontiguousarray(pos, dtype=np.float64)
        self._sizes = np.ascontiguousarray(sizes, dtype=np.float64)
        self._mov = np.ascontiguousarray(movable, dtype=np.int32)
        self._pm = np.ascontiguousarray(pn['pin_macro'], dtype=np.int32)
        self._px = np.ascontiguousarray(pn['pin_x'], dtype=np.float64)
        self._py = np.ascontiguousarray(pn['pin_y'], dtype=np.float64)
        self._nd = np.ascontiguousarray(pn['net_driver'], dtype=np.int32)
        self._nso = np.ascontiguousarray(pn['net_sinks_off'], dtype=np.int32)
        self._nsi = np.ascontiguousarray(pn['net_sinks_idx'], dtype=np.int32)
        self._nw = np.ascontiguousarray(pn['net_weight'], dtype=np.float64)
        if wl_extra_weight is not None:
            self._wxw = np.ascontiguousarray(wl_extra_weight, dtype=np.float64)
            wxw_ptr = self._wxw.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        else:
            self._wxw = None
            wxw_ptr = None
        self._n = int(n); self._nh = int(nh)
        self._lib = lib
        self._handle = lib.cong_state_create(
            self._pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._mov.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(self._n), ctypes.c_int(self._nh),
            ctypes.c_double(cw), ctypes.c_double(ch),
            ctypes.c_int(int(octx['gr'])), ctypes.c_int(int(octx['gc'])),
            ctypes.c_int(int(pn['smooth_range'])),
            ctypes.c_double(pn['hrpm']), ctypes.c_double(pn['vrpm']),
            ctypes.c_double(pn['h_alloc']), ctypes.c_double(pn['v_alloc']),
            self._pm.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._px.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._py.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(self._pm.shape[0]),
            self._nd.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._nso.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._nsi.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._nw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(self._nd.shape[0]),
            ctypes.c_double(float(pn['net_cnt_for_norm'])),
            wxw_ptr,
        )
        if not self._handle:
            raise RuntimeError("cong_state_create returned NULL")
        # Scratch outputs (reused across calls)
        self._out_wl = ctypes.c_double(0.0)
        self._out_den = ctypes.c_double(0.0)
        self._out_cong = ctypes.c_double(0.0)
        # Default scratch buffer for typical 1-macro applies; resized as needed.
        self._mac_buf = np.zeros(8, dtype=np.int32)
        self._nx_buf = np.zeros(8, dtype=np.float64)
        self._ny_buf = np.zeros(8, dtype=np.float64)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._handle:
            self._lib.cong_state_destroy(self._handle)
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def score(self):
        """Current full proxy. Returns (proxy, wl, den, cong)."""
        proxy = self._lib.cong_state_score_current(
            self._handle,
            ctypes.byref(self._out_wl),
            ctypes.byref(self._out_den),
            ctypes.byref(self._out_cong))
        return proxy, self._out_wl.value, self._out_den.value, self._out_cong.value

    def _ensure_buf(self, n):
        if n > self._mac_buf.shape[0]:
            self._mac_buf = np.zeros(n, dtype=np.int32)
            self._nx_buf = np.zeros(n, dtype=np.float64)
            self._ny_buf = np.zeros(n, dtype=np.float64)

    def apply(self, macros, new_x, new_y):
        """Apply position changes and stash old positions for one revert (next
        apply replaces the revert log); returns (proxy, wl, den, cong)."""
        n = len(macros)
        self._ensure_buf(n)
        self._mac_buf[:n] = macros
        self._nx_buf[:n] = new_x
        self._ny_buf[:n] = new_y
        proxy = self._lib.cong_state_apply(
            self._handle, ctypes.c_int(n),
            self._mac_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            self._nx_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            self._ny_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(self._out_wl),
            ctypes.byref(self._out_den),
            ctypes.byref(self._out_cong))
        return proxy, self._out_wl.value, self._out_den.value, self._out_cong.value

    def revert(self):
        """Undo the most recent apply. Returns the restored proxy."""
        proxy = self._lib.cong_state_revert(
            self._handle,
            ctypes.byref(self._out_wl),
            ctypes.byref(self._out_den),
            ctypes.byref(self._out_cong))
        return proxy, self._out_wl.value, self._out_den.value, self._out_cong.value

    def commit(self):
        """Discard the revert log; the most recent apply becomes permanent."""
        self._lib.cong_state_commit(self._handle)

    def rebuild(self):
        """Force a full rebuild of state to flush FP drift."""
        self._lib.cong_state_rebuild(self._handle)

    def get_pos(self):
        """Return current positions as an (n, 2) float64 array."""
        out = np.empty((self._n, 2), dtype=np.float64)
        self._lib.cong_state_get_pos(
            self._handle,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return out


# ILS subprocess pool config (parallel cong_relax_v2 workers for shake verify).
# Keep OMP_NUM_THREADS < 6: above that libgomp changes its reduction order,
# which perturbs the summed cost and regresses some designs.
_ILS_POLISH_K = int(os.environ.get("XP_ILS_K", "4"))
_ILS_POLISH_OMP = int(os.environ.get("XP_ILS_OMP", "4"))

@dataclass(frozen=True)
class DesignContext:
    """Everything about the design that never changes during a trajectory.

    The stage-4 operator passes each needed the same eleven arguments; bundling
    them keeps their signatures readable and makes it impossible to pass the
    wrong one positionally.
    """
    n_macros: int
    nh: int
    sizes: np.ndarray
    mov_i32: np.ndarray
    cw: float
    ch: float
    octx: dict
    pn: dict
    wl_extra_weight: object = None


# The two-phase decay tail every cc-reconcile appends: a short pass at 0.15
# bins, then a shorter one at 0.05. This is what settles the incremental
# CongState score back against a fresh evaluator recompute. Shared by all four
# reconcile sites so their schedules cannot drift apart. Read-only.
_CC_RECONCILE_PHASES = ((3, 0.15), (3, 0.05))

# Stage 4a post-polish hard-macro shake knobs.
_SHAKE_TOP_K = int(os.environ.get("XP_SHAKE_TOP_K", "60"))
_SHAKE_STEP_REL = float(os.environ.get("XP_SHAKE_STEP_REL", "0.125"))
_SHAKE_VERIFY_N = int(os.environ.get("XP_SHAKE_VERIFY_N", "4"))
_SHAKE_BUDGET_S = float(os.environ.get("XP_SHAKE_BUDGET_S", "3.5"))

# Low-cong benchmarks (c_probe < gate) get scaled-down R2/R3 cong_mult.
_ADAPT_C_PROBE_GATE = float(os.environ.get("XP_ADAPT_GATE", "1.85"))
_ADAPT_R2_SCALE = float(os.environ.get("XP_ADAPT_R2", "0.4"))
_ADAPT_R3_SCALE = float(os.environ.get("XP_ADAPT_R3", "0.3"))

def _ils_pack_args(sizes, movable, benchmark, nh, cw, ch, octx, pn,
                  hotspot_sort, wl_extra_weight=None):
    """Pack cong_relax_v2's static args into a worker-pickleable dict; mirrors
    _cong_relax_v2_c kwargs."""
    args = dict(
        sizes=np.ascontiguousarray(sizes, dtype=np.float64),
        movable=np.ascontiguousarray(movable, dtype=np.int32),
        n=int(benchmark.num_macros), nh=int(nh),
        cw=float(cw), ch=float(ch),
        gr=int(octx['gr']), gc=int(octx['gc']),
        smooth_range=int(pn['smooth_range']),
        hrpm=float(pn['hrpm']), vrpm=float(pn['vrpm']),
        h_alloc=float(pn['h_alloc']), v_alloc=float(pn['v_alloc']),
        pin_macro=np.ascontiguousarray(pn['pin_macro'], dtype=np.int32),
        pin_x=np.ascontiguousarray(pn['pin_x'], dtype=np.float64),
        pin_y=np.ascontiguousarray(pn['pin_y'], dtype=np.float64),
        net_driver=np.ascontiguousarray(pn['net_driver'], dtype=np.int32),
        net_sinks_off=np.ascontiguousarray(pn['net_sinks_off'], dtype=np.int32),
        net_sinks_idx=np.ascontiguousarray(pn['net_sinks_idx'], dtype=np.int32),
        net_weight=np.ascontiguousarray(pn['net_weight'], dtype=np.float64),
        net_cnt_for_norm=float(pn['net_cnt_for_norm']),
        hotspot_sort=int(hotspot_sort),
    )
    if wl_extra_weight is not None:
        args["wl_extra_weight"] = np.ascontiguousarray(
            wl_extra_weight, dtype=np.float64)
    return args


# Persistent K-worker ILS pool: spawned once per place() and reused via
# length-prefixed pickle frames over stdin/stdout - saves ~1.5s per request.
# Subprocesses, not threads: each worker needs its own libgomp runtime to get
# true K-way parallelism (a thread pool would share a single OMP team).
def _send_frame(stream, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack(">I", len(data)))
    stream.write(data)
    stream.flush()


def _recv_frame(stream):
    header = stream.read(4)
    if len(header) < 4:
        raise EOFError
    n = struct.unpack(">I", header)[0]
    data = bytearray()
    while len(data) < n:
        chunk = stream.read(n - len(data))
        if not chunk:
            raise EOFError
        data.extend(chunk)
    return pickle.loads(bytes(data))


class ILSWorkerPool:
    """Persistent K-worker subprocess pool for parallel cong_relax_v2; each
    worker owns its libgomp runtime giving true Kx parallelism
    (ThreadPoolExecutor would share one OMP team); async_start spawns
    concurrent with other work, wait_ready() blocks until all init acks
    arrive."""

    def __init__(self, args, k, omp_threads, so_path, async_start=False):
        self.k = int(k)
        self.so_path = so_path
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(omp_threads)
        env.setdefault("CUDA_VISIBLE_DEVICES", "")
        self._procs = []
        self._closed = False
        for i in range(self.k):
            p = subprocess.Popen(
                [sys.executable, _ILS_POLISH_WORKER],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=env)
            self._procs.append(p)
        init_payload = {"args": args, "so_path": so_path}
        for p in self._procs:
            _send_frame(p.stdin, init_payload)
        self._ready = False
        if not async_start:
            self.wait_ready()

    def wait_ready(self):
        """Block until all workers have ack'd init."""
        if self._ready:
            return
        for p in self._procs:
            try:
                ack = _recv_frame(p.stdout)
            except EOFError:
                err = (p.stderr.read().decode(errors="replace")[:500]
                       if p.stderr else "(no stderr)")
                raise RuntimeError(f"ILS worker died during init: {err}")
            if not (isinstance(ack, dict) and ack.get("status") == "ready"):
                raise RuntimeError(f"unexpected init ack: {ack!r}")
        self._ready = True

    def submit_verify(self, payloads):
        """Send N<=K verify requests (each payload has pos/n_iter/lr_bins) for
        true-parallel mini-polish verification."""
        self.wait_ready()
        if len(payloads) > self.k:
            raise ValueError(
                f"submit_verify: got {len(payloads)} payloads, max is {self.k}")
        self._pending_verify_n = len(payloads)
        for p, payload in zip(self._procs, payloads):
            _send_frame(p.stdin, payload)

    def collect_verify(self):
        """Read N responses matching submit_verify in submission order; entries
        are None for any worker that died."""
        n = getattr(self, '_pending_verify_n', 0)
        results = []
        for p in self._procs[:n]:
            try:
                r = _recv_frame(p.stdout)
            except EOFError:
                err = (p.stderr.read().decode(errors="replace")[:500]
                       if p.stderr else "(no stderr)")
                print(f"    [pool] verify-worker rc={p.returncode} died: {err}",
                      file=sys.stderr)
                results.append(None)
                continue
            results.append(r)
        self._pending_verify_n = 0
        return results

    def close(self):
        """Shut the worker pool down. Every failure here is swallowed on
        purpose: teardown runs from __del__ and from the end of a trajectory
        that may itself be unwinding, and a worker that already died is the
        normal case, not an error worth propagating."""
        if self._closed:
            return
        self._closed = True
        for p in self._procs:
            try:
                p.stdin.close()
            except Exception:
                pass  # already closed, or the worker exited first
        for p in self._procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
                try:
                    p.wait(timeout=2)
                except Exception:
                    pass  # killed and unreapable; nothing further to do
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        # Interpreter shutdown can already have torn down what close() touches.
        try:
            self.close()
        except Exception:
            pass


def _legalize_c(pos_hard, nh, sizes, movable, cw, ch, hw, hh, step_factor,
                  label="?"):
    """C spiral-search legalization; returns (pos, had_overlap)."""
    lib = _ensure_lib("leg")
    pos_hard = np.ascontiguousarray(pos_hard, dtype=np.float64)
    sizes_c = np.ascontiguousarray(sizes, dtype=np.float64)
    movable_c = np.ascontiguousarray(movable, dtype=np.int32)
    hw_c = np.ascontiguousarray(hw, dtype=np.float64)
    hh_c = np.ascontiguousarray(hh, dtype=np.float64)
    had_overlap = ctypes.c_int(0)
    lib.legalize(
        pos_hard.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(nh),
        sizes_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        movable_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        ctypes.c_double(cw), ctypes.c_double(ch),
        hw_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        hh_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(step_factor),
        ctypes.byref(had_overlap),
    )
    return pos_hard, bool(had_overlap.value)


# --- Tier-2 macro clearance (PDN channels) ---
# The Tier-2 evaluator pushes hard macros >=12 um apart for PDN channels
# (SCORING.md); mirroring that here keeps control of the final coordinates
# instead of letting its blind push reshuffle a dense layout. Canvas units are
# 1:1 microns at Tier-2, so the threshold applies directly. _clearance_applies()
# skips the tiny IBM canvases, where 12 would be nonsensical (ibm01 is ~23).

def _clearance_applies(clearance, cw, ch, sizes, nh, max_frac=0.10):
    """True only when `clearance` is physically sane for this design - i.e. small
    vs the canvas. Separates NG45 (mempool 12/885=1.4%) from IBM (12/23=52%)."""
    if nh <= 1 or clearance <= 0.0:
        return False
    return clearance <= max_frac * min(float(cw), float(ch))


def _hard_overlap_exists(pos, sizes, nh, eps=1e-6):
    """Any axis-aligned overlap among the first nh (hard) macros? Centers in
    pos."""
    for i in range(nh):
        hwi, hhi = sizes[i, 0] * 0.5, sizes[i, 1] * 0.5
        for j in range(i + 1, nh):
            hwj, hhj = sizes[j, 0] * 0.5, sizes[j, 1] * 0.5
            if (abs(pos[i, 0] - pos[j, 0]) < hwi + hwj - eps and
                    abs(pos[i, 1] - pos[j, 1]) < hhi + hhj - eps):
                return True
    return False


def _clearance_push(cur, sizes, nh, cw, ch, clearance, movable=None, n_iters=60):
    """Push hard-macro pairs apart to a >=`clearance` edge gap. Deterministic
    fixed-iteration relaxation: each pass resolves the smaller-penetration axis
    of every too-close pair, then clamps to canvas. Centers in/out; mirrors the
    Tier-2 evaluator's push.

    FIXED macros never move (that would trip validate_placement -> DQ): in a
    movable/fixed pair the movable one absorbs the full penetration, and
    both-fixed pairs are left as-is."""
    pos = cur.copy()
    if movable is None:
        mov = np.ones(nh, dtype=bool)
    else:
        mov = np.asarray(movable, dtype=bool)
    for _ in range(n_iters):
        moved = False
        for i in range(nh):
            hwi, hhi = sizes[i, 0] * 0.5, sizes[i, 1] * 0.5
            for j in range(i + 1, nh):
                if not mov[i] and not mov[j]:
                    continue  # both fixed - can't push either
                hwj, hhj = sizes[j, 0] * 0.5, sizes[j, 1] * 0.5
                req_x = hwi + hwj + clearance
                req_y = hhi + hhj + clearance
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                if abs(dx) < req_x and abs(dy) < req_y:
                    pen_x = req_x - abs(dx)
                    pen_y = req_y - abs(dy)
                    # Movable members share the push; a fixed member stays put.
                    wi = 0.5 if (mov[i] and mov[j]) else (1.0 if mov[i] else 0.0)
                    wj = 0.5 if (mov[i] and mov[j]) else (1.0 if mov[j] else 0.0)
                    if pen_x <= pen_y:
                        s = 1.0 if dx >= 0 else -1.0
                        pos[i, 0] += s * pen_x * wi
                        pos[j, 0] -= s * pen_x * wj
                    else:
                        s = 1.0 if dy >= 0 else -1.0
                        pos[i, 1] += s * pen_y * wi
                        pos[j, 1] -= s * pen_y * wj
                    moved = True
        for i in range(nh):
            if not mov[i]:
                continue  # never clamp (move) a fixed macro
            hwi, hhi = sizes[i, 0] * 0.5, sizes[i, 1] * 0.5
            pos[i, 0] = min(max(pos[i, 0], hwi), cw - hwi)
            pos[i, 1] = min(max(pos[i, 1], hhi), ch - hhi)
        if not moved:
            break
    return pos


# --- Tier-2 PDN horizontal-band channels ---
# The pairwise push guarantees a pairwise edge gap, not continuous, aligned
# channels: adjacent rows stagger, so a 12um layout still pinches, and any pinch
# below the PDN strap minimum fails pdngen with `[ERROR PDN-0179]`. The reflow
# below packs dense layouts into clean rows separated by a wide channel.
# NOTE: empirically this does NOT resolve PDN-0179 on ariane136 - that failure
# is a die/core-area artifact, independent of macro placement.


def _pdn_min_channels(pos, sizes, nh, eps=1e-6):
    """Narrowest facing channel among hard macros, per axis. A pair faces
    across a horizontal channel when their x-projections overlap with a positive
    vertical gap (where a PDN strap would route); symmetrically for vertical.
    Returns (min_horiz, min_vert), inf if no facing pair on that axis. Abutted
    and overlapping pairs are ignored. O(nh^2), sub-ms at nh<=136."""
    min_h_band = float('inf')   # smallest vertical gap between x-overlapping macros
    min_v_band = float('inf')   # smallest horizontal gap between y-overlapping macros
    for i in range(nh):
        xi, yi = pos[i, 0], pos[i, 1]
        hwi, hhi = sizes[i, 0] * 0.5, sizes[i, 1] * 0.5
        for j in range(i + 1, nh):
            hwj, hhj = sizes[j, 0] * 0.5, sizes[j, 1] * 0.5
            x_over = abs(xi - pos[j, 0]) < hwi + hwj - eps
            y_over = abs(yi - pos[j, 1]) < hhi + hhj - eps
            if x_over and not y_over:
                vgap = abs(yi - pos[j, 1]) - (hhi + hhj)
                if vgap > eps:
                    min_h_band = min(min_h_band, vgap)
            elif y_over and not x_over:
                hgap = abs(xi - pos[j, 0]) - (hwi + hwj)
                if hgap > eps:
                    min_v_band = min(min_v_band, hgap)
    return min_h_band, min_v_band


def _row_snap_pdn(cur, sizes, nh, cw, ch, channel_um, min_hgap=12.0,
                  movable=None):
    """Reflow the hard macros into horizontal rows separated by a uniform
    vertical channel of >=`channel_um`. Returns new centers [N,2] (soft macros
    untouched), or None if they cannot be tiled at this channel width, if any
    hard macro is FIXED (can't reflow around an arbitrary obstacle), or if a row
    doesn't fit - the caller falls back in every case.

    Displacement is kept minimal to limit the wirelength cost: rows follow
    current y-order and members current x-order, the row count R is the LARGEST
    still yielding a channel >= channel_um (least vertical spreading), and
    within a row x is spread only enough to enforce min_hgap. Row pitch uses the
    GLOBAL max macro height so mixed-height designs get a conservative channel."""
    if nh <= 1:
        return None
    mov = (np.ones(nh, dtype=bool) if movable is None
           else np.asarray(movable, dtype=bool))
    if not mov[:nh].all():
        return None  # fixed hard macro present - don't restructure around it
    max_h = float(sizes[:nh, 1].max())
    max_w = float(sizes[:nh, 0].max())
    # Largest R whose uniform channel (ch - R*max_h)/(R+1) >= channel_um, capped
    # so the per-row column count still tiles horizontally with min_hgap.
    cols_cap = max(1, int((cw - min_hgap) // (max_w + min_hgap)))
    best_R = 0
    for R in range(1, nh + 1):
        chan = (ch - R * max_h) / (R + 1)
        if chan < channel_um:
            break  # channel only shrinks as R grows
        cols = (nh + R - 1) // R
        if cols <= cols_cap:
            best_R = R
    if best_R == 0:
        return None  # can't fit even one row's worth at this channel width
    R = best_R
    chan = (ch - R * max_h) / (R + 1)
    out = cur.copy()
    order = sorted(range(nh), key=lambda k: (cur[k, 1], cur[k, 0]))
    # Even row counts: distribute nh macros across R rows (early rows get the +1).
    base, extra = divmod(nh, R)
    row_sizes = [base + (1 if r < extra else 0) for r in range(R)]
    pos_in_order = 0
    for r in range(R):
        cnt = row_sizes[r]
        members = order[pos_in_order:pos_in_order + cnt]
        pos_in_order += cnt
        # Row band center y: top margin `chan`, then alternating macro/channel.
        y_c = chan + r * (max_h + chan) + max_h * 0.5
        # 1D fixed-order legalization within the row (Abacus-style): keep x-order
        # and stay near current x while guaranteeing >=min_hgap gaps in canvas.
        # Forward pass enforces the left neighbour + left edge, backward pass the
        # right; together they pack overlap-free whenever the row fits. If it
        # doesn't (leftmost forced below its half-width), bail so the caller
        # falls back rather than emitting an overlap.
        members.sort(key=lambda k: cur[k, 0])
        m = len(members)
        hwm = [sizes[k, 0] * 0.5 for k in members]
        xs = [float(cur[k, 0]) for k in members]
        # forward: x[i] >= max(desired, left-neighbour right + gap + hw); +
        # left edge
        for idx in range(m):
            lo = hwm[idx] if idx == 0 else xs[idx - 1] + hwm[idx - 1] + min_hgap + hwm[idx]
            if xs[idx] < lo:
                xs[idx] = lo
        # backward: x[i] <= right-neighbour left - gap - hw; + right edge
        if xs[m - 1] > cw - hwm[m - 1]:
            xs[m - 1] = cw - hwm[m - 1]
        for idx in range(m - 2, -1, -1):
            hi = xs[idx + 1] - hwm[idx + 1] - min_hgap - hwm[idx]
            if xs[idx] > hi:
                xs[idx] = hi
        if xs[0] < hwm[0] - 1e-6:
            return None  # row does not fit at this min_hgap -> caller falls back
        for k, x in zip(members, xs):
            out[k, 0] = x
            out[k, 1] = y_c
    return out


def _best_effort_legal_tensor(benchmark):
    """Guaranteed-non-raising LEGAL fallback placement, used only when the full
    pipeline cannot produce one (every best-of-N trajectory crashed, e.g. OOM on a
    large design, or no CUDA at all). Legalizes the INPUT placement - resolves
    hard-macro overlaps among MOVABLE macros and clamps to canvas; fixed macros
    stay put. A weaker-but-VALID placement beats a crash / None return (which is
    a guaranteed DQ). Returns float32 [N, 2]."""
    try:
        pos = benchmark.macro_positions.detach().cpu().numpy().astype(np.float64).copy()
        sizes = benchmark.macro_sizes.detach().cpu().numpy().astype(np.float64)
        nh = int(benchmark.num_hard_macros)
        cw = float(benchmark.canvas_width); ch = float(benchmark.canvas_height)
        full_mov = benchmark.get_movable_mask().detach().cpu().numpy().astype(bool)
        if nh > 1:
            # The C spiral legalizer resolves DENSE overlaps the pairwise push
            # cannot (it deadlocks on packed designs, leaving overlaps on
            # ariane/nvdla/ibm). Run it first, then a pairwise push to finish any
            # residual; push-only if the lib is unavailable. Hard macros only
            # (soft may overlap); fixed macros pinned via the movable mask.
            try:
                _ensure_lib("leg")
                _leg = {'nh': nh, 'sizes': sizes[:nh], 'movable': full_mov[:nh],
                        'cw': cw, 'ch': ch,
                        'hw': sizes[:nh, 0] / 2.0, 'hh': sizes[:nh, 1] / 2.0}
                _out_hard, _ = _legalize_two_pass(pos[:nh].copy(), _leg,
                                                  label="emergency-legal")
                pos[:nh] = _out_hard
            except Exception:
                pass
            pos = _clearance_push(pos, sizes, nh, cw, ch, 0.0,
                                  movable=full_mov[:nh], n_iters=1000)
            # Guaranteed-valid finisher: if overlaps persist (the spiral
            # legalizer can stick on dense random inputs - observed on the NG45
            # designs), shelf-pack the MOVABLE hard macros on a uniform grid
            # sized to the largest macro. Overlap-free by construction and in
            # bounds whenever it fits (hard-util ~50% on NG45/IBM, so it does).
            if _hard_overlap_exists(pos, sizes, nh):
                cw_cell = float(sizes[:nh, 0].max()) + 1.0
                ch_cell = float(sizes[:nh, 1].max()) + 1.0
                ncols = max(1, int(cw // cw_cell))
                mov_idx = [i for i in range(nh) if full_mov[i]]
                nrows = (len(mov_idx) + ncols - 1) // ncols
                if nrows * ch_cell <= ch:  # only if the grid fits the canvas
                    for slot, i in enumerate(mov_idx):
                        r, c = divmod(slot, ncols)
                        pos[i, 0] = c * cw_cell + cw_cell * 0.5
                        pos[i, 1] = r * ch_cell + ch_cell * 0.5
        out = torch.tensor(pos, dtype=torch.float32)
        _mov = torch.as_tensor(full_mov, dtype=torch.bool)
        _hw = torch.as_tensor(sizes[:, 0] * 0.5, dtype=torch.float32)
        _hh = torch.as_tensor(sizes[:, 1] * 0.5, dtype=torch.float32)
        _eps = 1e-3
        _nx = torch.minimum(torch.maximum(out[:, 0], _hw + _eps), float(cw) - _hw - _eps)
        _ny = torch.minimum(torch.maximum(out[:, 1], _hh + _eps), float(ch) - _hh - _eps)
        out[:, 0] = torch.where(_mov, _nx, out[:, 0])
        out[:, 1] = torch.where(_mov, _ny, out[:, 1])
        return out
    except Exception:
        # Absolute last resort: input positions unchanged (still better than a
        # raise -> DQ; the evaluator's own legalizer may yet rescue it).
        return benchmark.macro_positions.detach().cpu().to(torch.float32).clone()


def _legalize_two_pass(pos_hard, leg, fine=0.10, coarse=0.25, label="?"):
    """Try `fine` step then fall back to `coarse` on residual overlap; both
    passes copy pos_hard before the C call so the caller's array isn't
    mutated."""
    out, overlap = _legalize_c(
        pos_hard.copy(), leg['nh'], leg['sizes'], leg['movable'],
        leg['cw'], leg['ch'], leg['hw'], leg['hh'], fine, label=label)
    if overlap:
        out, overlap = _legalize_c(
            pos_hard.copy(), leg['nh'], leg['sizes'], leg['movable'],
            leg['cw'], leg['ch'], leg['hw'], leg['hh'], coarse, label=label)
    return out, overlap


# --- Configuration ---


@dataclass
class GPConfig:
    """Xplace global placement parameters."""
    target_density: float = 0.75
    density_weight: float = 8e-5
    num_bins: int = 128
    stop_overflow: float = 0.07
    inner_iter: int = 2000
    lr: float = 0.01
    wa_coeff: float = 4.0
    seed: int = 0
    density_weight_coef: float = 1.05  # Xplace density-weight ramp per iter.
    ignore_net_degree: int = 100


@dataclass
class RefineConfig:
    """Analytical refinement parameters."""
    cong_mult: float = 1.0      # congestion loss multiplier.
    den_mult: float = 2.0       # density loss multiplier.
    rep_margin: float = 0.05    # repulsion margin (grows with iteration).
    lr: float = 0.005           # Adam learning rate.
    gamma_mult: float = 1.0     # WA-HPWL gamma scaling.
    n_iters: int = 200          # optimization iterations.


# GP parameter tiers by design size: (max_macros, GPConfig override).
GP_TIERS = [
    (1500, GPConfig(num_bins=128, inner_iter=2000, ignore_net_degree=100)),
    (2500, GPConfig(num_bins=96,  inner_iter=1500, ignore_net_degree=75)),
]
GP_TIER_DEFAULT = GPConfig(num_bins=32, inner_iter=800, ignore_net_degree=30, stop_overflow=0.10)


# --- Xplace environment setup ---

_XPLACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Xplace")
_XPLACE_BUILD = os.path.join(_XPLACE_DIR, "build")
_ld_paths = [
    os.path.join(torch.__path__[0], "lib"),
    # _ensure_xplace preloads the cpybin libs RTLD_GLOBAL; their SONAMEs match,
    # so DT_NEEDED resolves without LD_LIBRARY_PATH.
    os.path.join(_XPLACE_DIR, "cpp_to_py", "cpybin"),
    # Fallbacks for a source-built Xplace.
    os.path.join(_XPLACE_BUILD, "cpp_to_py", "common"),
    os.path.join(_XPLACE_BUILD, "thirdparty", "flute"),
]
# When running inside a conda/micromamba env, its lib dir may hold the CUDA and
# Boost runtimes Xplace links against.
if os.environ.get("CONDA_PREFIX"):
    _ld_paths.append(os.path.join(os.environ["CONDA_PREFIX"], "lib"))
os.environ["LD_LIBRARY_PATH"] = ":".join(_ld_paths) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["OMP_NUM_THREADS"] = "16"
# Must be set BEFORE the CUDA context inits for deterministic cuBLAS GEMMs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if _XPLACE_DIR not in sys.path:
    sys.path.insert(0, _XPLACE_DIR)

_xplace_loaded = False
_Flute = None
_IOParser = None


def _ensure_xplace():
    global _xplace_loaded, _Flute, _IOParser
    if _xplace_loaded:
        return
    for lib_path in _ld_paths:
        for so in ("libxplace_common.so", "libflute.so"):
            p = os.path.join(lib_path, so)
            if os.path.exists(p):
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
    from src.core.flute import Flute
    from utils.io_parser import IOParser
    _Flute = Flute
    _IOParser = IOParser
    powv = os.path.join(_XPLACE_DIR, "thirdparty/flute/POWV9.dat")
    post = os.path.join(_XPLACE_DIR, "thirdparty/flute/POST9.dat")
    shutil.copy2(powv, "/tmp/POWV9.dat")
    shutil.copy2(post, "/tmp/POST9.dat")
    _Flute.register(8, "/tmp/POWV9.dat", "/tmp/POST9.dat")
    os.makedirs("/tmp/xp_run/run/log", exist_ok=True)
    _xplace_loaded = True


# --- Helpers ---


def _gp_config_for_design(num_macros: int) -> GPConfig:
    """Select GP parameters scaled to design size."""
    for max_macros, cfg in GP_TIERS:
        if num_macros <= max_macros:
            return cfg
    return GP_TIER_DEFAULT


# --- Xplace GP runner ---


def _extract_positions(node_pos, data):
    """Convert Xplace node positions to {name: [llx, lly]} dict."""
    pos_cpu = node_pos.detach().cpu().numpy()
    node_names = data.node_names
    result = {}
    mov_lhs, mov_rhs = data.movable_index
    for i in range(mov_rhs - mov_lhs):
        name = node_names[i] if i < len(node_names) else f"node_{i}"
        cx = float(pos_cpu[i, 0]) * float(data.die_scale[0]) + float(data.die_shift[0])
        cy = float(pos_cpu[i, 1]) * float(data.die_scale[1]) + float(data.die_shift[1])
        w = float(data.node_size[i, 0]) * float(data.die_scale[0])
        h = float(data.node_size[i, 1]) * float(data.die_scale[1])
        result[name] = [float((cx - w / 2) / data.microns),
                        float((cy - h / 2) / data.microns)]
    return result


def _run_one_config(design_info, rawdb, gpdb, cfg: GPConfig, device, logger):
    """Run a single Xplace GP config; returns (positions, hpwl, overflow,
    gp_time, iters) or None on failure."""
    from src.database import PlaceData
    from src.initializer import get_init_density_map
    from src.param_scheduler import ParamScheduler
    from src.run_placement_nesterov import global_placement_main

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Xplace expects a flat args namespace; values below are fixed defaults.
    class A: pass
    args = A()
    args.gpu = device.index or 0
    args.seed = int(os.environ.get("XP_GP_SEED", str(cfg.seed)))
    args.deterministic = True
    args.global_placement = True
    args.lr = cfg.lr
    args.inner_iter = cfg.inner_iter
    args.wa_coeff = cfg.wa_coeff
    args.num_bin_x = cfg.num_bins
    args.num_bin_y = cfg.num_bins
    args.density_weight = cfg.density_weight
    args.density_weight_coef = cfg.density_weight_coef
    args.target_density = float(os.environ.get("XP_GP_TARGET_DENSITY", str(cfg.target_density)))
    args.use_filler = True
    args.noise_ratio = 0.025
    args.ignore_net_degree = cfg.ignore_net_degree
    args.use_eplace_nesterov = True
    args.clamp_node = True
    args.use_precond = True
    args.stop_overflow = cfg.stop_overflow
    args.enable_skip_update = True
    args.enable_sample_force = True
    args.mixed_size = True
    args.use_cell_inflate = False
    args.use_route_force = False
    args.route_freq = 1000
    args.num_route_iter = 400
    args.route_weight = 0
    args.congest_weight = 0
    args.pseudo_weight = 0
    args.timing_opt = False
    args.log_freq = 9999
    args.verbose_cpp_log = False
    args.cpp_log_level = 2
    args.legalization = False
    args.detail_placement = False
    args.draw_placement = False
    args.write_placement = False
    args.write_global_placement = False
    args.result_dir = "/tmp/xp_run"
    args.exp_id = "run"
    args.log_dir = "log"
    args.log_name = "xp.log"
    args.eval_dir = "eval"
    args.output_dir = "output"
    args.output_prefix = "placement"
    args.design_name = "design"
    args.dataset = "custom"
    args.min_area_inc = 0.01
    args.visualize_cgmap = False
    args.timing_freq = 1
    args.calibration = True
    args.calibration_step = 0.1
    args.timing_start_iter = 100
    args.timing_init_weight = 0.05
    args.decay_factor = 0.3
    args.decay_boost = 3
    args.wire_resistance_per_micron = 2.535
    args.wire_capacitance_per_micron = 0.16e-15
    args.dp_engine = "default"
    args.eval_by_external = False
    args.eval_engine = "ntuplace4dr"
    args.final_route_eval = False
    args.include_macros = False
    args.num_threads = 8

    try:
        data = PlaceData(args, logger, **copy.deepcopy(design_info))
        data = data.to(device)
        data = data.preprocess()
        get_init_density_map(rawdb, gpdb, data, args, logger)
        data.init_filler()
        ps = ParamScheduler(data, args, logger)
        node_pos, iteration, gp_hpwl, overflow, gp_time, _ = \
            global_placement_main(gpdb, rawdb, ps, data, args, logger, {})

        positions = _extract_positions(node_pos, data)
        del data, ps, node_pos
        torch.cuda.empty_cache()
        return positions, float(gp_hpwl), float(overflow), float(gp_time), int(iteration)
    except Exception as e:
        print(f"    [GP] config failed: {e}")
        torch.cuda.empty_cache()
        return None


# --- LEF/DEF generation ---

def _generate_lefdef(benchmark, plc, lef_file, def_file):
    """Generate LEF/DEF files for Xplace from benchmark + plc data."""
    db_unit = 2000
    cw, ch = benchmark.canvas_width, benchmark.canvas_height

    macro_pin_info = {}
    pin_to_macro = {}
    for mod in plc.modules_w_pins:
        if mod.get_type() in ("MACRO_PIN", "macro_pin"):
            mn = mod.get_macro_name() if hasattr(mod, "get_macro_name") else None
            if mn:
                ps = mod.get_name().split("/")[-1]
                xo = mod.x_offset if hasattr(mod, "x_offset") else 0
                yo = mod.y_offset if hasattr(mod, "y_offset") else 0
                macro_pin_info.setdefault(mn, []).append((ps, xo, yo))
                pin_to_macro[mod.get_name()] = mn

    min_h = min(benchmark.macro_sizes[:benchmark.num_hard_macros, 1]).item()
    site_w = min_h

    # LEF
    with open(lef_file, "w") as fp:
        fp.write("VERSION 5.8 ;\nBUSBITCHARS \"[]\" ;\nDIVIDERCHAR \"/\" ;\n\n")
        fp.write("UNITS\n  DATABASE MICRONS 2000 ;\nEND UNITS\n\n")
        fp.write(f"SITE CORE_SITE\n  SIZE {site_w} BY {min_h} ;\n  CLASS CORE ;\n"
                 f"  SYMMETRY Y ;\nEND CORE_SITE\n\n")
        pitch = round(min_h / 7.5, 6)
        lw = round(pitch / 2, 6)
        for li in range(1, 7):
            d = "HORIZONTAL" if li % 2 == 1 else "VERTICAL"
            fp.write(f"LAYER M{li}\n  TYPE ROUTING ;\n  DIRECTION {d} ;\n"
                     f"  PITCH {pitch} ;\n  WIDTH {lw} ;\n  SPACING {lw} ;\n"
                     f"END M{li}\n\n")
        for i in range(benchmark.num_macros):
            name = benchmark.macro_names[i]
            w = benchmark.macro_sizes[i, 0].item()
            h = benchmark.macro_sizes[i, 1].item()
            is_hard = i < benchmark.num_hard_macros
            ref = f"REF_{name}" if is_hard else f"SREF_{name}"
            cls = "BLOCK" if is_hard else "CORE"
            fp.write(f"MACRO {ref}\n  CLASS {cls} ;\n  ORIGIN 0 0 ;\n"
                     f"  FOREIGN {ref} 0 0 ;\n  SIZE {w} BY {h} ;\n  SYMMETRY X Y ;\n")
            if not is_hard:
                fp.write("  SITE CORE_SITE ;\n")
            pins = macro_pin_info.get(name, [])
            for pn, xo, yo in pins:
                px, py = xo + w / 2, yo + h / 2
                fp.write(f"  PIN {pn}\n    DIRECTION INPUT ;\n    USE SIGNAL ;\n"
                         f"    PORT\n      LAYER M1 ;\n"
                         f"        RECT {px} {py} {px} {py} ;\n    END\n  END {pn}\n")
            if not pins:
                fp.write(f"  PIN dummy\n    DIRECTION INPUT ;\n    USE SIGNAL ;\n"
                         f"    PORT\n      LAYER M1 ;\n"
                         f"        RECT {w/2} {h/2} {w/2} {h/2} ;\n    END\n  END dummy\n")
            fp.write(f"END {ref}\n\n")
        fp.write("END LIBRARY\n")

    # DEF
    with open(def_file, "w") as fp:
        fp.write("VERSION 5.8 ;\nDIVIDERCHAR \"/\" ;\nBUSBITCHARS \"[]\" ;\n\n")
        fp.write(f"DESIGN {benchmark.name} ;\nUNITS DISTANCE MICRONS {db_unit} ;\n")
        fp.write(f"DIEAREA ( 0 0 ) ( {int(cw * db_unit)} {int(ch * db_unit)} ) ;\n\n")
        ns = int(cw / site_w)
        nr = int(ch / min_h)
        hd = int(min_h * db_unit)
        wd = int(site_w * db_unit)
        for i in range(nr):
            o = "N" if i % 2 == 0 else "FS"
            fp.write(f"ROW ROW_{i} CORE_SITE 0 {i * hd} {o} DO {ns} BY 1 STEP {wd} 0 ;\n")
        fp.write("\n")

        # Tracks, required for Xplace GRDatabase init: HORIZONTAL layers use
        # TRACKS Y, VERTICAL layers use TRACKS X.
        pitch_db = int(pitch * db_unit)
        cw_db = int(cw * db_unit)
        ch_db = int(ch * db_unit)
        ntr_x = max(2, cw_db // max(pitch_db, 1))
        ntr_y = max(2, ch_db // max(pitch_db, 1))
        for li in range(1, 7):
            if li % 2 == 1:
                fp.write(f"TRACKS Y 0 DO {ntr_y} STEP {pitch_db} LAYER M{li} ;\n")
            else:
                fp.write(f"TRACKS X 0 DO {ntr_x} STEP {pitch_db} LAYER M{li} ;\n")
        fp.write("\n")

        # Coarse routing grid; absent => Xplace defaults to 512x512.
        gcell_n = 32
        gcx_step = max(1, cw_db // gcell_n)
        gcy_step = max(1, ch_db // gcell_n)
        fp.write(f"GCELLGRID X 0 DO {gcell_n + 1} STEP {gcx_step} ;\n")
        fp.write(f"GCELLGRID Y 0 DO {gcell_n + 1} STEP {gcy_step} ;\n\n")

        fp.write(f"COMPONENTS {benchmark.num_macros} ;\n")
        for i in range(benchmark.num_macros):
            name = benchmark.macro_names[i]
            is_hard = i < benchmark.num_hard_macros
            ref = f"REF_{name}" if is_hard else f"SREF_{name}"
            w = benchmark.macro_sizes[i, 0].item()
            h = benchmark.macro_sizes[i, 1].item()
            cx = benchmark.macro_positions[i, 0].item()
            cy = benchmark.macro_positions[i, 1].item()
            llx = int((cx - w / 2) * db_unit)
            lly = int((cy - h / 2) * db_unit)
            st = "FIXED" if benchmark.macro_fixed[i].item() else "PLACED"
            fp.write(f"  - {name} {ref} + {st} ( {llx} {lly} ) N ;\n")
        fp.write("END COMPONENTS\n\n")

        # Ports with real positions sourced from plc.
        port_names_list = [plc.modules_w_pins[idx].get_name() for idx in plc.port_indices]
        fp.write(f"PINS {len(port_names_list)} ;\n")
        for i, idx in enumerate(plc.port_indices):
            pmod = plc.modules_w_pins[idx]
            pn = pmod.get_name()
            px = int(pmod.x * db_unit)
            py = int(pmod.y * db_unit)
            fp.write(f"  - {pn} + NET {pn} + DIRECTION INPUT + USE SIGNAL\n")
            fp.write("    + PORT + LAYER M1 ( -100 -100 ) ( 100 100 )\n")
            fp.write(f"    + FIXED ( {px} {py} ) N ;\n")
        fp.write("END PINS\n\n")

        port_set = set()
        for idx in plc.port_indices:
            port_set.add(plc.modules_w_pins[idx].get_name())
        nets = []
        net_id = 0
        for mod in plc.modules_w_pins:
            sinks = getattr(mod, "sink", {})
            if not sinks:
                continue
            sn = mod.get_name()
            st = mod.get_type()
            conns = []
            if st in ("PORT", "port"):
                conns.append(f"( PIN {sn} )")
            elif st in ("MACRO_PIN", "macro_pin"):
                mn = pin_to_macro.get(sn)
                if mn:
                    conns.append(f"( {mn} {sn.split('/')[-1]} )")
            for _, sink_pins in sinks.items():
                for sp in sink_pins:
                    mn = pin_to_macro.get(sp)
                    if mn:
                        conns.append(f"( {mn} {sp.split('/')[-1]} )")
                    elif sp in port_set:
                        conns.append(f"( PIN {sp} )")
            if len(conns) >= 2:
                nn = sn if st in ("PORT", "port") else f"n{net_id}"
                net_id += 1
                nets.append(f"  - {nn} " + " ".join(conns) + " ;")
        fp.write(f"NETS {len(nets)} ;\n")
        for ne in nets:
            fp.write(ne + "\n")
        fp.write("END NETS\n\nEND DESIGN\n")


# --- Main Placer ---


class XplacePlacer:

    def __init__(self):
        pass

    def _load_plc(self, name):
        from macro_place.loader import load_benchmark_from_dir
        from pathlib import Path
        # ICCAD04 (ibmXX) -> Testcases/ICCAD04/<name>/
        # NG45 (<X>_ng45) -> Flows/NanGate45/<X>/netlist/output_CT_Grouping/
        # ASAP7 (<X>_asap7) -> Flows/ASAP7/<X>/netlist/output_CT_Grouping/
        # bp_quad_ng45 ships its protobufs under CodeElements (mirrors
        # scripts/evaluate_with_orfs.py:271).
        candidates = []
        if name.endswith("_ng45"):
            base = name[:-len("_ng45")]
            if base == "bp_quad":
                candidates.append(Path("external/MacroPlacement/CodeElements/SimulatedAnnealingGWTW/test/bp_ng45"))
            candidates.append(Path("external/MacroPlacement/Flows/NanGate45") / base / "netlist/output_CT_Grouping")
        elif name.endswith("_asap7"):
            base = name[:-len("_asap7")]
            candidates.append(Path("external/MacroPlacement/Flows/ASAP7") / base / "netlist/output_CT_Grouping")
        candidates.append(Path("external/MacroPlacement/Testcases/ICCAD04") / name)
        for root in candidates:
            if root.exists():
                _, plc = load_benchmark_from_dir(str(root))
                return plc
        return None

    @staticmethod
    def _parse_lefdef(benchmark, lef_file, def_file):
        """Parse LEF/DEF into Xplace's C++ rawdb/gpdb plus Python design_info;
        returns (parser_obj, rawdb, gpdb, design_info)."""
        parser_obj = _IOParser()
        params = {"lef": os.path.abspath(lef_file),
                  "def": os.path.abspath(def_file),
                  "design_name": benchmark.name, "benchmark": "custom"}
        rawdb, gpdb = parser_obj.read(
            params, verbose_log=False, log_level=2,
            lite_mode=True, random_place=False, num_threads=8)
        design_info = parser_obj.preprocess_design_info(gpdb)
        return parser_obj, rawdb, gpdb, design_info

    def _run_gp(self, benchmark, parsed, device, logger):
        """Run Xplace GP on parsed (parser, rawdb, gpdb, design_info), then
        drop all C++ refs and flush Xplace caches; returns (positions_dict,
        hpwl, overflow, gp_time, iters) or None."""
        parser_obj, rawdb, gpdb, design_info = parsed

        gp_cfg = _gp_config_for_design(benchmark.num_macros)
        print(f"    nh={benchmark.num_hard_macros} macros={benchmark.num_macros} "
              f"bins={gp_cfg.num_bins} iters={gp_cfg.inner_iter} deg={gp_cfg.ignore_net_degree}")

        t1 = time.time()
        result = _run_one_config(design_info, rawdb, gpdb, gp_cfg, device, logger)

        # Free C++ objects + Xplace global GPU caches.
        del design_info, rawdb, gpdb, parser_obj
        self._flush_xplace_caches()
        gc.collect()
        torch.cuda.empty_cache()

        if result is None:
            print(f"    FAILED {time.time()-t1:.1f}s")
        else:
            positions, hpwl, overflow, gp_time, iters = result
            print(f"    hpwl={hpwl:.0f}  ovfl={overflow:.4f}  iters={iters}  {time.time()-t1:.1f}s")
        return result

    @staticmethod
    def _flush_xplace_caches():
        """Clear Xplace's module-level global caches that hold GPU tensors."""
        try:
            from src.core.wa_wirelength_hpwl import hpwl_cache
            hpwl_cache.masked_scale_partial_hpwl = None
        except ImportError:
            pass
        try:
            from src.core.torch_dct import fft_basis_cache, dct_matrix
            fft_basis_cache.dct.clear()
            fft_basis_cache.idct.clear()
            dct_matrix.dctA.clear()
            dct_matrix.idctA.clear()
        except ImportError:
            pass
        try:
            from src.core.route_force import route_cache
            route_cache.reset()
        except (ImportError, AttributeError):
            pass
        try:
            from src.detail_placement import preprocess_db_cache
            preprocess_db_cache.reset()
        except (ImportError, AttributeError):
            pass

    def _map_positions(self, positions, benchmark):
        """Map Xplace GP output {name: [llx, lly]} back to center-based pos_np."""
        name_to_idx = {n: i for i, n in enumerate(benchmark.macro_names)}
        pos_np = benchmark.macro_positions.numpy().astype(np.float64).copy()
        for name, (llx, lly) in positions.items():
            if name in name_to_idx:
                idx = name_to_idx[name]
                w = benchmark.macro_sizes[idx, 0].item()
                h = benchmark.macro_sizes[idx, 1].item()
                pos_np[idx, 0] = llx + w / 2
                pos_np[idx, 1] = lly + h / 2
        return pos_np

    def _refine_and_legalize(self, pos_np, octx, cfg: RefineConfig, label, leg):
        """Analytical refinement followed by spiral legalization (`leg` carries
        the topology/size dict needed by _legalize_two_pass); returns
        refined+legalized positions or None."""
        t1 = time.time()
        try:
            refined = self._analytical_refine(pos_np, octx, cfg)
        except Exception as e:
            print(f"    [{label}] refine failed: {e}")
            return None
        nh = leg['nh']
        # The tighter step stays closer to target and so gives better WL; fall
        # back to the coarser one only if overlap remains.
        out, overlap = _legalize_two_pass(refined[:nh], leg, coarse=0.20, label=label)
        if overlap:
            print(f"    [{label}] overlap after legalization (both passes)")
            return None
        refined[:nh] = out
        print(f"    [{label}] {time.time()-t1:.1f}s")
        return refined

    @staticmethod
    def _micro_move(pos_np, nh, sizes, cw, ch, dx, dy, cs):
        """Greedy per-hard-macro 4-neighbor search via CongState (cs must
        already be in sync with pos_np); small dx/dy keeps moves inside the
        local basin so polish reconciles drift cleanly."""
        t0 = time.time()
        deltas = [(dx, 0), (-dx, 0), (0, dy), (0, -dy)]
        pos = pos_np.copy()
        base, _, _, _ = cs.score()
        cur = base
        order = sorted(range(nh), key=lambda m: -(sizes[m, 0] * sizes[m, 1]))
        tested = 0
        accepted = 0
        for m in order:
            mi = int(m)
            ox, oy = pos[m, 0], pos[m, 1]
            hw, hh = sizes[m, 0] * 0.5, sizes[m, 1] * 0.5
            best_d = None
            best_p = cur
            for ddx, ddy in deltas:
                nx = ox + ddx
                ny = oy + ddy
                if nx - hw < 0 or nx + hw > cw:
                    continue
                if ny - hh < 0 or ny + hh > ch:
                    continue
                p, _, _, _ = cs.apply([mi], [nx], [ny])
                tested += 1
                if p < best_p - 1e-7:
                    best_p = p
                    best_d = (ddx, ddy)
                cs.revert()
            if best_d is not None:
                new_x = ox + best_d[0]
                new_y = oy + best_d[1]
                cs.apply([mi], [new_x], [new_y])
                cs.commit()
                pos[m, 0] = new_x
                pos[m, 1] = new_y
                cur = best_p
                accepted += 1
        print(f"    step=({dx:.1f},{dy:.1f}):  "
              f"{accepted}/{tested} accepted  fast {base:.4f}->{cur:.4f}  "
              f"{time.time()-t0:.2f}s")
        return pos

    @staticmethod
    def _klein4_orient(cur, pn, sizes, nh):
        """Per hard macro, pick the Klein-4 orientation {N, FN, FS, S} minimizing
        total bbox-HPWL on its incident nets. Returns an int8[nh] code array
        (0=N, 1=FN, 2=FS, 3=S) for the Tier-2 orientations.pt sidecar.

        NON-DESTRUCTIVE: pin offsets are restored after each trial, so positions
        and the internal proxy are unchanged. Tier-1 recomputes pins from the
        original orientation, so flips only matter at Tier-2 - recorded, never
        re-polished around (that would perturb Tier-1 for no gain).

        Klein-4 only: the 90 deg rotations swap W x H and are disallowed
        (fakeram45 SRAMs aren't rotatable). All four preserve the footprint.
        Pin-offset transforms for a macro of size W x H, origin lower-left:
          N  (R0)  : (ox, oy)
          FN (MY)  : (W - ox, oy)        mirror about vertical axis
          FS (MX)  : (ox, H - oy)        mirror about horizontal axis
          S  (R180): (W - ox, H - oy)
        """
        pin_macro = pn['pin_macro']
        pin_x = pn['pin_x']
        pin_y = pn['pin_y']
        net_driver = pn['net_driver']
        net_sinks_off = pn['net_sinks_off']
        net_sinks_idx = pn['net_sinks_idx']
        nn = net_driver.shape[0]

        pin_to_nets = [[] for _ in range(pin_macro.shape[0])]
        for k in range(nn):
            d_pin = int(net_driver[k])
            if 0 <= d_pin < len(pin_to_nets):
                pin_to_nets[d_pin].append(k)
            s_lo = int(net_sinks_off[k]); s_hi = int(net_sinks_off[k + 1])
            for p in net_sinks_idx[s_lo:s_hi]:
                pi = int(p)
                if 0 <= pi < len(pin_to_nets):
                    pin_to_nets[pi].append(k)

        def _net_hpwl(k_net):
            d_pin = int(net_driver[k_net])
            if d_pin < 0:
                return 0.0
            s_lo = int(net_sinks_off[k_net]); s_hi = int(net_sinks_off[k_net + 1])

            def _abs(pi):
                m = int(pin_macro[pi])
                if m < 0:
                    return (float(pin_x[pi]), float(pin_y[pi]))
                return (float(cur[m, 0]) + float(pin_x[pi]),
                        float(cur[m, 1]) + float(pin_y[pi]))
            pts = [_abs(d_pin)] + [_abs(int(p)) for p in net_sinks_idx[s_lo:s_hi]]
            if len(pts) < 2:
                return 0.0
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            return (max(xs) - min(xs)) + (max(ys) - min(ys))

        codes = np.zeros(nh, dtype=np.int8)
        for m in range(nh):
            macro_pins = np.where(pin_macro == m)[0]
            if len(macro_pins) == 0:
                continue
            affected = set()
            for pi in macro_pins:
                affected.update(pin_to_nets[int(pi)])
            if not affected:
                continue
            affected = list(affected)
            W = float(sizes[m, 0]); H = float(sizes[m, 1])
            ox = pin_x[macro_pins].copy()
            oy = pin_y[macro_pins].copy()

            best_wl = sum(_net_hpwl(k) for k in affected)  # N
            best_code = 0
            for code, (nx, ny) in (
                    (1, (W - ox, oy)),
                    (2, (ox, H - oy)),
                    (3, (W - ox, H - oy))):
                pin_x[macro_pins] = nx
                pin_y[macro_pins] = ny
                wl = sum(_net_hpwl(k) for k in affected)
                if wl < best_wl - 1e-6:
                    best_wl = wl
                    best_code = code
            pin_x[macro_pins] = ox  # restore - non-destructive
            pin_y[macro_pins] = oy
            codes[m] = best_code
        return codes

    def _window_reorder(self, cur, ctx, cong_c_stats, run_cong_c_fn,
                        top_W=None, K=4, budget_s=2.0):
        """Window-reorder soft macros: for each of top_W hot soft macros
        enumerate the K!-1 non-identity permutations of K nearest soft
        neighbors, accept best by exact CongState proxy; runs a brief
        n_iter=2 polish after to lock cumulative gains."""
        n_macros, nh, sizes, mov_i32 = (ctx.n_macros, ctx.nh, ctx.sizes,
                                        ctx.mov_i32)
        cw, ch, octx, pn = ctx.cw, ctx.ch, ctx.octx, ctx.pn
        wl_extra_weight = ctx.wl_extra_weight
        from itertools import permutations
        if top_W is None:
            top_W = 30
        if n_macros - nh <= K:
            return cur

        # Pin count is the heat heuristic for picking the hot soft macros.
        soft_idx = octx['soft_movable']
        if soft_idx.size < K:
            return cur
        heats = pn['soft_pin_counts_full'][soft_idx]
        order = np.argsort(-heats)
        top_soft = soft_idx[order[:top_W]]
        soft_pos = cur[soft_idx]

        identity = tuple(range(K))
        non_id_perms = [p for p in permutations(range(K)) if p != identity]

        accepted = 0
        evaluated = 0
        with CongState(cur, sizes, mov_i32, n_macros, nh, cw, ch,
                       octx, pn, wl_extra_weight=wl_extra_weight) as cs:
            for hot_m in top_soft:
                # K nearest soft macros by Euclidean distance (includes self).
                my = cur[hot_m]
                d2 = ((soft_pos[:, 0] - my[0]) ** 2
                      + (soft_pos[:, 1] - my[1]) ** 2)
                nearest_in_soft = np.argpartition(d2, K - 1)[:K]
                window_macros = soft_idx[nearest_in_soft].tolist()
                positions = [(float(cur[m, 0]), float(cur[m, 1]))
                             for m in window_macros]

                base_proxy, _, _, _ = cs.score()
                best_proxy = base_proxy
                best_perm = None
                best_breakdown = None
                for perm in non_id_perms:
                    new_x = [positions[perm[i]][0] for i in range(K)]
                    new_y = [positions[perm[i]][1] for i in range(K)]
                    p, wl, den, cong = cs.apply(window_macros, new_x, new_y)
                    evaluated += 1
                    if p < best_proxy - 1e-7:
                        best_proxy = p
                        best_perm = perm
                        best_breakdown = (wl, den, cong)
                    cs.revert()

                if best_perm is not None:
                    new_x = [positions[best_perm[i]][0] for i in range(K)]
                    new_y = [positions[best_perm[i]][1] for i in range(K)]
                    cs.apply(window_macros, new_x, new_y)
                    cs.commit()
                    for i, m in enumerate(window_macros):
                        cur[m, 0] = new_x[i]
                        cur[m, 1] = new_y[i]
                    accepted += 1
                    cong_c_stats['proxy'] = best_proxy
                    cong_c_stats['wl'] = best_breakdown[0]
                    cong_c_stats['den'] = best_breakdown[1]
                    cong_c_stats['cong'] = best_breakdown[2]

        print(f"    {evaluated} perms evaluated, {accepted} windows accepted, "
              f"proxy={cong_c_stats.get('proxy', 0):.4f}")
        # Reconcile after accepts: shake-cc/fine-cc schedule absorbs CongState
        # drift.
        if accepted > 0:
            cur = run_cong_c_fn(cur, n_iter=3, lr_bins=0.5, label="wr-cc",
                                extra_phases=_CC_RECONCILE_PHASES)
        return cur

    def _long_shake(self, cur, ctx, cong_c_stats, run_cong_c_fn, cc_call_fn,
                    leg_dict,
                    top_K_hard=60, top_K_soft=30, budget_s=4.0):
        """Multi-scale post-polish relocation: probe 8 dirs x 5 scales for top
        hot hard+soft macros, gate via CongState incremental proxy with a
        fresh cc_call(n_iter=0) verify before paying the reconcile polish
        (CongState drifts after many commits and density formulas differ by
        ~0.005/macro)."""
        n_macros, nh, sizes, mov_i32 = (ctx.n_macros, ctx.nh, ctx.sizes,
                                        ctx.mov_i32)
        cw, ch, octx, pn = ctx.cw, ctx.ch, ctx.octx, ctx.pn
        wl_extra_weight = ctx.wl_extra_weight
        t0 = time.time()
        gw = octx['gw']
        gh = octx['gh']
        dirs = np.array([
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, 1), (1, -1), (-1, -1),
        ], dtype=np.float64)
        scales = (0.125, 0.25, 0.5, 1.0, 2.0)

        hard_pin_counts = pn['hard_pin_counts']
        soft_pin_counts = pn['soft_pin_counts_full'][nh:]
        hard_movable = octx['hard_movable']
        soft_movable = octx['soft_movable']
        if hard_movable.size:
            top_hard = hard_movable[
                np.argsort(-hard_pin_counts[hard_movable])][:top_K_hard]
        else:
            top_hard = np.array([], dtype=np.int64)
        if soft_movable.size:
            top_soft = soft_movable[
                np.argsort(-soft_pin_counts[soft_movable - nh])][:top_K_soft]
        else:
            top_soft = np.array([], dtype=np.int64)

        pre_pos = cur.copy()
        pre_proxy = cong_c_stats.get('proxy')

        evaluated = 0
        accepted = 0
        with CongState(cur, sizes, mov_i32, n_macros, nh, cw, ch,
                       octx, pn, wl_extra_weight=wl_extra_weight) as cs:
            base_proxy, _, _, _ = cs.score()
            cur_proxy = base_proxy
            order = np.concatenate([top_hard, top_soft])
            for m in order:
                mi = int(m)
                ox = float(cur[m, 0])
                oy = float(cur[m, 1])
                hw = float(sizes[m, 0]) * 0.5
                hh = float(sizes[m, 1]) * 0.5
                best_delta = 0.0
                best_dx = 0.0
                best_dy = 0.0
                best_proxy_local = None
                for s in scales:
                    dxs = gw * s
                    dys = gh * s
                    for ux, uy in dirs:
                        nx = ox + ux * dxs
                        ny = oy + uy * dys
                        if nx - hw < 0 or nx + hw > cw:
                            continue
                        if ny - hh < 0 or ny + hh > ch:
                            continue
                        proxy, _, _, _ = cs.apply([mi], [nx], [ny])
                        evaluated += 1
                        delta = proxy - cur_proxy
                        if delta < best_delta - 1e-9:
                            best_delta = delta
                            best_dx = ux * dxs
                            best_dy = uy * dys
                            best_proxy_local = proxy
                        cs.revert()
                if best_proxy_local is not None:
                    nx = ox + best_dx
                    ny = oy + best_dy
                    cs.apply([mi], [nx], [ny])
                    cs.commit()
                    cur[m, 0] = nx
                    cur[m, 1] = ny
                    cur_proxy = best_proxy_local
                    accepted += 1
            final_reported = cur_proxy

        print(f"    probes={evaluated} accept={accepted} reported "
              f"{base_proxy:.4f}->{final_reported:.4f} "
              f"({time.time()-t0:.2f}s)")

        if accepted == 0:
            return cur

        # Cheap fresh-rebuild gate: if cc_call n_iter=0 proxy doesn't beat pre,
        # skip polish.
        (_, _, _, p_v0, _, _, _) = cc_call_fn(
            cur.copy(), n_iter=0, lr_bins=2.0)
        if pre_proxy is not None and p_v0 >= pre_proxy - 1e-6:
            print(f"    rebuild-verify {p_v0:.4f} >= pre {pre_proxy:.4f}; "
                  f"revert (skip polish)")
            cur[:] = pre_pos[:]
            return cur

        # Shake-cc/fine-cc reconcile so polish-final ~ fresh-init before
        # window-reorder.
        cur = run_cong_c_fn(cur, n_iter=3, lr_bins=0.5, label="long-cc",
                            extra_phases=_CC_RECONCILE_PHASES)
        post_proxy = cong_c_stats['proxy']

        legalized, overlap = _legalize_two_pass(cur[:nh], leg_dict, label="long_shake")
        if not overlap:
            cur[:nh] = legalized

        # No fresh-rebuild gate here - c_probe gate upstream is the guard.
        print(f"    post-polish {post_proxy:.4f} (vs pre {pre_proxy:.4f})")
        return cur

    def _hardmove_pass(self, cur, ctx, cong_c_stats, run_cong_c_fn,
                       cc_call_fn, leg_dict,
                       step_rel=0.25, budget_s=4.0, n_passes=3):
        """Hardmove: hottest-first scan of all hard macros, 16 candidates each
        (axis/diag at step + 8 knight-like at 2*step), CongState-incremental
        scored with AABB-other guard; multi-pass with cs.rebuild() between
        passes to flush drift; followed by legalize + n_iter=2 reconcile and
        fresh-rebuild verify."""
        n_macros, nh, sizes, mov_i32 = (ctx.n_macros, ctx.nh, ctx.sizes,
                                        ctx.mov_i32)
        cw, ch, octx, pn = ctx.cw, ctx.ch, ctx.octx, ctx.pn
        wl_extra_weight = ctx.wl_extra_weight
        t0 = time.time()
        gw = octx['gw']
        gh = octx['gh']
        dx = gw * step_rel
        dy = gh * step_rel
        # 16 moves: axis+diag at step, plus 8 knight-like at (2*step,
        # step)/(step, 2*step).
        moves = np.array([
            (dx, 0), (-dx, 0), (0, dy), (0, -dy),
            (dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy),
            (2 * dx, dy), (-2 * dx, dy), (2 * dx, -dy), (-2 * dx, -dy),
            (dx, 2 * dy), (-dx, 2 * dy), (dx, -2 * dy), (-dx, -2 * dy),
        ], dtype=np.float64)

        pin_counts = pn['hard_pin_counts']
        hard_movable = octx['hard_movable']
        if hard_movable.size == 0:
            return cur
        # Hottest-first: earlier accepts shift the basin so high-pin-count
        # macros go first.
        order = hard_movable[np.argsort(-pin_counts[hard_movable])]

        pre_pos = cur.copy()
        pre_proxy = cong_c_stats.get('proxy')

        # The AABB-other guard keeps accepts from crowding into regions the
        # legalizer would then tear apart. Its (lo, hi) cache refreshes on
        # every accept.
        hard_x = cur[:nh, 0].copy()
        hard_y = cur[:nh, 1].copy()
        hard_hw = octx['hard_hw']
        hard_hh = octx['hard_hh']
        hard_lo_x = hard_x - hard_hw
        hard_hi_x = hard_x + hard_hw
        hard_lo_y = hard_y - hard_hh
        hard_hi_y = hard_y + hard_hh

        evaluated = 0
        rejected_overlap = 0
        accepted = 0
        passes_done = 0
        moves_dx = moves[:, 0]
        moves_dy = moves[:, 1]
        with CongState(cur, sizes, mov_i32, n_macros, nh, cw, ch,
                       octx, pn, wl_extra_weight=wl_extra_weight) as cs:
            base_proxy, _, _, _ = cs.score()
            cur_proxy = base_proxy
            # Multi-pass with empty-pass termination.
            for pass_i in range(n_passes):
                pass_accepts = 0
                for m in order:
                    mi = int(m)
                    ox = float(cur[m, 0])
                    oy = float(cur[m, 1])
                    hw = float(hard_hw[m])
                    hh = float(hard_hh[m])
                    # Filter all 16 candidates against the canvas and hard
                    # overlaps in one vectorized pass - ~16x fewer numpy
                    # launches than one candidate at a time.
                    nx_arr = ox + moves_dx
                    ny_arr = oy + moves_dy
                    canvas_ok = ((nx_arr - hw >= 0) & (nx_arr + hw <= cw)
                                 & (ny_arr - hh >= 0) & (ny_arr + hh <= ch))
                    coll = ((nx_arr[:, None] + hw > hard_lo_x[None, :])
                            & (nx_arr[:, None] - hw < hard_hi_x[None, :])
                            & (ny_arr[:, None] + hh > hard_lo_y[None, :])
                            & (ny_arr[:, None] - hh < hard_hi_y[None, :]))
                    coll[:, mi] = False
                    overlap_any = coll.any(axis=1)
                    rejected_overlap += int((canvas_ok & overlap_any).sum())
                    valid = canvas_ok & ~overlap_any
                    best_delta = 0.0
                    best_dx = 0.0
                    best_dy = 0.0
                    best_proxy_local = None
                    for ci in np.flatnonzero(valid):
                        ddx = float(moves_dx[ci])
                        ddy = float(moves_dy[ci])
                        nx = float(nx_arr[ci])
                        ny = float(ny_arr[ci])
                        proxy, _, _, _ = cs.apply([mi], [nx], [ny])
                        evaluated += 1
                        delta = proxy - cur_proxy
                        if delta < best_delta - 1e-9:
                            best_delta = delta
                            best_dx = ddx
                            best_dy = ddy
                            best_proxy_local = proxy
                        cs.revert()
                    if best_proxy_local is not None:
                        new_x = ox + best_dx
                        new_y = oy + best_dy
                        cs.apply([mi], [new_x], [new_y])
                        cs.commit()
                        cur[m, 0] = new_x
                        cur[m, 1] = new_y
                        hard_x[m] = new_x
                        hard_y[m] = new_y
                        hard_lo_x[m] = new_x - hw
                        hard_hi_x[m] = new_x + hw
                        hard_lo_y[m] = new_y - hh
                        hard_hi_y[m] = new_y + hh
                        cur_proxy = best_proxy_local
                        accepted += 1
                        pass_accepts += 1
                passes_done = pass_i + 1
                if pass_accepts == 0:
                    break
                # Flush CongState incremental-score drift before next pass.
                cs.rebuild()
            final_reported = cur_proxy

        print(f"    passes={passes_done} probes={evaluated} "
              f"reject_overlap={rejected_overlap} "
              f"accept={accepted}/{len(order)} "
              f"reported {base_proxy:.4f}->{final_reported:.4f} "
              f"({time.time()-t0:.2f}s)")

        if accepted == 0:
            return cur

        # Fresh-rebuild gate, as in long-shake: the CongState incremental score
        # drifts after many commits.
        (_, _, _, p_v0, _, _, _) = cc_call_fn(
            cur.copy(), n_iter=0, lr_bins=2.0)
        if pre_proxy is not None and p_v0 >= pre_proxy - 1e-6:
            print(f"    rebuild-verify {p_v0:.4f} >= pre {pre_proxy:.4f}; "
                  f"revert (skip polish)")
            cur[:] = pre_pos[:]
            return cur

        # Legalize, then an n_iter=2 reconcile polish: the first clears
        # overlaps the relocation introduced, the second settles the
        # density-formula mismatch between cong_relax_v2 and the evaluator.
        legalized, overlap = _legalize_two_pass(cur[:nh], leg_dict, label="hardmove")
        if not overlap:
            cur[:nh] = legalized

        cur = run_cong_c_fn(cur, n_iter=2, lr_bins=0.15, label="hm-cc",
                            extra_phases=None)
        post_proxy = cong_c_stats['proxy']
        print(f"    post-polish {post_proxy:.4f} (vs pre {pre_proxy:.4f})")
        return cur

    def _softmove_pass(self, cur, ctx, cong_c_stats, run_cong_c_fn,
                       cc_call_fn,
                       step_rel=0.25, budget_s=3.0, n_passes=2,
                       gpu_state=None):
        """Soft-macro translation pass - _hardmove_pass analog over [nh,
        n_macros); only hard-soft AABB overlap is checked (soft-soft handled
        by the proxy density term); no legalize step (legalize is
        hard-only)."""
        n_macros, nh, sizes, mov_i32 = (ctx.n_macros, ctx.nh, ctx.sizes,
                                        ctx.mov_i32)
        cw, ch, octx, pn = ctx.cw, ctx.ch, ctx.octx, ctx.pn
        wl_extra_weight = ctx.wl_extra_weight
        t0 = time.time()
        if n_macros - nh <= 0:
            return cur
        gw = octx['gw']
        gh = octx['gh']
        dx = gw * step_rel
        dy = gh * step_rel
        moves = np.array([
            (dx, 0), (-dx, 0), (0, dy), (0, -dy),
            (dx, dy), (-dx, dy), (dx, -dy), (-dx, -dy),
            (2 * dx, dy), (-2 * dx, dy), (2 * dx, -dy), (-2 * dx, -dy),
            (dx, 2 * dy), (-dx, 2 * dy), (dx, -2 * dy), (-dx, -2 * dy),
        ], dtype=np.float64)

        soft_pin_counts = pn['soft_pin_counts_full'][nh:]
        soft_movable = octx['soft_movable']
        if soft_movable.size == 0:
            return cur
        order = soft_movable[np.argsort(-soft_pin_counts[soft_movable - nh])]

        # Pre-compute hard (lo, hi) once; hard positions don't move during a
        # soft pass.
        hard_x = cur[:nh, 0]
        hard_y = cur[:nh, 1]
        hard_hw = octx['hard_hw']
        hard_hh = octx['hard_hh']
        hard_lo_x = hard_x - hard_hw
        hard_hi_x = hard_x + hard_hw
        hard_lo_y = hard_y - hard_hh
        hard_hi_y = hard_y + hard_hh

        def _overlaps_hard(nx, ny, hw, hh):
            return bool(((nx + hw > hard_lo_x) & (nx - hw < hard_hi_x)
                         & (ny + hh > hard_lo_y) & (ny - hh < hard_hi_y)).any())

        pre_pos = cur.copy()
        pre_proxy = cong_c_stats.get('proxy')

        evaluated = 0
        rejected_overlap = 0
        accepted = 0
        passes_done = 0
        with CongState(cur, sizes, mov_i32, n_macros, nh, cw, ch,
                       octx, pn, wl_extra_weight=wl_extra_weight) as cs:
            base_proxy, _, _, _ = cs.score()
            cur_proxy = base_proxy
            if gpu_state is not None:
                from abuplace.GPU.softmove_gpu_batched import (
                    gpu_softmove_pass)
                cur, _stats = gpu_softmove_pass(
                    gpu_state, cs, cur, sizes, mov_i32, n_macros, nh,
                    cw, ch, octx,
                    step_rel=step_rel, n_passes=n_passes, budget_s=budget_s)
                evaluated = _stats["probes"]
                accepted = _stats["accepts"]
                passes_done = _stats["passes"]
                final_reported = _stats["final_proxy"]
            else:
                for pass_i in range(n_passes):
                    pass_accepts = 0
                    for m in order:
                        mi = int(m)
                        ox = float(cur[m, 0])
                        oy = float(cur[m, 1])
                        hw = float(sizes[m, 0]) * 0.5
                        hh = float(sizes[m, 1]) * 0.5
                        best_delta = 0.0
                        best_dx = 0.0
                        best_dy = 0.0
                        best_proxy_local = None
                        for ddx, ddy in moves:
                            nx = ox + ddx
                            ny = oy + ddy
                            if nx - hw < 0 or nx + hw > cw:
                                continue
                            if ny - hh < 0 or ny + hh > ch:
                                continue
                            if _overlaps_hard(nx, ny, hw, hh):
                                rejected_overlap += 1
                                continue
                            proxy, _, _, _ = cs.apply([mi], [nx], [ny])
                            evaluated += 1
                            delta = proxy - cur_proxy
                            if delta < best_delta - 1e-9:
                                best_delta = delta
                                best_dx = ddx
                                best_dy = ddy
                                best_proxy_local = proxy
                            cs.revert()
                        if best_proxy_local is not None:
                            cs.apply([mi], [ox + best_dx], [oy + best_dy])
                            cs.commit()
                            cur[m, 0] = ox + best_dx
                            cur[m, 1] = oy + best_dy
                            cur_proxy = best_proxy_local
                            accepted += 1
                            pass_accepts += 1
                    passes_done = pass_i + 1
                    if pass_accepts == 0:
                        break
                    cs.rebuild()
                final_reported = cur_proxy

        path_tag = "gpu" if gpu_state is not None else "cpu"
        print(f"    [{path_tag}] passes={passes_done} probes={evaluated} "
              f"reject_overlap={rejected_overlap} "
              f"accept={accepted}/{len(order)} "
              f"reported {base_proxy:.4f}->{final_reported:.4f} "
              f"({time.time()-t0:.2f}s)")

        if accepted == 0:
            return cur

        # Fresh-rebuild gate (CongState drift accumulates after many commits).
        (_, _, _, p_v0, _, _, _) = cc_call_fn(
            cur.copy(), n_iter=0, lr_bins=2.0)
        if pre_proxy is not None and p_v0 >= pre_proxy - 1e-6:
            print(f"    rebuild-verify {p_v0:.4f} >= pre {pre_proxy:.4f}; "
                  f"revert (skip polish)")
            cur[:] = pre_pos[:]
            return cur

        # Reconcile polish only (no legalize - soft macros aren't row-legalized).
        cur = run_cong_c_fn(cur, n_iter=2, lr_bins=0.15, label="sm-cc",
                            extra_phases=None)
        post_proxy = cong_c_stats['proxy']
        print(f"    post-polish {post_proxy:.4f} (vs pre {pre_proxy:.4f})")
        return cur

    @staticmethod
    def _extract_pin_net_data(benchmark, plc):
        """Build flat pin/net CSR arrays for cong_relax_v2:
        pin_macro/pin_x/pin_y (port pin offsets are absolute), net_driver,
        net_sinks_off+idx (CSR), net_weight, plus
        hrpm/vrpm/h_alloc/v_alloc/smooth_range proxy knobs."""
        n_total = benchmark.num_macros
        nh = benchmark.num_hard_macros
        name_to_bidx = {}
        for bi, pi in enumerate(plc.hard_macro_indices):
            if bi < nh:
                name_to_bidx[plc.modules_w_pins[pi].get_name()] = bi
        for bi, pi in enumerate(plc.soft_macro_indices):
            v = nh + bi
            if v < n_total:
                name_to_bidx[plc.modules_w_pins[pi].get_name()] = v

        # mod_idx -> compact pin idx for MACRO_PIN/PORT entries.
        mod_to_pin = [-1] * len(plc.modules_w_pins)
        pin_macro = []
        pin_x = []
        pin_y = []
        for mi, mod in enumerate(plc.modules_w_pins):
            t = mod.get_type()
            if t == 'PORT':
                px, py = mod.get_pos()
                mod_to_pin[mi] = len(pin_macro)
                pin_macro.append(-1)
                pin_x.append(float(px))
                pin_y.append(float(py))
            elif t == 'MACRO_PIN':
                ref_idx = plc.get_ref_node_id(mi)
                if ref_idx < 0:
                    continue
                ref_name = plc.modules_w_pins[ref_idx].get_name()
                bidx = name_to_bidx.get(ref_name)
                if bidx is None or bidx >= n_total or bidx < 0:
                    continue
                ox, oy = mod.get_offset()
                mod_to_pin[mi] = len(pin_macro)
                pin_macro.append(int(bidx))
                pin_x.append(float(ox))
                pin_y.append(float(oy))

        # Build nets from plc.nets: dict driver_pin_name -> list[sink_pin_name].
        net_driver = []
        net_sinks_idx = []
        net_sinks_off = [0]
        net_weight = []
        for driver_name, sinks in plc.nets.items():
            d_mi = plc.mod_name_to_indices.get(driver_name)
            if d_mi is None:
                continue
            d_pin = mod_to_pin[d_mi]
            if d_pin < 0:
                continue
            valid_sinks = []
            for s_name in sinks:
                s_mi = plc.mod_name_to_indices.get(s_name)
                if s_mi is None:
                    continue
                s_pin = mod_to_pin[s_mi]
                if s_pin < 0:
                    continue
                valid_sinks.append(s_pin)
            if not valid_sinks:
                continue
            try:
                w = float(plc.modules_w_pins[d_mi].get_weight())
                if w < 1.0:
                    w = 1.0
            except Exception:
                w = 1.0
            net_driver.append(d_pin)
            net_sinks_idx.extend(valid_sinks)
            net_sinks_off.append(len(net_sinks_idx))
            net_weight.append(w)

        # net_cnt for HPWL normalization matches plc.net_cnt (sum of net weights).
        nw_arr = np.asarray(net_weight, dtype=np.float64)
        try:
            net_cnt_norm = float(plc.net_cnt)
            if net_cnt_norm <= 0.0:
                net_cnt_norm = float(nw_arr.sum()) if nw_arr.size else 1.0
        except Exception:
            net_cnt_norm = float(nw_arr.sum()) if nw_arr.size else 1.0
        return {
            'pin_macro': np.asarray(pin_macro, dtype=np.int32),
            'pin_x': np.asarray(pin_x, dtype=np.float64),
            'pin_y': np.asarray(pin_y, dtype=np.float64),
            'net_driver': np.asarray(net_driver, dtype=np.int32),
            'net_sinks_off': np.asarray(net_sinks_off, dtype=np.int32),
            'net_sinks_idx': np.asarray(net_sinks_idx, dtype=np.int32),
            'net_weight': nw_arr,
            'net_cnt_for_norm': net_cnt_norm,
            'hrpm': float(plc.hroutes_per_micron),
            'vrpm': float(plc.vroutes_per_micron),
            'h_alloc': float(plc.hrouting_alloc),
            'v_alloc': float(plc.vrouting_alloc),
            'smooth_range': int(plc.smooth_range),
        }

    def _build_opt_context(self, benchmark, pin_csr):
        n = benchmark.num_macros
        nh = benchmark.num_hard_macros
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        sizes = benchmark.macro_sizes.numpy().astype(np.float64)
        movable = benchmark.get_movable_mask().numpy()
        gr = benchmark.grid_rows
        gc = benchmark.grid_cols
        # Pre-compute the movable index subsets so 4d/4c/4e/4f skip the
        # per-stage rebuild.
        mov_i32 = movable.astype(np.int32)
        all_idx = np.arange(n, dtype=np.int64)
        hard_movable = all_idx[:nh][mov_i32[:nh].astype(bool)]
        soft_movable = all_idx[nh:][mov_i32[nh:].astype(bool)]
        # Static AABB half-extents - basin_jump_v3._aabb_cache and the
        # per-pass hw/hh lookups in operator passes can use these directly.
        hard_hw = (sizes[:nh, 0] / 2.0).astype(np.float64)
        hard_hh = (sizes[:nh, 1] / 2.0).astype(np.float64)
        return {
            'n': n, 'nh': nh, 'cw': cw, 'ch': ch,
            'gr': gr, 'gc': gc,
            'gw': cw / gc, 'gh': ch / gr,
            'sr': 2,
            'sizes_np': sizes, 'movable_np': movable,
            'hard_movable': hard_movable, 'soft_movable': soft_movable,
            'hard_hw': hard_hw, 'hard_hh': hard_hh,
            'pin_csr': pin_csr,
        }

    def _analytical_refine(self, pos_np, octx, cfg: RefineConfig):
        """Analytical refinement: C Adam on WL + density + congestion +
        repulsion."""
        pin_csr = octx['pin_csr']
        return _refine_adam_c(
            pos_np,
            octx['n'], octx['nh'],
            octx['sizes_np'], octx['movable_np'],
            octx['cw'], octx['ch'],
            octx['gr'], octx['gc'],
            octx['sr'],
            pin_csr['pin_macro'], pin_csr['pin_x'], pin_csr['pin_y'],
            pin_csr['net_pin_off'], pin_csr['net_pin_idx'],
            pin_csr['net_weight'],
            cfg.lr, cfg.n_iters,
            cfg.cong_mult, cfg.den_mult, cfg.rep_margin, cfg.gamma_mult,
        )

    def _place_best_of_n(self, benchmark: Benchmark) -> torch.Tensor:
        """Run the full pipeline once per congestion-loss trajectory and return
        the lowest-proxy LEGAL result.

        The RUDY congestion loss improves the GP output, but the polish stage -
        co-tuned to the default-GP basin - can INVERT that gain, turning a
        post-GP win into a final regression at stage 2. Which designs invert is
        unpredictable, so every setting gets a full run. Non-regressing by
        construction: 'off' runs first and is always a candidate. Costs ~3x wall.
        """
        self._in_bo2 = True
        # Trajectory list (comma-sep): "off" = RUDY off, any other token =
        # RUDY on with that kernel. The default three cover the scalar-rudy
        # wins, the rudy_hv wins, and the designs where RUDY inverts.
        traj = [t.strip() for t in os.environ.get(
            "XP_RUDY_BO_KERNELS", "off,rudy,rudy_hv").split(",") if t.strip()]
        # Wall guard so 3x can't blow the 1-hr/benchmark limit: 'off' (first)
        # always runs; a later trajectory is skipped if it likely won't fit.
        budget_s = float(os.environ.get("XP_RUDY_BO_BUDGET_S", "2500"))
        _t_bo = time.time()
        _max_traj = 0.0
        _saved = (os.environ.get("XP_CONG_LOSS"),
                  os.environ.get("XP_CONG_LOSS_KERNEL"))
        best_pos, best_p, best_tag = None, float("inf"), None
        _off_pos = None
        try:
            for tag in traj:
                if best_pos is not None and (
                        time.time() - _t_bo) + _max_traj > budget_s:
                    print(f"  [BO2] skip '{tag}' - wall budget "
                          f"({budget_s:.0f}s) would be exceeded", flush=True)
                    continue
                if tag == "off":
                    os.environ["XP_CONG_LOSS"] = "0"
                else:
                    os.environ["XP_CONG_LOSS"] = "1"
                    os.environ["XP_CONG_LOSS_KERNEL"] = tag
                print(f"  [BO2] === trajectory '{tag}' ===", flush=True)
                _t_traj = time.time()
                # Per-trajectory crash guard: one trajectory failing (e.g. GPU
                # OOM - 3 sequential pipelines fragment an 8GB allocator) must
                # not crash the placement. Log, free the GPU, and keep the best
                # completed one ('off' runs FIRST, so a fallback always exists).
                try:
                    pos = self.place(benchmark)
                    _max_traj = max(_max_traj, time.time() - _t_traj)
                    p = float(getattr(self, "_last_proxy", 1e9))
                    # DQ-safety: only ever SELECT a trajectory that is legal
                    # (zero hard-macro overlap + in canvas bounds). A lower-
                    # proxy but invalid trajectory must never be shipped.
                    # 'off' is the validated baseline + guaranteed fallback.
                    _pn = pos.detach().cpu().numpy().astype(np.float64)
                    _nhb = benchmark.num_hard_macros
                    _szb = benchmark.macro_sizes.numpy().astype(np.float64)
                    _cwb = float(benchmark.canvas_width)
                    _chb = float(benchmark.canvas_height)
                    _hwb = _szb[:, 0] * 0.5
                    _hhb = _szb[:, 1] * 0.5
                    _oob = bool(((_pn[:, 0] - _hwb < -1e-6)
                                 | (_pn[:, 0] + _hwb > _cwb + 1e-6)
                                 | (_pn[:, 1] - _hhb < -1e-6)
                                 | (_pn[:, 1] + _hhb > _chb + 1e-6)).any())
                    _valid = (not _oob) and not _hard_overlap_exists(
                        _pn, _szb, _nhb)
                    print(f"  [BO2] {tag} final_proxy={p:.6f} "
                          f"valid={int(_valid)}", flush=True)
                    if tag == "off":
                        _off_pos = pos  # validated baseline; emergency fallback
                    if _valid and p < best_p:
                        best_p, best_pos, best_tag = p, pos, tag
                except Exception as _bo_e:
                    print(f"  [BO2] trajectory '{tag}' FAILED "
                          f"({type(_bo_e).__name__}: {_bo_e}); skipping, "
                          f"keeping best-so-far", flush=True)
                finally:
                    # Free GPU between trajectories - 3 sequential pipelines
                    # fragment the 8GB allocator and OOM by trajectory 3.
                    # Runs on BOTH success and failure paths.
                    gc.collect()
                    torch.cuda.empty_cache()
        finally:
            self._in_bo2 = False
            for _k, _v in zip(("XP_CONG_LOSS", "XP_CONG_LOSS_KERNEL"), _saved):
                if _v is None:
                    os.environ.pop(_k, None)
                else:
                    os.environ[_k] = _v
        if best_pos is None:
            # No trajectory passed the legality gate.
            if _off_pos is not None:
                # Ship the validated 'off' baseline.
                best_pos, best_tag = _off_pos, "off(fallback)"
            else:
                # EVERY trajectory crashed (e.g. GPU OOM on all, or no CUDA)
                # - return a best-effort LEGAL placement rather than None,
                # which would crash the evaluator -> DQ.
                print("  [BO2] all trajectories failed - emergency legal "
                      "fallback", flush=True)
                return _best_effort_legal_tensor(benchmark)
        print(f"  [BO2] selected='{best_tag}' proxy={best_p:.6f}", flush=True)
        return best_pos

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        """Run the full placement pipeline: GP -> refine cascade -> eval."""
        # Delegate to the best-of-N wrapper unless it is already driving us.
        if (os.environ.get("XP_RUDY_BO2", "1") == "1"
                and not getattr(self, "_in_bo2", False)):
            return self._place_best_of_n(benchmark)

        t0 = time.time()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        np.random.seed(42)
        # Deterministic CUDA: cuDNN and cuBLAS pinned, scatter_add_/index_add_
        # forced deterministic. warn_only=True because two things escape this
        # flag - Xplace's third-party kernels (nondeterministic in principle,
        # bit-stable in practice) and the Triton atomics in decode_scatter /
        # polish_emit.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

        # CPU-only safety: this pipeline needs CUDA. Without a GPU every stage
        # raises, but those raises are caught per trajectory and the best-of-N
        # layer falls back to _best_effort_legal_tensor - a valid result rather
        # than a DQ. No full CPU re-placement path on purpose: untested code is
        # riskier here than the fallback.
        if not torch.cuda.is_available():
            print("  [warn] CUDA not available - GPU stages will fail; relying on "
                  "the legal-fallback safety net (no full CPU placement path).",
                  flush=True)

        # Flush leftover GPU memory from any previous benchmark.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        placement = benchmark.macro_positions.clone()

        cache_dir = os.path.join(_REPO_ROOT, "benchmarks", "xplace_cache")
        os.makedirs(cache_dir, exist_ok=True)
        lef_file = os.path.join(cache_dir, f"{benchmark.name}.lef")
        def_file = os.path.join(cache_dir, f"{benchmark.name}.def")
        cache_hit = os.path.exists(lef_file) and os.path.exists(def_file)

        # Load the .plc on a background thread: the C++ parser releases the
        # GIL, so Xplace init and the LEF/DEF parse overlap it.
        plc_holder = {}

        def _load_plc_worker():
            plc_holder['plc'] = self._load_plc(benchmark.name)

        plc_thread = threading.Thread(target=_load_plc_worker, daemon=True)
        plc_thread.start()

        # On cache miss, plc is on the critical path (needed to generate LEF/DEF).
        if not cache_hit:
            plc_thread.join()
            plc = plc_holder.get('plc')
            if plc is None:
                return placement
            _generate_lefdef(benchmark, plc, lef_file, def_file)

        # Xplace init + LEF/DEF parse (overlaps with plc load on cache hit).
        _ensure_xplace()
        device = torch.device("cuda:0")
        xp_logger = logging.getLogger("xplace")
        xp_logger.setLevel(logging.CRITICAL)
        parsed = self._parse_lefdef(benchmark, lef_file, def_file)

        # plc must be ready before the prep thread reads it.
        plc_thread.join()
        plc = plc_holder.get('plc')
        if plc is None:
            return placement

        # Prep for refine + eval. Runs alongside GP's GPU-bound kernels, which
        # drop the GIL during CUDA ops.
        prep = {}

        def _prep_worker():
            pin_map = {}
            for idx, mod in enumerate(plc.modules_w_pins):
                if mod.get_type() == 'MACRO_PIN' and hasattr(mod, 'get_macro_name'):
                    name = mod.get_macro_name()
                    pin_map.setdefault(name, []).append(idx)
            plc._macro_pin_map = pin_map

            pin_net = self._extract_pin_net_data(benchmark, plc)
            # Static pin-count tables. Pure functions of pin_macro/nh/n_macros,
            # so the operator passes that need them share one cached copy.
            _bench_nh = benchmark.num_hard_macros
            _bench_n = benchmark.num_macros
            _pm_arr = pin_net['pin_macro']
            _pm_hard_filt = _pm_arr[(_pm_arr >= 0) & (_pm_arr < _bench_nh)]
            _pm_soft_filt = _pm_arr[(_pm_arr >= _bench_nh) & (_pm_arr < _bench_n)]
            pin_net['hard_pin_counts'] = np.bincount(_pm_hard_filt, minlength=_bench_nh).astype(np.int64)
            pin_net['soft_pin_counts_full'] = np.bincount(_pm_soft_filt, minlength=_bench_n).astype(np.int64)
            prep['pin_net'] = pin_net

            # Combined pin-net CSR for the C refiner, built with a vectorized
            # cumsum plus a boolean-mask scatter.
            nd = pin_net['net_driver']
            nso = pin_net['net_sinks_off']
            nsi = pin_net['net_sinks_idx']
            nn_p = nd.shape[0]
            sinks_per_net = (nso[1:] - nso[:-1]).astype(np.int32)
            net_pin_off = np.zeros(nn_p + 1, dtype=np.int32)
            net_pin_off[1:] = np.cumsum(sinks_per_net + 1, dtype=np.int32)
            total_refs = int(net_pin_off[nn_p])
            net_pin_idx = np.empty(total_refs, dtype=np.int32)
            # Drivers land at each net's start offset; remaining slots are
            # sinks in net-major order.
            net_pin_idx[net_pin_off[:-1]] = nd
            is_driver = np.zeros(total_refs, dtype=bool)
            is_driver[net_pin_off[:-1]] = True
            net_pin_idx[~is_driver] = nsi
            prep['pin_csr'] = {
                'pin_macro': pin_net['pin_macro'],
                'pin_x': pin_net['pin_x'],
                'pin_y': pin_net['pin_y'],
                'net_pin_off': net_pin_off,
                'net_pin_idx': net_pin_idx,
                'net_weight': pin_net['net_weight'],
            }

            # Preload C extension libs to skip dlopen cost on first timed use.
            for _key in ("leg", "refine", "cong"):
                _ensure_lib(_key)

        prep_thread = threading.Thread(target=_prep_worker, daemon=True)
        prep_thread.start()

        # STAGE: GP - Xplace global placement.
        print("  == gp ==")
        gp_result = self._run_gp(benchmark, parsed, device, xp_logger)
        if gp_result is not None:
            positions, hpwl, overflow, gp_time, iters = gp_result
            cur = self._map_positions(positions, benchmark)
        else:
            cur = benchmark.macro_positions.numpy().astype(np.float64).copy()

        prep_thread.join()
        pn = prep['pin_net']
        octx = self._build_opt_context(benchmark, prep['pin_csr'])
        n_macros = octx['n']
        nh = octx['nh']
        cw, ch = octx['cw'], octx['ch']
        sizes = octx['sizes_np']
        movable = octx['movable_np']
        mov_i32 = movable.astype(np.int32)

        leg = {
            'nh': nh,
            'sizes': sizes[:nh],
            'movable': movable[:nh],
            'cw': cw, 'ch': ch,
            'hw': sizes[:nh, 0] / 2.0,
            'hh': sizes[:nh, 1] / 2.0,
        }

        # wl_extra_weight: per-net multiplier on the WL gradient only
        # (refine.c:265, congestion.c); None means uniform. XP_CRIT_WEIGHTS=
        # fanout biases toward high-fanout nets as a stand-in for slack-driven
        # weighting, which would need a cluster->ODB net mapping this placer does
        # not build. Strength via XP_CRIT_ALPHA: weight = 1 + alpha*log1p(fanout-1).
        wl_extra_weight = None
        _crit_mode = os.environ.get("XP_CRIT_WEIGHTS", "")
        if _crit_mode == "fanout":
            _alpha = float(os.environ.get("XP_CRIT_ALPHA", "1.0"))
            # fanout per net = 1 (driver) + n_sinks
            _nso = pn['net_sinks_off']
            _fanout = (_nso[1:] - _nso[:-1]) + 1  # int per net
            _w = 1.0 + _alpha * np.log1p(_fanout - 1).astype(np.float64)
            wl_extra_weight = np.ascontiguousarray(_w, dtype=np.float64)
            print(f"    wl_extra_weight: fanout-weighted, alpha={_alpha:.2f}, "
                  f"range=[{_w.min():.2f}, {_w.max():.2f}], "
                  f"mean={_w.mean():.2f}, nets={len(_w)}")

        # Design facts the stage-4 operator passes all need, bundled once.
        design_ctx = DesignContext(
            n_macros=n_macros, nh=nh, sizes=sizes, mov_i32=mov_i32,
            cw=cw, ch=ch, octx=octx, pn=pn,
            wl_extra_weight=wl_extra_weight)

        # Spawn the ILS pool now so worker startup overlaps the cascade.
        # hotspot_sort=1 enables the hot-region soft list and cold-macro
        # pruning.
        ils_worker_args = _ils_pack_args(sizes, movable, benchmark, nh,
                                         cw, ch, octx, pn, hotspot_sort=1,
                                         wl_extra_weight=wl_extra_weight)
        ils_pool = ILSWorkerPool(
            ils_worker_args,
            k=_ILS_POLISH_K,
            omp_threads=_ILS_POLISH_OMP,
            so_path=os.path.join(_EXT_DIR, _C_EXTENSIONS["cong"]["so"]),
            async_start=True,
        )

        # Holds the last cong-c breakdown - used as the proxy eval source
        # downstream.
        cong_c_stats = {}

        # Build the persistent GPUState async during Stage 1 so its ~3.4s
        # from_pn cost overlaps refine + cc-cleanup. Topology and allocation
        # only - positions are refreshed on first use.
        _gpu_state_holder = {"state": None, "thread": None}
        _gpu_state_init_pos = cur.copy()

        def _build_gpu_state_blocking():
            try:
                from abuplace.GPU.state import GPUState as _GS
                _gpu_state_holder["state"] = _GS.from_pn(
                    pn=pn, pin_csr=octx['pin_csr'], octx=octx,
                    sizes_np=sizes, positions_np=_gpu_state_init_pos,
                    movable_np=mov_i32, n_macros=n_macros, n_hard=nh)
            except Exception as _exc:
                print(f"  [persistent-state] build failed: "
                      f"{type(_exc).__name__}: {_exc}; falling back")
                _gpu_state_holder["state"] = None

        _gpu_state_holder["thread"] = threading.Thread(
            target=_build_gpu_state_blocking, daemon=True,
            name="gpu_state_build")
        _gpu_state_holder["thread"].start()

        def _get_or_build_gpu_state():
            th = _gpu_state_holder["thread"]
            if th is not None:
                th.join()
                _gpu_state_holder["thread"] = None
            return _gpu_state_holder["state"]

        def _refresh_gpu_state(cur_pos):
            """Sync persistent GPUState.pos to cur_pos and re-run init_state;
            returns the state or None on failure."""
            s = _get_or_build_gpu_state()
            if s is None:
                return None
            try:
                from abuplace.GPU.init import init_state as _init
                s.pos.copy_(torch.from_numpy(np.ascontiguousarray(
                    cur_pos, dtype=np.float64)).to(s.device, s.dtype))
                _init(s)
                return s
            except Exception as _exc:
                print(f"    [refresh-state] failed: "
                      f"{type(_exc).__name__}: {_exc}")
                return None

        def _cc_call(cur_pos, n_iter=0, lr_bins=2.0, extra_phases=None,
                     hotspot_sort=0):
            r = _cong_relax_v2_c(
                cur_pos, sizes, mov_i32,
                n=n_macros, nh=nh, cw=cw, ch=ch,
                gr=octx['gr'], gc=octx['gc'],
                smooth_range=pn['smooth_range'],
                hrpm=pn['hrpm'], vrpm=pn['vrpm'],
                h_alloc=pn['h_alloc'], v_alloc=pn['v_alloc'],
                pin_macro=pn['pin_macro'], pin_x=pn['pin_x'], pin_y=pn['pin_y'],
                net_driver=pn['net_driver'],
                net_sinks_off=pn['net_sinks_off'],
                net_sinks_idx=pn['net_sinks_idx'],
                net_weight=pn['net_weight'],
                net_cnt_for_norm=pn['net_cnt_for_norm'],
                n_iter=n_iter, lr_bins=lr_bins,
                hotspot_sort=hotspot_sort,
                extra_phases=extra_phases,
                wl_extra_weight=wl_extra_weight,
            )
            return r

        def _run_cong_c(cur_pos, n_iter, lr_bins, label, extra_phases=None):
            t_cc = time.time()
            (new_pos, c_init, c_final, p_init, p_final,
             wl_final, den_final) = _cc_call(
                cur_pos, n_iter=n_iter, lr_bins=lr_bins,
                extra_phases=extra_phases)
            # Refresh cong_c_stats from a fresh-init proxy. The polish-internal
            # p_final drifts from a fresh re-init by 0.001-0.030
            # (density-formula mismatch plus FP accumulation), which would
            # mislead every downstream stage that reads it.
            if n_iter > 0:
                (_, c_fr, _, p_fr, _, wl_fr, den_fr) = _cc_call(
                    new_pos, n_iter=0, lr_bins=2.0)
                print(f"    [{label}] cong={c_init:.4f}->{c_fr:.4f} "
                      f"proxy={p_init:.4f}->{p_fr:.4f} "
                      f"(polish-internal {p_final:.4f}) "
                      f"{time.time()-t_cc:.2f}s")
                cong_c_stats['proxy'] = p_fr
                cong_c_stats['wl'] = wl_fr
                cong_c_stats['den'] = den_fr
                cong_c_stats['cong'] = c_fr
            else:
                # n_iter=0 already gives fresh-init proxy.
                print(f"    [{label}] cong={c_init:.4f}->{c_final:.4f} "
                      f"proxy={p_init:.4f}->{p_final:.4f} "
                      f"{time.time()-t_cc:.2f}s")
                cong_c_stats['proxy'] = p_final
                cong_c_stats['wl'] = wl_final
                cong_c_stats['den'] = den_final
                cong_c_stats['cong'] = c_final
            return new_pos

        def _run_gpu_hybrid_5perturb(cur_pos, label="cong-c hybrid5p"):
            """GPU init polish + 5x perturb-and-brief + C-short refine, drop-in
            for _run_cong_c; falls back to the C polish on exception
            (XP_GPU_HYBRID_VERBOSE=1 prints per-stage timings)."""
            from abuplace.GPU.polish_hybrid_5perturb import (
                gpu_hybrid_5perturb_polish)
            t_cc = time.time()
            verbose_hybrid = os.environ.get(
                "XP_GPU_HYBRID_VERBOSE", "0") == "1"
            try:
                (new_pos, c_init, c_final, p_init, p_final,
                 wl_final, den_final) = gpu_hybrid_5perturb_polish(
                    cur_pos, sizes, mov_i32, n_macros, nh, cw, ch,
                    octx, pn,
                    c_short_call=lambda p: _cc_call(
                        p, n_iter=3, lr_bins=1.0,
                        extra_phases=_CC_RECONCILE_PHASES),
                    verbose=verbose_hybrid,
                    state=_get_or_build_gpu_state(),
                )
            except Exception as exc:
                import traceback
                print(f"    [{label}] GPU hybrid FAILED: "
                      f"{type(exc).__name__}: {exc}")
                if verbose_hybrid:
                    traceback.print_exc()
                print(f"    [{label}] falling back to C polish "
                      f"(n_iter=12 + 6/6/6/6 extras)")
                return _run_cong_c(
                    cur_pos, n_iter=12, lr_bins=2.0, label=label,
                    extra_phases=[(6, 1.5), (6, 0.5), (6, 0.15), (6, 0.05)])
            wall = time.time() - t_cc
            # Mirror _run_cong_c's log shape with a hybrid5p tag.
            print(f"    [{label}] cong={c_init:.4f}->{c_final:.4f} "
                  f"proxy={p_init:.4f}->{p_final:.4f} "
                  f"wl={wl_final:.4f} den={den_final:.4f} {wall:.2f}s")
            cong_c_stats['proxy'] = p_final
            cong_c_stats['wl'] = wl_final
            cong_c_stats['den'] = den_final
            cong_c_stats['cong'] = c_final
            return new_pos

        # STAGE: TUNE - post-GP cong probe picks R1 cong_mult and gates micro-move.
        print("  == tune ==")
        _gp_cc = _cc_call(cur.copy())
        c_probe = _gp_cc[1]
        gp_proxy = float(_gp_cc[3])
        gp_wl = float(_gp_cc[5])
        gp_den = float(_gp_cc[6])
        print(f"    [post-GP] proxy={gp_proxy:.6f} cong={c_probe:.4f} "
              f"wl={gp_wl:.4f} den={gp_den:.4f}")
        if c_probe >= 1.9:
            cong_r1 = 4.0
        elif c_probe >= 1.4:
            cong_r1 = 3.5
        else:
            cong_r1 = 2.5
        use_micro_move = c_probe < 1.9
        print(f"    gp_cong={c_probe:.3f}  ->  cong_r1={cong_r1}  "
              f"micro_move={'on' if use_micro_move else 'off'}")

        # STAGE 1: REFINE - Adam (R1/R2/R3) + cong-relax cleanup.
        print("  == stage 1: refine ==")
        _t_stage1 = time.time()
        # R1: Adam refine + legalize.
        refined = self._refine_and_legalize(
            cur, octx,
            RefineConfig(cong_mult=cong_r1, rep_margin=0.10, lr=0.05,
                         n_iters=250),
            "R1", leg)
        if refined is not None:
            cur = refined
        # R1-cc: shallow cleanup that populates cong_c_stats. Its congestion
        # reading drives the adaptive R2/R3 cong_mult below.
        cur = _run_cong_c(cur, n_iter=6, lr_bins=2.0, label="R1-cc")

        # On low-congestion designs a small cong_mult bump (~1.05-1.10) lets
        # the Stage 2 polish settle into a better basin.
        post_r1_cong = cong_c_stats['cong']
        if c_probe < _ADAPT_C_PROBE_GATE:
            r2_cong = 1.0 + _ADAPT_R2_SCALE * max(0.0, post_r1_cong - 1.0)
            r3_cong = 1.0 + _ADAPT_R3_SCALE * max(0.0, post_r1_cong - 1.0)
            print(f"    adaptive R2 cong_mult={r2_cong:.2f} R3={r3_cong:.2f}"
                  f"  (post-R1 cong={post_r1_cong:.3f})")
        else:
            r2_cong = 1.0
            r3_cong = 1.0

        # R2: Adam + legalize, no cc-cleanup - small-step Adam doesn't shift
        # the basin enough.
        refined = self._refine_and_legalize(
            cur, octx, RefineConfig(cong_mult=r2_cong), "R2", leg)
        if refined is not None:
            cur = refined

        # R3: smaller-step Adam + legalize, then a shallow cc cleanup.
        refined = self._refine_and_legalize(
            cur, octx, RefineConfig(lr=0.003, cong_mult=r3_cong), "R3", leg)
        if refined is not None:
            cur = refined
        cur = _run_cong_c(cur, n_iter=6, lr_bins=2.0, label="R3-cc")

        # Regression guard - snapshot the post-refine VALID baseline, shipped at
        # return if the whole cascade ended up worse than it. A catastrophe
        # guard, not a best-of-cascade pick: taking the best intermediate is the
        # proxybest dead-end that inverts at the polish. "Final strictly worse
        # than refine" never trips on the healthy path.
        _refine_cur = cur.copy()
        _refine_proxy = float(cong_c_stats.get('proxy', float('inf')))
        print(f"    stage 1 total: {time.time()-_t_stage1:.2f}s "
              f"proxy={cong_c_stats['proxy']:.4f}")

        # Stage 2 prep: 4-way +-0.5-bin hard micro-moves. Skipped on
        # high-congestion designs, where fast/exact drift costs more than the
        # moves gain.
        if use_micro_move:
            print("  --- stage 2 prep: micro-move ---")
            dx_mm = octx['gw'] * 0.5
            dy_mm = octx['gh'] * 0.5
            with CongState(cur, sizes, mov_i32, n_macros, nh,
                           cw, ch, octx, pn,
                           wl_extra_weight=wl_extra_weight) as cs_mm:
                moved = self._micro_move(cur, nh, sizes, cw, ch,
                                         dx_mm, dy_mm, cs_mm)
            # Micro-moves can introduce overlaps; re-legalize before polish.
            legalized, overlap = _legalize_two_pass(moved[:nh], leg, label="micro_move")
            if not overlap:
                moved[:nh] = legalized
                cur = moved
                print("    post-legalize: ok")
            else:
                print("    post-legalize: overlap remains; keeping pre-move")

        # STAGE 2: POLISH - GPU init + 5xperturb + c-short.
        print("  == stage 2: polish ==")
        _t_stage2 = time.time()
        # XP_POLISH_IS_TOPK=999 commits every improving candidate per
        # independent set; XP_POLISH_IS_BBOX=0 skips the bbox check when
        # building the partition.
        os.environ.setdefault("XP_POLISH_IS_TOPK", "999")
        os.environ.setdefault("XP_POLISH_IS_BBOX", "0")
        # OOM/GPU guard: a polish failure (e.g. GPU OOM on a large design)
        # skips this stage rather than dropping the whole trajectory - `cur`
        # keeps its valid post-R3 value. Happy path unchanged.
        try:
            cur = _run_gpu_hybrid_5perturb(cur, label="gpu+5perturb+c-short")
        except Exception as _p_exc:
            print(f"    [stage 2] polish FAILED ({type(_p_exc).__name__}: "
                  f"{_p_exc}); skipping, keeping post-refine placement", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"    stage 2 total: {time.time()-_t_stage2:.2f}s "
              f"proxy={cong_c_stats['proxy']:.4f}")

        # STAGE 3: ESCAPE - basin-jump v3 (GPU chain + CPU operator stack).
        # XP_BJ_BUDGET overrides the wall budget; v3_applied tells the later
        # stages which work v3 already covered.
        v3_applied = False
        print("  == stage 3: escape (basin-jump v3) ==")
        t_bj = time.time()
        try:
            from abuplace.GPU.basin_jump_v3 import basin_jump_v3
            pre_bj_pos = cur.copy()
            pre_bj_proxy = cong_c_stats.get('proxy')
            _legalize_for_bj = lambda pos_hard, label="bj_internal": _legalize_two_pass(pos_hard, leg, label=label)
            # Hand v3 the persistent gpu_state so its inner basin_jump_v2 skips
            # ~3.4s of GPUState.from_pn.
            _refresh_gpu_state(cur)
            bj_result = basin_jump_v3(
                cur, sizes, mov_i32, n_macros, nh, cw, ch, octx, pn,
                cc_call=_cc_call,
                gpu_state=_get_or_build_gpu_state(),
                pre_bj_proxy=pre_bj_proxy,
                wl_extra_weight=wl_extra_weight,
                verbose=True,
                legalize_fn=_legalize_for_bj)
            bj_cur = bj_result['cur']
            bj_proxy = bj_result['final_proxy']
            pre_bj_proxy_raw = bj_result['base_proxy']
            accepted_rounds = bj_result['accepted_rounds']
            aborted = bj_result['aborted_first']
            chain_wall = time.time() - t_bj
            print(f"    chain band-cong={bj_result['cong']:.3f} "
                  f"rounds={bj_result['rounds_run']}/{bj_result['total_rounds']} "
                  f"accepted={accepted_rounds} "
                  f"c-proxy {pre_bj_proxy_raw:.4f}->{bj_proxy:.4f} "
                  f"chain={chain_wall:.2f}s"
                  f"{' [aborted on B1 reject]' if aborted else ''}")

            if accepted_rounds > 0:
                # bj_cur is already row-legal (v3 legalized + final-polished
                # internally).
                cur = _run_cong_c(bj_cur, n_iter=2, lr_bins=0.15,
                                   label="bj-cc")
                post_proxy = cong_c_stats['proxy']
                if (pre_bj_proxy is not None
                        and post_proxy >= pre_bj_proxy - 1e-6):
                    print(f"    rebuild-verify {post_proxy:.4f} "
                          f">= pre {pre_bj_proxy:.4f}; revert")
                    cur = pre_bj_pos
                    cur = _run_cong_c(cur, n_iter=0, lr_bins=2.0,
                                      label="bj-revert-cc")
                else:
                    print(f"    post-bj {post_proxy:.4f} "
                          f"(vs pre {pre_bj_proxy:.4f})")
                    v3_applied = True
            else:
                print("    no accepted rounds; skipping bj-cc")
            print(f"    stage 3 total: {time.time()-t_bj:.2f}s "
                  f"proxy={cong_c_stats['proxy']:.4f}")
        except Exception as _bj_exc:
            print(f"    [stage 3] FAILED: "
                  f"{type(_bj_exc).__name__}: {_bj_exc}; skipping")

        # STAGE 4: OPERATE - discrete-move operators (4a-4f).
        # 4a hard-shake: 8-direction, 0.125-bin moves on the top-K hot hard
        # macros. CongState screens the candidates, then an n_iter=2
        # mini-polish verifies - the fast screen and the exact score can
        # diverge.
        print("  == stage 4a: operate (hard-shake) ==")
        t_shake = time.time()
        gw_shake = octx['gw']
        gh_shake = octx['gh']
        dx_shake = gw_shake * _SHAKE_STEP_REL
        dy_shake = gh_shake * _SHAKE_STEP_REL
        shake_deltas = [
            (dx_shake, 0), (-dx_shake, 0), (0, dy_shake), (0, -dy_shake),
            (dx_shake, dy_shake), (-dx_shake, dy_shake),
            (dx_shake, -dy_shake), (-dx_shake, -dy_shake),
        ]
        pm_shake = pn['pin_macro']
        pm_valid_shake = pm_shake[(pm_shake >= 0) & (pm_shake < nh)]
        pin_counts_shake = np.bincount(pm_valid_shake, minlength=nh)
        shake_order = np.argsort(-pin_counts_shake)[:_SHAKE_TOP_K]

        # CongState incremental scoring is ~30x faster per probe than a fresh
        # rebuild.
        with CongState(cur, sizes, mov_i32, n_macros, nh, cw, ch, octx, pn,
                       wl_extra_weight=wl_extra_weight) as cs:
            cur_fast, _, _, _ = cs.score()
            shake_candidates = []
            for m in shake_order:
                mi = int(m)
                ox, oy = cur[m, 0], cur[m, 1]
                hw_m, hh_m = sizes[m, 0] * 0.5, sizes[m, 1] * 0.5
                for ddx, ddy in shake_deltas:
                    nx, ny = ox + ddx, oy + ddy
                    if nx - hw_m < 0 or nx + hw_m > cw: continue
                    if ny - hh_m < 0 or ny + hh_m > ch: continue
                    p, _, _, _ = cs.apply([mi], [nx], [ny])
                    if p < cur_fast - 1e-7:
                        shake_candidates.append(
                            (p - cur_fast, mi, float(ddx), float(ddy)))
                    cs.revert()
        shake_candidates.sort()

        def _shake_verify(pos):
            (_, _, _, _, p_final, _, _) = _cc_call(
                pos.copy(), n_iter=2, lr_bins=1.0)
            return p_final

        # Verify in parallel on the otherwise-idle ILS pool (each worker owns
        # its libgomp runtime). Falls back to sequential on any failure.
        cur_exact = _shake_verify(cur)
        shake_accepted = 0
        to_verify = shake_candidates[:_SHAKE_VERIFY_N]
        if to_verify:
            n_par = min(len(to_verify), ils_pool.k)
            payloads = []
            cands = []
            for fast_d, m, ddx, ddy in to_verify[:n_par]:
                pos_try = cur.copy()
                pos_try[m, 0] += ddx
                pos_try[m, 1] += ddy
                payloads.append({
                    "pos": pos_try, "n_iter": 2, "lr_bins": 1.0,
                })
                cands.append((int(m), float(ddx), float(ddy)))
            try:
                ils_pool.submit_verify(payloads)
                results = ils_pool.collect_verify()
            except Exception as e:
                print(f"    [shake] pool verify failed ({e}); falling back to "
                      f"sequential")
                results = None

            if results is not None and any(r is not None for r in results):
                # Best-first commit. The top-N hot macros are spatially
                # distinct, so their per-move gains compose; shake-cc
                # reconciles afterwards.
                pairs = [
                    (r["proxy_final"], m, ddx, ddy)
                    for (m, ddx, ddy), r in zip(cands, results)
                    if r is not None
                ]
                pairs.sort(key=lambda x: x[0])
                for p_final, m, ddx, ddy in pairs:
                    if p_final < cur_exact - 1e-7:
                        cur_exact = min(cur_exact, p_final)
                        cur[m, 0] += ddx
                        cur[m, 1] += ddy
                        shake_accepted += 1
            else:
                # Fallback: sequential verify.
                for fast_d, m, ddx, ddy in to_verify:
                    old_x, old_y = cur[m, 0], cur[m, 1]
                    cur[m, 0] = old_x + ddx
                    cur[m, 1] = old_y + ddy
                    p_exact = _shake_verify(cur)
                    if p_exact < cur_exact - 1e-7:
                        cur_exact = p_exact
                        shake_accepted += 1
                    else:
                        cur[m, 0] = old_x
                        cur[m, 1] = old_y
        print(f"    cands={len(shake_candidates)} accept={shake_accepted}  "
              f"exact={cur_exact:.4f}  ({time.time()-t_shake:.2f}s)")

        if shake_accepted > 0:
            # Legalize before polishing - the reverse order would let
            # cong_c_stats hide the cost of the legalize.
            legalized, overlap = _legalize_two_pass(cur[:nh], leg, label="hard_shake")
            if not overlap:
                cur[:nh] = legalized
            cur = _run_cong_c(cur, n_iter=3, lr_bins=0.5, label="shake-cc",
                              extra_phases=_CC_RECONCILE_PHASES)
        print(f"    stage 4a total: {time.time()-t_shake:.2f}s "
              f"proxy={cong_c_stats['proxy']:.4f}")

        # 4b pair-swap: swap same-size hard pairs whose fast proxy drops, then
        # re-polish. Skipped at c_probe >= 1.9, where the gain saturates and
        # fast/exact drift turns it into a regression.
        print("  == stage 4b: operate (pair-swap) ==")
        t_ps0 = time.time()
        if c_probe < 1.5:
            sw_budget, ps_sched = 1.0, [(3, 0.5), (3, 0.15), (3, 0.05)]
        elif c_probe < 1.9:
            sw_budget, ps_sched = 3.0, [(6, 0.5), (3, 0.05)]
        else:
            sw_budget, ps_sched = 0.0, [(3, 0.5), (3, 0.15), (3, 0.05)]

        if sw_budget > 0:
            tested = 0
            accepted = 0
            pm = pn['pin_macro']
            pm_valid = pm[(pm >= 0) & (pm < nh)]
            pin_counts = np.bincount(pm_valid, minlength=nh)
            buckets = {}
            for i in range(nh):
                key = (round(float(sizes[i, 0]), 6),
                       round(float(sizes[i, 1]), 6))
                buckets.setdefault(key, []).append(i)
            pairs = []
            for group in buckets.values():
                if len(group) < 2:
                    continue
                for a in range(len(group)):
                    for b in range(a + 1, len(group)):
                        i, j = group[a], group[b]
                        pairs.append(
                            (int(pin_counts[i]) + int(pin_counts[j]), i, j))
            pairs.sort(reverse=True)

            # CongState is ~30x cheaper per probe than a full rebuild but
            # accumulates FP drift that eventually fakes an improvement - hence
            # the periodic rebuild(). sw_budget caps wall time per c_probe band;
            # SW_EARLY_EXIT bails once the hottest-pin pairs stop accepting
            # (colder pairs essentially never do).
            REBUILD_EVERY = 30
            SW_EARLY_EXIT = 500
            t_sw = time.time()
            pre_swap_pos = cur.copy()
            with CongState(cur, sizes, mov_i32, n_macros, nh,
                           cw, ch, octx, pn,
                           wl_extra_weight=wl_extra_weight) as cs:
                cur_fast, _, _, _ = cs.score()
                commits_since_rebuild = 0
                for (_, i, j) in pairs:
                    if accepted == 0 and tested >= SW_EARLY_EXIT:
                        break
                    if time.time() - t_sw > sw_budget:
                        break
                    xi, yi = cur[i, 0], cur[i, 1]
                    xj, yj = cur[j, 0], cur[j, 1]
                    p, _, _, _ = cs.apply([i, j], [xj, xi], [yj, yi])
                    tested += 1
                    if p < cur_fast - 1e-7:
                        cur[[i, j]] = cur[[j, i]]
                        cur_fast = p
                        accepted += 1
                        cs.commit()
                        commits_since_rebuild += 1
                        if commits_since_rebuild >= REBUILD_EVERY:
                            cs.rebuild()
                            cur_fast, _, _, _ = cs.score()
                            commits_since_rebuild = 0
                    else:
                        cs.revert()
            # Fresh-rebuild verify catches fast-proxy drift; revert if post >= pre.
            if accepted > 0:
                pre_p = float(_cc_call(pre_swap_pos, n_iter=0, lr_bins=2.0)[3])
                post_p = float(_cc_call(cur, n_iter=0, lr_bins=2.0)[3])
                print(f"    swap fresh-verify pre={pre_p:.4f} post={post_p:.4f}")
                if post_p >= pre_p - 1e-6:
                    print(f"    swap rebuild-verify {post_p:.4f} >= "
                          f"pre {pre_p:.4f}; revert")
                    cur = pre_swap_pos
                    accepted = 0
            print(f"    swap tested={tested} accepted={accepted} "
                  f"({time.time()-t_sw:.2f}s)")

        # With no swaps accepted, cur is unchanged and the previous stage's
        # reconcile still holds - skip the legalize + ps-cc that would repeat
        # ~1.8s of identical work.
        if sw_budget > 0 and accepted > 0:
            legalized, overlap = _legalize_two_pass(cur[:nh], leg, label="pair_swap")
            if not overlap:
                cur[:nh] = legalized
            n0, lr0 = ps_sched[0]
            extra = ps_sched[1:] if len(ps_sched) > 1 else None
            cur = _run_cong_c(cur, n_iter=n0, lr_bins=lr0, label="ps-cc",
                              extra_phases=extra)
        print(f"    pair-swap total: {time.time()-t_ps0:.2f}s")

        # 4c hardmove: 16 candidates per hard macro (axis + diagonal at step,
        # plus 8 knight-like at 2*step), scored through CongState. One pass
        # when v3 already ran - its sweeps cover most of this - otherwise
        # three.
        print("  == stage 4c: operate (hardmove) ==")
        t_hm = time.time()
        hm_n_passes = 1 if v3_applied else 3
        if v3_applied:
            print("    v3 applied -> light hardmove (n_passes=1)")
        # CPU only: the GPU hardmove regressed, because scoring a batch against
        # a frozen baseline misses the gain from each accept within the batch.
        cur = self._hardmove_pass(
            cur, design_ctx, cong_c_stats, _run_cong_c, _cc_call, leg,
            n_passes=hm_n_passes)
        print(f"    hardmove total: {time.time()-t_hm:.2f}s")

        # 4d softmove: 16 candidates per soft macro at step=0.25. Only
        # hard-soft overlap is checked; soft-soft is the density term's job.
        print("  == stage 4d: operate (softmove) ==")
        t_sm = time.time()
        # OOM/GPU guard: softmove uses the GPU batched path; a failure skips
        # the stage (cur keeps its valid pre-softmove value) instead of dropping
        # the whole trajectory. Happy path unchanged.
        try:
            cur = self._softmove_pass(
                cur, design_ctx, cong_c_stats, _run_cong_c, _cc_call,
                gpu_state=_refresh_gpu_state(cur))
        except Exception as _sm_exc:
            print(f"    [stage 4d] softmove FAILED ({type(_sm_exc).__name__}: "
                  f"{_sm_exc}); skipping", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"    softmove total: {time.time()-t_sm:.2f}s")

        # 4e long-shake: 8 directions x 5 scales over the top-60 hard and
        # top-30 soft macros. Confined to c_probe in [1.85, 2.0) - outside that
        # band the density-formula mismatch turns it into a regression.
        if 1.85 <= c_probe < 2.0:
            print("  == stage 4e: operate (long-shake) ==")
            t_long = time.time()
            cur = self._long_shake(
                cur, design_ctx, cong_c_stats, _run_cong_c, _cc_call, leg)
            print(f"    long-shake total: {time.time()-t_long:.2f}s")
        else:
            print(f"  == stage 4e: operate (long-shake) == SKIP (c_probe={c_probe:.3f} "
                  f"outside [1.85, 2.0))")

        # 4f window-reorder: VNS-style k-tuple permutations of soft macros
        # (ABCDPlace, Lin et al. TCAD'20). Picks up the soft macros 4c-4e left
        # untouched.
        print("  == stage 4f: operate (window-reorder) ==")
        t_wr = time.time()
        cur = self._window_reorder(
            cur, design_ctx, cong_c_stats, _run_cong_c,
            top_W=12, K=5, budget_s=2.0)
        print(f"    window-reorder total: {time.time()-t_wr:.2f}s")

        _tier2_on = os.environ.get("XP_TIER2", "1") != "0"

        # 4g Klein-4 orientation (Tier-2): record the min-HPWL orientation per hard
        # macro for the orientations.pt sidecar. Non-destructive + no re-polish, so
        # Tier-1 proxy (which ignores orientation) is unchanged. Default-on;
        # XP_TIER2=0 disables the Tier-2 features for A/B baselines.
        self._last_orientations = None
        if _tier2_on:
            print("  == stage 4g: Klein-4 orientation (Tier-2 sidecar) ==")
            t_or = time.time()
            codes = self._klein4_orient(cur, pn, sizes, nh)
            self._last_orientations = codes
            uniq, cnt = np.unique(codes, return_counts=True)
            dist = {int(u): int(c) for u, c in zip(uniq, cnt)}
            print(f"    orientations N/FN/FS/S = "
                  f"{dist.get(0,0)}/{dist.get(1,0)}/{dist.get(2,0)}/{dist.get(3,0)}"
                  f"  ({time.time()-t_or:.2f}s)")

        # EVAL - reuse the last cc-reconcile's exact proxy.
        print("  == eval ==")
        proxy = cong_c_stats['proxy']
        print(f"    proxy={proxy:.4f}  wl={cong_c_stats['wl']:.3f}  "
              f"den={cong_c_stats['den']:.3f}  cong={cong_c_stats['cong']:.3f}  "
              f"(via cong-c)")
        # Regression guard (catastrophe only): if the full cascade ended up
        # strictly WORSE than the post-refine baseline, ship the baseline. Never
        # trips on the healthy path (polish/operators only improve proxy); guards
        # against a stage going haywire on an untested design.
        if proxy > _refine_proxy + 1e-9:
            print(f"  == regression-guard == final proxy {proxy:.4f} > post-refine "
                  f"{_refine_proxy:.4f}; reverting to refine baseline", flush=True)
            cur = _refine_cur
            proxy = _refine_proxy
        # 4h Tier-2 clearance: pre-push hard macros to >=12 um so we control the
        # final coordinates rather than the evaluator's blind push. SCORING.md
        # says it pushes to >=12 um regardless and advises leaving that gap in
        # the submission, so mirroring it here (same algorithm, same target)
        # cannot do worse. Reverts on residual overlap; Tier-1-safe because
        # _clearance_applies() is False on the tiny IBM canvases.
        # XP_CLEARANCE_UM=0 disables.
        clearance_um = float(os.environ.get("XP_CLEARANCE_UM", "12"))
        if _tier2_on and _clearance_applies(clearance_um, cw, ch, sizes, nh):
            t_cl = time.time()
            pushed = _clearance_push(cur, sizes, nh, cw, ch, clearance_um,
                                     movable=mov_i32[:nh].astype(bool))
            if not _hard_overlap_exists(pushed, sizes, nh):
                cur = pushed
                print(f"  == clearance == pushed hard macros to >={clearance_um:g}um "
                      f"PDN channels  ({time.time()-t_cl:.2f}s)")
            else:
                print(f"  == clearance == push left overlap; reverted "
                      f"(evaluator will handle)  ({time.time()-t_cl:.2f}s)")

            # 4h.2 PDN horizontal-band guarantee (Tier-2). When the narrowest
            # band is under target, reflow the hard macros into clean rows
            # separated by a uniform wide channel. Verifier-gated, so adequate
            # layouts are untouched. Costs wirelength and proxy (ariane136:
            # 0.656 -> 0.817) and does NOT actually fix PDN-0179 - see the
            # note above _pdn_min_channels(). XP_PDN_ROWSNAP=0 disables;
            # XP_PDN_CHANNEL_UM sets the target width (default 30 um).
            if os.environ.get("XP_PDN_ROWSNAP", "1") != "0":
                channel_um = float(os.environ.get("XP_PDN_CHANNEL_UM", "30"))
                hgap_um = float(os.environ.get("XP_PDN_HGAP", "12"))
                h_band, _v_band = _pdn_min_channels(cur, sizes, nh)
                if h_band < channel_um - 1e-3:
                    t_rs = time.time()
                    snapped = _row_snap_pdn(cur, sizes, nh, cw, ch, channel_um,
                                            min_hgap=hgap_um,
                                            movable=mov_i32[:nh].astype(bool))
                    if (snapped is not None
                            and not _hard_overlap_exists(snapped, sizes, nh)):
                        h_after, _ = _pdn_min_channels(snapped, sizes, nh)
                        if h_after >= channel_um - 1e-3:
                            cur = snapped
                            print(f"  == pdn-rowsnap == horiz band {h_band:.1f}->"
                                  f"{h_after:.1f}um (target {channel_um:g}); "
                                  f"rows reflowed  ({time.time()-t_rs:.2f}s)")
                        else:
                            print(f"  == pdn-rowsnap == snap verify failed "
                                  f"({h_after:.1f}<{channel_um:g}); kept push "
                                  f"({time.time()-t_rs:.2f}s)")
                    else:
                        # Could not tile at this channel (or left overlap): widen
                        # the pairwise push to the channel target as a best-effort
                        # fallback so at least pairwise gaps are bigger than the
                        # evaluator's 12 um, then accept only if overlap-free.
                        wider = _clearance_push(cur, sizes, nh, cw, ch, channel_um,
                                                movable=mov_i32[:nh].astype(bool))
                        if not _hard_overlap_exists(wider, sizes, nh):
                            cur = wider
                            print(f"  == pdn-rowsnap == tiling infeasible; widened "
                                  f"pairwise gaps to {channel_um:g}um instead "
                                  f"({time.time()-t_rs:.2f}s)")
                        else:
                            print(f"  == pdn-rowsnap == band {h_band:.1f}um tight but "
                                  f"snap+widen infeasible; left to evaluator "
                                  f"({time.time()-t_rs:.2f}s)")

        # Final legality repair (unconditional): the cascade should leave hard
        # macros overlap-free, but if anything slipped through, repair it here
        # so place() can NEVER return an overlapping placement (a clean DQ).
        # No-op when already overlap-free.
        if nh > 1 and _hard_overlap_exists(cur, sizes, nh):
            print("  == legality-repair == residual hard overlap found; "
                  "legalizing before return", flush=True)
            cur = _clearance_push(cur, sizes, nh, cw, ch, 0.0,
                                  movable=mov_i32[:nh].astype(bool), n_iters=300)

        # Final canvas-bounds clamp. The float32 cast can nudge a boundary macro
        # a sub-ULP past the edge, failing validate_placement at zero overlap -
        # a Tier-1 DQ. Clamp the float32 output (the dtype the evaluator checks),
        # movable macros only, since moving a fixed one trips a different check.
        # eps is far above a float32 ULP yet sub-micron, so it keeps x+w/2 in
        # bounds after rounding without moving the proxy.
        out = torch.tensor(cur, dtype=torch.float32)
        _eps = 1e-3
        _mov = torch.as_tensor(mov_i32, dtype=torch.bool)
        _hw = torch.as_tensor(sizes[:, 0] * 0.5, dtype=torch.float32)
        _hh = torch.as_tensor(sizes[:, 1] * 0.5, dtype=torch.float32)
        _nx = torch.minimum(torch.maximum(out[:, 0], _hw + _eps), float(cw) - _hw - _eps)
        _ny = torch.minimum(torch.maximum(out[:, 1], _hh + _eps), float(ch) - _hh - _eps)
        _clamped = ((_nx != out[:, 0]) | (_ny != out[:, 1])) & _mov
        out[:, 0] = torch.where(_mov, _nx, out[:, 0])
        out[:, 1] = torch.where(_mov, _ny, out[:, 1])
        n_clamped = int(_clamped.sum())
        if n_clamped:
            print(f"  == bounds-clamp == {n_clamped} movable macro(s) pulled "
                  f"inside canvas (float32 edge)")

        # NaN/Inf sanitization - the last DQ the overlap and bounds repairs
        # cannot catch: a degenerate net or divide-by-zero can emit a non-finite
        # coordinate that passes both yet fails validate_placement. Restore the
        # offending macros from their finite input positions and re-clamp; if
        # any remain non-finite, fall back to a fully legal placement.
        if not bool(torch.isfinite(out).all()):
            n_bad = int((~torch.isfinite(out)).any(dim=1).sum())
            print(f"  == nan-guard == {n_bad} macro(s) with non-finite coords; "
                  f"repairing before return", flush=True)
            _inp = benchmark.macro_positions.detach().cpu().to(torch.float32)
            out = torch.where(torch.isfinite(out), out, _inp)
            _cx = torch.minimum(torch.maximum(out[:, 0], _hw + _eps), float(cw) - _hw - _eps)
            _cy = torch.minimum(torch.maximum(out[:, 1], _hh + _eps), float(ch) - _hh - _eps)
            out[:, 0] = torch.where(_mov, _cx, out[:, 0])
            out[:, 1] = torch.where(_mov, _cy, out[:, 1])
            if not bool(torch.isfinite(out).all()):
                # Input itself was non-finite (pathological) - use the builder.
                out = _best_effort_legal_tensor(benchmark)

        print(f"  == done == proxy={proxy:.4f}  total={time.time()-t0:.1f}s")
        self._last_proxy = float(proxy)  # for XP_RUDY_BO2 trajectory selection
        ils_pool.close()
        return out
