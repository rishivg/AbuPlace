"""Persistent worker process for parallel cong_relax_v2 mini-polishes; each
worker owns its libgomp runtime so K workers scale linearly
(ThreadPoolExecutor would share one OMP team). Stdio frames are 4-byte BE
length + pickle; deliberately avoids importing placer.py to skip the
Xplace/Torch init tree."""

import ctypes
import pickle
import struct
import sys
import time

import numpy as np

_DBL_PTR = ctypes.POINTER(ctypes.c_double)
_INT_PTR = ctypes.POINTER(ctypes.c_int)


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


def _set_argtypes(lib):
    lib.cong_relax_v2.restype = None
    lib.cong_relax_v2.argtypes = [
        _DBL_PTR, _DBL_PTR, _INT_PTR,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        _INT_PTR, _DBL_PTR, _DBL_PTR,
        ctypes.c_int,
        _INT_PTR, _INT_PTR, _INT_PTR,
        _DBL_PTR,
        ctypes.c_int,
        ctypes.c_double,
        _DBL_PTR,                       # wl_extra_weight (NULL = identity)
        ctypes.c_int,
        ctypes.c_double,
        _DBL_PTR, _DBL_PTR,
        _DBL_PTR, _DBL_PTR,
        _DBL_PTR, _DBL_PTR,
        ctypes.c_int,
        ctypes.c_int,
        _INT_PTR, _DBL_PTR,
    ]


def _build_state(args):
    """Pack static topology + scalars into reusable ctypes pointers so
    per-request work is a copyto + cong_relax_v2 call with no fresh
    allocations."""
    sizes = np.ascontiguousarray(args["sizes"], dtype=np.float64)
    movable = np.ascontiguousarray(args["movable"], dtype=np.int32)
    pin_macro = np.ascontiguousarray(args["pin_macro"], dtype=np.int32)
    pin_x = np.ascontiguousarray(args["pin_x"], dtype=np.float64)
    pin_y = np.ascontiguousarray(args["pin_y"], dtype=np.float64)
    net_driver = np.ascontiguousarray(args["net_driver"], dtype=np.int32)
    net_sinks_off = np.ascontiguousarray(args["net_sinks_off"], dtype=np.int32)
    net_sinks_idx = np.ascontiguousarray(args["net_sinks_idx"], dtype=np.int32)
    net_weight = np.ascontiguousarray(args["net_weight"], dtype=np.float64)

    wxw_arr = args.get("wl_extra_weight")
    if wxw_arr is not None:
        wxw_arr = np.ascontiguousarray(wxw_arr, dtype=np.float64)
        wxw_ptr = wxw_arr.ctypes.data_as(_DBL_PTR)
    else:
        wxw_ptr = None

    pos_buf = np.empty((int(args["n"]), 2), dtype=np.float64)

    return {
        # Keep numpy arrays alive - their buffers back the ctypes pointers.
        "_keep": (sizes, movable, pin_macro, pin_x, pin_y,
                  net_driver, net_sinks_off, net_sinks_idx, net_weight,
                  wxw_arr, pos_buf),
        "pos_buf": pos_buf,
        "pos_ptr": pos_buf.ctypes.data_as(_DBL_PTR),
        "sizes_ptr": sizes.ctypes.data_as(_DBL_PTR),
        "movable_ptr": movable.ctypes.data_as(_INT_PTR),
        "pin_macro_ptr": pin_macro.ctypes.data_as(_INT_PTR),
        "pin_x_ptr": pin_x.ctypes.data_as(_DBL_PTR),
        "pin_y_ptr": pin_y.ctypes.data_as(_DBL_PTR),
        "net_driver_ptr": net_driver.ctypes.data_as(_INT_PTR),
        "net_sinks_off_ptr": net_sinks_off.ctypes.data_as(_INT_PTR),
        "net_sinks_idx_ptr": net_sinks_idx.ctypes.data_as(_INT_PTR),
        "net_weight_ptr": net_weight.ctypes.data_as(_DBL_PTR),
        "wxw_ptr": wxw_ptr,
        # Cached scalar args.
        "n": ctypes.c_int(int(args["n"])),
        "nh": ctypes.c_int(int(args["nh"])),
        "cw": ctypes.c_double(float(args["cw"])),
        "ch": ctypes.c_double(float(args["ch"])),
        "gr": ctypes.c_int(int(args["gr"])),
        "gc": ctypes.c_int(int(args["gc"])),
        "smooth_range": ctypes.c_int(int(args["smooth_range"])),
        "hrpm": ctypes.c_double(float(args["hrpm"])),
        "vrpm": ctypes.c_double(float(args["vrpm"])),
        "h_alloc": ctypes.c_double(float(args["h_alloc"])),
        "v_alloc": ctypes.c_double(float(args["v_alloc"])),
        "n_pins": ctypes.c_int(int(pin_macro.shape[0])),
        "n_nets": ctypes.c_int(int(net_driver.shape[0])),
        "net_cnt_for_norm": ctypes.c_double(float(args["net_cnt_for_norm"])),
        "hotspot_sort": ctypes.c_int(int(args.get("hotspot_sort", 0))),
        "out_cong_init": ctypes.c_double(0.0),
        "out_cong_final": ctypes.c_double(0.0),
        "out_proxy_init": ctypes.c_double(0.0),
        "out_proxy_final": ctypes.c_double(0.0),
        "out_wl": ctypes.c_double(0.0),
        "out_den": ctypes.c_double(0.0),
    }


def _run_relax(state, lib, pos_in, n_iter, lr_bins):
    """Run one mini-polish on `pos_in`; returns (pos_out, cong_final,
    proxy_final, wl_final, den_final)."""
    np.copyto(state["pos_buf"], pos_in)
    lib.cong_relax_v2(
        state["pos_ptr"], state["sizes_ptr"], state["movable_ptr"],
        state["n"], state["nh"],
        state["cw"], state["ch"],
        state["gr"], state["gc"],
        state["smooth_range"],
        state["hrpm"], state["vrpm"],
        state["h_alloc"], state["v_alloc"],
        state["pin_macro_ptr"], state["pin_x_ptr"], state["pin_y_ptr"],
        state["n_pins"],
        state["net_driver_ptr"], state["net_sinks_off_ptr"],
        state["net_sinks_idx_ptr"], state["net_weight_ptr"],
        state["n_nets"],
        state["net_cnt_for_norm"],
        state["wxw_ptr"],
        ctypes.c_int(int(n_iter)), ctypes.c_double(float(lr_bins)),
        ctypes.byref(state["out_cong_init"]),
        ctypes.byref(state["out_cong_final"]),
        ctypes.byref(state["out_proxy_init"]),
        ctypes.byref(state["out_proxy_final"]),
        ctypes.byref(state["out_wl"]),
        ctypes.byref(state["out_den"]),
        state["hotspot_sort"],
        ctypes.c_int(0), None, None,    # no extra_phases
    )
    # Copy out - pos_buf is reused next request, pickle would otherwise alias it.
    return (state["pos_buf"].copy(),
            state["out_cong_final"].value,
            state["out_proxy_final"].value,
            state["out_wl"].value,
            state["out_den"].value)


def main():
    in_stream = sys.stdin.buffer
    out_stream = sys.stdout.buffer

    init = _recv_frame(in_stream)
    lib = ctypes.CDLL(init["so_path"])
    _set_argtypes(lib)
    state = _build_state(init["args"])
    _send_frame(out_stream, {"status": "ready"})

    while True:
        try:
            req = _recv_frame(in_stream)
        except EOFError:
            return
        t0 = time.time()
        pos_out, c_final, p_final, wl_f, den_f = _run_relax(
            state, lib, req["pos"], req["n_iter"], req["lr_bins"])
        _send_frame(out_stream, {
            "pos": pos_out,
            "proxy_final": p_final,
            "cong_final": c_final,
            "wl_final": wl_f,
            "den_final": den_f,
            "elapsed": time.time() - t0,
        })


if __name__ == "__main__":
    main()
