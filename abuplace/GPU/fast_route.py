"""CPU-side mirror of GPUState topology + dynamic grids; static CSR/topology
cached on the GPUState so only positions, pin row/col/abs, and the
H/V/density grids are re-pulled per construction."""

import numpy as np


class CPUPinCache:
    """CPU-side mirror of pin row/col/abs and per-net bboxes; kept in sync with
    GPUState via explicit refresh calls."""

    def __init__(self, state):
        # Dynamic pulls: positions/grids/pin_abs - refreshed each construction.
        self.pin_row = state.pin_row.cpu().numpy().astype(np.int64).copy()
        self.pin_col = state.pin_col.cpu().numpy().astype(np.int64).copy()
        self.pin_abs_x = state.pin_abs_x.cpu().numpy().astype(np.float64).copy()
        self.pin_abs_y = state.pin_abs_y.cpu().numpy().astype(np.float64).copy()
        self.pos = state.pos.cpu().numpy().astype(np.float64).copy()
        self.H_final = state.H_final.cpu().numpy().astype(np.float64).copy()
        self.V_final = state.V_final.cpu().numpy().astype(np.float64).copy()
        self.grid_occupied = state.grid_occupied.cpu().numpy().astype(np.float64).copy()

        # The topology is static after build_mn_csr, so cache it on state
        # rather than pulling ~1MB+ per polish_v1 call.
        if getattr(state, "_cpu_static_cache", None) is None:
            if state.mn_offsets is None:
                state.build_mn_csr()
            cache = {
                "sizes": state.sizes.cpu().numpy().astype(np.float64).copy(),
                "pin_macro": state.pin_macro.cpu().numpy().astype(np.int64).copy(),
                "pin_x_off": state.pin_x_off.cpu().numpy().astype(np.float64).copy(),
                "pin_y_off": state.pin_y_off.cpu().numpy().astype(np.float64).copy(),
                "is_port": state.is_port.cpu().numpy().copy(),
                "net_driver": state.net_driver.cpu().numpy().astype(np.int64).copy(),
                "net_sinks_off": state.net_sinks_off.cpu().numpy().astype(np.int64).copy(),
                "net_sinks_idx": state.net_sinks_idx.cpu().numpy().astype(np.int64).copy(),
                "net_weight": state.net_weight.cpu().numpy().astype(np.float64).copy(),
                "mp_offsets": state.mp_offsets.cpu().numpy().astype(np.int64).copy(),
                "mp_pin_ids": state.mp_pin_ids.cpu().numpy().astype(np.int64).copy(),
                "mn_offsets": state.mn_offsets.cpu().numpy().astype(np.int64).copy(),
                "mn_net_ids": state.mn_net_ids.cpu().numpy().astype(np.int64).copy(),
                "net_pin_off": state.net_pin_off.cpu().numpy().astype(np.int64).copy(),
                "net_pin_idx": state.net_pin_idx.cpu().numpy().astype(np.int64).copy(),
            }
            state._cpu_static_cache = cache
        cache = state._cpu_static_cache
        self.sizes = cache["sizes"]
        self.pin_macro = cache["pin_macro"]
        self.pin_x_off = cache["pin_x_off"]
        self.pin_y_off = cache["pin_y_off"]
        self.is_port = cache["is_port"]
        self.net_driver = cache["net_driver"]
        self.net_sinks_off = cache["net_sinks_off"]
        self.net_sinks_idx = cache["net_sinks_idx"]
        self.net_weight = cache["net_weight"]
        self.mp_offsets = cache["mp_offsets"]
        self.mp_pin_ids = cache["mp_pin_ids"]
        self.mn_offsets = cache["mn_offsets"]
        self.mn_net_ids = cache["mn_net_ids"]
        self.net_pin_off = cache["net_pin_off"]
        self.net_pin_idx = cache["net_pin_idx"]

        self.gw = state.gw
        self.gh = state.gh
        self.gr = state.gr
        self.gc = state.gc

    def sync_grids_from_gpu(self, state):
        """Re-pull H_final/V_final/grid_occupied to the CPU shadow after a
        state mutation so compute_dirs neighborhood reads stay fresh without
        per-call GPU pulls."""
        self.H_final = state.H_final.cpu().numpy().astype(np.float64)
        self.V_final = state.V_final.cpu().numpy().astype(np.float64)
        self.grid_occupied = state.grid_occupied.cpu().numpy().astype(np.float64)
