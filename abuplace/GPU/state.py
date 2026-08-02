"""GPUState: GPU-resident polish state mirroring C cong_relax_v2 State;
canonical "S" only since candidate batching evaluates many poses against it
in one kernel - no per-thread "T" replicas needed; float64/int64 throughout
to match C double/int."""

import numpy as np
import torch


class GPUState:
    """GPU-resident polish state; build via from_pn(...), mutate through
    apply_move/forward helpers, with forward.py reading attrs and route.py
    mutating grids in place."""

    @classmethod
    def from_pn(cls, pn, pin_csr, octx, sizes_np, positions_np,
                movable_np, n_macros, n_hard, *,
                hrpm=None, vrpm=None, h_alloc=None, v_alloc=None,
                smooth_range=None, hpwl_norm=None,
                device=None, dtype=torch.float64):
        if device is None:
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        return cls(
            pn=pn, pin_csr=pin_csr, octx=octx,
            sizes_np=sizes_np, positions_np=positions_np,
            movable_np=movable_np, n_macros=n_macros, n_hard=n_hard,
            hrpm=hrpm, vrpm=vrpm, h_alloc=h_alloc, v_alloc=v_alloc,
            smooth_range=smooth_range, hpwl_norm=hpwl_norm,
            device=device, dtype=dtype,
        )

    def __init__(self, *, pn, pin_csr, octx, sizes_np, positions_np,
                 movable_np, n_macros, n_hard,
                 hrpm, vrpm, h_alloc, v_alloc, smooth_range, hpwl_norm,
                 device, dtype):
        self.device = device
        self.dtype = dtype
        self.idtype = torch.int64

        self.cw = float(octx['cw'])
        self.ch = float(octx['ch'])
        self.gr = int(octx['gr'])
        self.gc = int(octx['gc'])
        self.ng = self.gr * self.gc
        self.gw = self.cw / self.gc
        self.gh = self.ch / self.gr
        self.grid_area = self.gw * self.gh
        # Routing knobs default to pn[] values unless caller overrides.
        self.hrpm = float(hrpm if hrpm is not None else pn['hrpm'])
        self.vrpm = float(vrpm if vrpm is not None else pn['vrpm'])
        self.h_alloc = float(h_alloc if h_alloc is not None else pn['h_alloc'])
        self.v_alloc = float(v_alloc if v_alloc is not None else pn['v_alloc'])
        self.smooth_range = int(smooth_range if smooth_range is not None
                                else pn['smooth_range'])
        self.grid_h_routes = self.gh * self.hrpm
        self.grid_v_routes = self.gw * self.vrpm

        self.n = int(n_macros)
        self.nh = int(n_hard)
        sizes_t = torch.from_numpy(np.asarray(sizes_np, dtype=np.float64)).to(device, dtype)
        self.sizes = sizes_t                               # (n, 2)
        self.movable = torch.from_numpy(
            np.asarray(movable_np, dtype=np.int32)).to(device)  # (n,) int32
        self.pos = torch.from_numpy(
            np.asarray(positions_np, dtype=np.float64)).to(device, dtype).clone()  # (n, 2)

        # Soft-macro index list = movable[i] for i >= nh.
        soft_mask_np = np.zeros(self.n, dtype=bool)
        mov = np.asarray(movable_np, dtype=np.int32).astype(bool)
        soft_mask_np[self.nh:] = mov[self.nh:]
        self.soft_idx_init = torch.from_numpy(
            np.where(soft_mask_np)[0].astype(np.int64)).to(device)

        pin_macro_np = np.asarray(pn['pin_macro'], dtype=np.int64)
        pin_x_np = np.asarray(pn['pin_x'], dtype=np.float64)
        pin_y_np = np.asarray(pn['pin_y'], dtype=np.float64)
        self.pin_macro = torch.from_numpy(pin_macro_np).to(device)         # (np,) int64
        self.pin_x_off = torch.from_numpy(pin_x_np).to(device, dtype)
        self.pin_y_off = torch.from_numpy(pin_y_np).to(device, dtype)
        self.is_port = (self.pin_macro < 0)
        self.np_pins = int(self.pin_macro.shape[0])

        # The net CSR keeps driver and sinks separate to match C route_net,
        # where the driver is the source for the two- and three-pin patterns.
        # pin_csr's combined CSR is used only for the HPWL scatter_reduce.
        self.net_driver = torch.from_numpy(
            np.asarray(pn['net_driver'], dtype=np.int64)).to(device)
        self.net_sinks_off = torch.from_numpy(
            np.asarray(pn['net_sinks_off'], dtype=np.int64)).to(device)
        self.net_sinks_idx = torch.from_numpy(
            np.asarray(pn['net_sinks_idx'], dtype=np.int64)).to(device)
        self.net_weight = torch.from_numpy(
            np.asarray(pn['net_weight'], dtype=np.float64)).to(device, dtype)
        self.nn = int(self.net_weight.shape[0])
        self.net_cnt_for_norm = float(pn['net_cnt_for_norm'])
        if hpwl_norm is not None:
            self.hpwl_norm = float(hpwl_norm)
        else:
            denom = self.net_cnt_for_norm if self.net_cnt_for_norm > 0.0 else 1.0
            self.hpwl_norm = (self.cw + self.ch) * denom

        # Combined pin-net CSR mirroring playground.gpu.GPUContext.pin_to_net,
        # used for the HPWL scatter_reduce.
        net_pin_off_np = np.asarray(pin_csr['net_pin_off'], dtype=np.int64)
        net_pin_idx_np = np.asarray(pin_csr['net_pin_idx'], dtype=np.int64)
        self.net_pin_off = torch.from_numpy(net_pin_off_np).to(device)
        self.net_pin_idx = torch.from_numpy(net_pin_idx_np).to(device)
        counts = (self.net_pin_off[1:] - self.net_pin_off[:-1]).to(self.idtype)
        net_ids_t = torch.arange(self.nn, device=device, dtype=self.idtype)
        self.pin_to_net = torch.repeat_interleave(net_ids_t, counts)
        self.net_has_pins = (counts > 0)

        # macro->pins CSR - used by candidate eval to refresh only one macro's pins.
        valid = pin_macro_np >= 0
        valid_macros = pin_macro_np[valid]
        valid_pins = np.where(valid)[0].astype(np.int64)
        order = np.argsort(valid_macros, kind='stable')
        sorted_macros = valid_macros[order]
        sorted_pins = valid_pins[order]
        mp_offsets_np = np.zeros(self.n + 1, dtype=np.int64)
        np.add.at(mp_offsets_np[1:], sorted_macros, 1)
        np.cumsum(mp_offsets_np, out=mp_offsets_np)
        self.mp_offsets = torch.from_numpy(mp_offsets_np).to(device)
        self.mp_pin_ids = torch.from_numpy(sorted_pins).to(device)

        # macro->nets CSR built lazily by build_mn_csr.
        self.mn_offsets = None  # int64 (n+1,)
        self.mn_net_ids = None  # int64

        ng = self.ng
        self.H_net = torch.zeros(ng, device=device, dtype=dtype)
        self.V_net = torch.zeros(ng, device=device, dtype=dtype)
        self.H_macro = torch.zeros(ng, device=device, dtype=dtype)
        self.V_macro = torch.zeros(ng, device=device, dtype=dtype)
        self.H_final = torch.zeros(ng, device=device, dtype=dtype)
        self.V_final = torch.zeros(ng, device=device, dtype=dtype)
        self.grid_occupied = torch.zeros(ng, device=device, dtype=dtype)

        self.pin_abs_x = torch.zeros(self.np_pins, device=device, dtype=dtype)
        self.pin_abs_y = torch.zeros(self.np_pins, device=device, dtype=dtype)
        self.pin_row = torch.zeros(self.np_pins, device=device, dtype=self.idtype)
        self.pin_col = torch.zeros(self.np_pins, device=device, dtype=self.idtype)

        self.net_hpwl = torch.zeros(self.nn, device=device, dtype=dtype)
        self.net_xmin = torch.zeros(self.nn, device=device, dtype=dtype)
        self.net_xmax = torch.zeros(self.nn, device=device, dtype=dtype)
        self.net_ymin = torch.zeros(self.nn, device=device, dtype=dtype)
        self.net_ymax = torch.zeros(self.nn, device=device, dtype=dtype)
        self.total_hpwl = torch.zeros((), device=device, dtype=dtype)

        self.cong_cost = torch.zeros((), device=device, dtype=dtype)
        self.density_cost = torch.zeros((), device=device, dtype=dtype)
        self.wl_cost = torch.zeros((), device=device, dtype=dtype)
        self.full_cost = torch.zeros((), device=device, dtype=dtype)

    def refresh_all_pin_cache(self):
        """Recompute pin_abs_x/y and pin_row/col for ALL pins from current
        self.pos."""
        safe_macro = torch.where(self.is_port,
                                 torch.zeros_like(self.pin_macro),
                                 self.pin_macro)
        macro_x = self.pos[safe_macro, 0]
        macro_y = self.pos[safe_macro, 1]
        self.pin_abs_x = torch.where(self.is_port, self.pin_x_off,
                                     macro_x + self.pin_x_off)
        self.pin_abs_y = torch.where(self.is_port, self.pin_y_off,
                                     macro_y + self.pin_y_off)
        col_f = torch.floor(self.pin_abs_x / self.gw).to(self.idtype)
        row_f = torch.floor(self.pin_abs_y / self.gh).to(self.idtype)
        self.pin_col = torch.clamp(col_f, 0, self.gc - 1)
        self.pin_row = torch.clamp(row_f, 0, self.gr - 1)

    def build_mn_csr(self):
        """Build macro->nets CSR on CPU (dedup-per-net is awkward on GPU);
        topology-only so this runs once at init."""
        pin_macro_np = self.pin_macro.cpu().numpy()
        net_driver_np = self.net_driver.cpu().numpy()
        net_sinks_off_np = self.net_sinks_off.cpu().numpy()
        net_sinks_idx_np = self.net_sinks_idx.cpu().numpy()

        # Pass 1: build per-net unique macro set and tally per-macro counts.
        counts = np.zeros(self.n, dtype=np.int64)
        per_net_macros = []
        for ni in range(self.nn):
            seen = set()
            d = int(net_driver_np[ni])
            dm = int(pin_macro_np[d])
            if dm >= 0:
                seen.add(dm)
            for k in range(int(net_sinks_off_np[ni]), int(net_sinks_off_np[ni + 1])):
                p = int(net_sinks_idx_np[k])
                pm = int(pin_macro_np[p])
                if pm >= 0:
                    seen.add(pm)
            per_net_macros.append(seen)
            for mm in seen:
                counts[mm] += 1

        offsets = np.zeros(self.n + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(counts)
        total = int(offsets[-1])
        net_ids = np.zeros(max(total, 1), dtype=np.int64)
        cursor = np.zeros(self.n, dtype=np.int64)
        for ni in range(self.nn):
            for mm in per_net_macros[ni]:
                net_ids[offsets[mm] + cursor[mm]] = ni
                cursor[mm] += 1

        self.mn_offsets = torch.from_numpy(offsets).to(self.device)
        self.mn_net_ids = torch.from_numpy(net_ids).to(self.device)

    def snapshot(self):
        """Return cloned tensors of all position-dependent state plus scalar
        costs; topology-only buffers (sizes, pin_macro, mn_*, mp_*) are
        intentionally skipped."""
        return {
            'pos': self.pos.clone(),
            'H_net': self.H_net.clone(),
            'V_net': self.V_net.clone(),
            'H_macro': self.H_macro.clone(),
            'V_macro': self.V_macro.clone(),
            'H_final': self.H_final.clone(),
            'V_final': self.V_final.clone(),
            'grid_occupied': self.grid_occupied.clone(),
            'net_hpwl': self.net_hpwl.clone(),
            'net_xmin': self.net_xmin.clone(),
            'net_xmax': self.net_xmax.clone(),
            'net_ymin': self.net_ymin.clone(),
            'net_ymax': self.net_ymax.clone(),
            'total_hpwl': self.total_hpwl.clone(),
            'cong_cost': self.cong_cost.clone(),
            'density_cost': self.density_cost.clone(),
            'wl_cost': self.wl_cost.clone(),
            'full_cost': self.full_cost.clone(),
        }

    def restore(self, snap):
        """In-place restore from a snapshot dict (copy_ into self.*)."""
        self.pos.copy_(snap['pos'])
        self.H_net.copy_(snap['H_net'])
        self.V_net.copy_(snap['V_net'])
        self.H_macro.copy_(snap['H_macro'])
        self.V_macro.copy_(snap['V_macro'])
        self.H_final.copy_(snap['H_final'])
        self.V_final.copy_(snap['V_final'])
        self.grid_occupied.copy_(snap['grid_occupied'])
        self.net_hpwl.copy_(snap['net_hpwl'])
        self.net_xmin.copy_(snap['net_xmin'])
        self.net_xmax.copy_(snap['net_xmax'])
        self.net_ymin.copy_(snap['net_ymin'])
        self.net_ymax.copy_(snap['net_ymax'])
        self.total_hpwl.copy_(snap['total_hpwl'])
        self.cong_cost.copy_(snap['cong_cost'])
        self.density_cost.copy_(snap['density_cost'])
        self.wl_cost.copy_(snap['wl_cost'])
        self.full_cost.copy_(snap['full_cost'])
        # Pin cache is cheap to rebuild after restoring positions.
        self.refresh_all_pin_cache()
