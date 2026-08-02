"""Pure-scoring batched probe evaluator for non-polish operators
(hardmove/softmove/pair-swap/window-reorder); takes (macro_id, cand_pos)
probes and returns proxy deltas without mutating state by reusing the
eval_is_batched delta path; caller must ctx.refresh() after committing
externally."""

import os

import numpy as np
import torch

from .batched_candidates import (
    batched_compute_cong, batched_compute_density,
    batched_smooth_hnet_to_hfinal, batched_smooth_vnet_to_vfinal,
)
from .eval_is_batched import _polish_emit_multi
from .fast_route import CPUPinCache
from .polish_kernel_tune import autotune_kernel_bounds


def _chunk_target_bytes():
    """Per-chunk GPU working-set budget (default 1.5 GiB, override via
    XP_PROBE_CHUNK_BYTES); calibrated for 8GB-class GPUs sharing memory with
    PyTorch + Xplace caches."""
    return int(os.environ.get("XP_PROBE_CHUNK_BYTES", str(1500 * 1024 * 1024)))


class BatchedProbeContext:
    """Reusable scoring context: builds CPUPinCache, pin_abs GPU mirror,
    autotuned kernel bounds and density bin grids once per state, then
    evaluates many probes via .evaluate()."""

    def __init__(self, state):
        self.state = state
        self.cpu = CPUPinCache(state)
        # state.pin_abs_x already lives on GPU after init_state; clone for safety.
        if hasattr(state, "pin_abs_x") and state.pin_abs_x is not None:
            self.pin_abs_x = state.pin_abs_x.clone().to(torch.float64)
            self.pin_abs_y = state.pin_abs_y.clone().to(torch.float64)
        else:
            self.pin_abs_x = torch.from_numpy(np.ascontiguousarray(
                self.cpu.pin_abs_x, dtype=np.float64)).to(state.device)
            self.pin_abs_y = torch.from_numpy(np.ascontiguousarray(
                self.cpu.pin_abs_y, dtype=np.float64)).to(state.device)
        autotune_kernel_bounds(state)
        device = state.device
        cells_r = torch.arange(state.gr, device=device, dtype=torch.float64)
        cells_c = torch.arange(state.gc, device=device, dtype=torch.float64)
        self._bin_y_lo = (cells_r * state.gh).to(torch.float64)
        self._bin_y_hi = (self._bin_y_lo + state.gh)
        self._bin_x_lo = (cells_c * state.gw).to(torch.float64)
        self._bin_x_hi = (self._bin_x_lo + state.gw)

    def refresh(self):
        """Re-clone pin_abs_x/y from state after external mutation so
        subsequent evaluate() calls see fresh positions."""
        if hasattr(self.state, "pin_abs_x") and self.state.pin_abs_x is not None:
            self.pin_abs_x = self.state.pin_abs_x.clone().to(torch.float64)
            self.pin_abs_y = self.state.pin_abs_y.clone().to(torch.float64)

    def evaluate(self, macro_ids, positions):
        """Score B probes (macro_ids[i] moved to positions[i]); auto-chunks
        B*ng to fit GPU memory; returns float64 (B,) where deltas[i] =
        proxy(after move) - state.full_cost (negative = improving)."""
        if isinstance(macro_ids, np.ndarray):
            macros_np = macro_ids if macro_ids.dtype == np.int32 \
                else macro_ids.astype(np.int32)
        else:
            macros_np = np.asarray(macro_ids, dtype=np.int32)
        B = len(macros_np)
        if B == 0:
            return np.zeros(0, dtype=np.float64)
        if isinstance(positions, np.ndarray):
            pos_np = positions if positions.dtype == np.float64 \
                else positions.astype(np.float64)
        else:
            pos_np = np.asarray(positions, dtype=np.float64)
        if pos_np.ndim == 1:
            pos_np = pos_np.reshape(-1, 2)

        # Per-probe footprint: 7x (B,ng) float64 buffers plus 2x (B,MAX_CELLS)
        # int32/float32 emit scratch. Chunk so the total stays under
        # target_bytes (default 1.5GB), which is what keeps the largest designs
        # off OOM.
        ng = self.state.ng
        max_cells = self.state._kernel_bounds["MAX_CELLS_PROBES"]
        bytes_per_probe = ng * 8 * 7 + max_cells * 4 * 4
        target_bytes = int(_chunk_target_bytes())
        chunk_size = max(64, target_bytes // max(1, bytes_per_probe))
        if B <= chunk_size:
            return self._evaluate_chunk(macros_np, pos_np)

        out = np.empty(B, dtype=np.float64)
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            out[start:end] = self._evaluate_chunk(
                macros_np[start:end], pos_np[start:end])
        return out

    def _evaluate_chunk(self, macros_np, pos_np):
        """Score a single chunk that fits in GPU memory and return per-probe
        (new_full - cur_full) deltas."""
        state = self.state
        cpu = self.cpu
        device = state.device
        dtype = state.dtype
        gr = state.gr
        gc = state.gc
        sr = state.smooth_range

        B = len(macros_np)
        nx_np = np.ascontiguousarray(pos_np[:, 0])
        ny_np = np.ascontiguousarray(pos_np[:, 1])

        # Dedup macros so each unique-orig is emitted only once.
        unique_macros, probe_to_orig = np.unique(macros_np, return_inverse=True)
        M = len(unique_macros)
        orig_pos_np = cpu.pos[unique_macros]  # (M, 2)
        orig_macros_np = unique_macros.astype(np.int32)

        # Wider MAX_CELLS_PROBES than polish bound - large-displacement cands
        # emit more cells.
        combined_macros = np.concatenate([orig_macros_np, macros_np])
        combined_pos = np.concatenate([orig_pos_np, pos_np], axis=0)
        bnd = state._kernel_bounds
        combined_H, combined_V = _polish_emit_multi(
            state, cpu, combined_macros, combined_pos,
            MAX_CELLS=bnd["MAX_CELLS_PROBES"],
            MAX_NETS_PER_MACRO=bnd["MAX_NETS_PER_MACRO"],
            MAX_PINS_PER_NET=bnd["MAX_PINS_PER_NET"],
            SLOTS=bnd["SLOTS"])
        orig_H_net = combined_H[:M]
        orig_V_net = combined_V[:M]
        cand_H_net = combined_H[M:]
        cand_V_net = combined_V[M:]

        probe_to_orig_t = torch.from_numpy(
            probe_to_orig.astype(np.int64)).to(device)
        delta_H_net = cand_H_net - orig_H_net[probe_to_orig_t]
        delta_V_net = cand_V_net - orig_V_net[probe_to_orig_t]

        sizes_orig_t = state.sizes[
            torch.from_numpy(unique_macros.astype(np.int64)).to(device)]
        sizes_cand_t = state.sizes[
            torch.from_numpy(macros_np.astype(np.int64)).to(device)]
        hw_orig = sizes_orig_t[:, 0] * 0.5
        hh_orig = sizes_orig_t[:, 1] * 0.5
        hw_cand = sizes_cand_t[:, 0] * 0.5
        hh_cand = sizes_cand_t[:, 1] * 0.5
        orig_x_t = torch.from_numpy(
            np.ascontiguousarray(orig_pos_np[:, 0])).to(device)
        orig_y_t = torch.from_numpy(
            np.ascontiguousarray(orig_pos_np[:, 1])).to(device)
        cand_x_t = torch.from_numpy(nx_np).to(device)
        cand_y_t = torch.from_numpy(ny_np).to(device)

        new_density_pc = self._density_grid(cand_x_t, cand_y_t, hw_cand, hh_cand)
        orig_density_pm = self._density_grid(orig_x_t, orig_y_t, hw_orig, hh_orig)
        delta_density = new_density_pc - orig_density_pm[probe_to_orig_t]

        from abuplace.kernels.hpwl_delta import hpwl_subset_at
        hpwl_combined = hpwl_subset_at(
            state, cpu, combined_macros, combined_pos,
            pin_abs_x_gpu=self.pin_abs_x, pin_abs_y_gpu=self.pin_abs_y)
        orig_hpwl_pm = hpwl_combined[:M]
        cand_hpwl_pb = hpwl_combined[M:]
        delta_hpwl_t = cand_hpwl_pb - orig_hpwl_pm[probe_to_orig_t]
        delta_hpwl_per = delta_hpwl_t.cpu().numpy()

        delta_H_final = batched_smooth_hnet_to_hfinal(delta_H_net, sr, gr, gc)
        delta_V_final = batched_smooth_vnet_to_vfinal(delta_V_net, sr, gr, gc)
        H_final_pc = state.H_final.unsqueeze(0) + delta_H_final
        V_final_pc = state.V_final.unsqueeze(0) + delta_V_final
        grid_occ_pc = state.grid_occupied.unsqueeze(0) + delta_density
        cong_pc = batched_compute_cong(V_final_pc, H_final_pc)
        den_pc = batched_compute_density(grid_occ_pc, state.grid_area)

        base_total_hpwl = float(state.total_hpwl.item())
        new_total_hpwl_pc = base_total_hpwl + delta_hpwl_per
        if state.hpwl_norm > 0.0:
            new_wl_pc = new_total_hpwl_pc / state.hpwl_norm
        else:
            new_wl_pc = np.zeros(B, dtype=np.float64)
        new_wl_t = torch.from_numpy(new_wl_pc).to(device, dtype)
        new_full_pc = new_wl_t + 0.5 * den_pc + 0.5 * cong_pc

        cur_full = float(state.full_cost.item())
        deltas = (new_full_pc - cur_full).cpu().numpy().astype(np.float64)
        return deltas

    def _density_grid(self, px, py, hw, hh):
        """Per-probe bbox-cell overlap area as (B, ng) float64."""
        state = self.state
        ng = state.ng
        ly = py - hh; hy = py + hh
        lx = px - hw; hx = px + hw
        oy_lo = torch.maximum(self._bin_y_lo[None, :], ly[:, None])
        oy_hi = torch.minimum(self._bin_y_hi[None, :], hy[:, None])
        oy = (oy_hi - oy_lo).clamp(min=0.0)
        ox_lo = torch.maximum(self._bin_x_lo[None, :], lx[:, None])
        ox_hi = torch.minimum(self._bin_x_hi[None, :], hx[:, None])
        ox = (ox_hi - ox_lo).clamp(min=0.0)
        return (oy[:, :, None] * ox[:, None, :]).reshape(-1, ng).to(state.dtype)
