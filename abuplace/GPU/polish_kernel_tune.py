"""Per-benchmark kernel-constant autotune for polish_v1; MAX_NETS/PINS/SLOTS
come from state topology, MAX_CELLS uses an empirical heuristic plus a wider
MAX_CELLS_PROBES variant for large-shift batched probes; cached on
state._kernel_bounds."""


def _next_pow2(n):
    """Smallest power-of-2 >= n (n is clamped to >=1)."""
    n = max(1, int(n))
    return 1 << (n - 1).bit_length()


def autotune_kernel_bounds(state):
    """Return cached/derived kernel bounds (MAX_NETS/PINS/SLOTS/CELLS plus a
    wider CELLS_PROBES) from topology stats; pow-2 where Triton requires."""
    cached = getattr(state, "_kernel_bounds", None)
    if cached is not None:
        return cached

    mn_diff = (state.mn_offsets[1:] - state.mn_offsets[:-1]).cpu().numpy()
    max_nets = int(mn_diff.max()) if mn_diff.size > 0 else 0
    sinks_diff = (state.net_sinks_off[1:] - state.net_sinks_off[:-1]).cpu().numpy()
    max_sinks = int(sinks_diff.max()) if sinks_diff.size > 0 else 0
    max_pins_per_net = max_sinks + 1

    # +16 net buffer, pow-2 for Triton; +4 pin buffer; SLOTS pow-2 since
    # tl.arange requires it.
    MAX_NETS_PER_MACRO = max(32, _next_pow2(max_nets + 16))
    MAX_PINS_PER_NET = max(20, max_pins_per_net + 4)
    SLOTS = max(16, _next_pow2(max_pins_per_net + 4))

    # MAX_CELLS is tuned for polish (~5 cells/net typical); MAX_CELLS_PROBES is
    # the wider bound the batched probes need for 2-gcell shifts (~16
    # cells/net).
    avg_cells_per_net = 8
    cells_estimate = max_nets * avg_cells_per_net + 256
    # No probe can emit more than 4*grid_size cells.
    cells_cap = state.gr * state.gc * 4
    # Round up to multiple of EMIT_BLK=32.
    raw = min(cells_estimate, cells_cap)
    MAX_CELLS = max(512, ((raw + 31) // 32) * 32)
    cells_estimate_wide = max_nets * 16 + 1024
    raw_wide = min(cells_estimate_wide, cells_cap)
    MAX_CELLS_PROBES = max(1024, ((raw_wide + 31) // 32) * 32)

    bounds = {
        "MAX_NETS_PER_MACRO": MAX_NETS_PER_MACRO,
        "MAX_PINS_PER_NET": MAX_PINS_PER_NET,
        "SLOTS": SLOTS,
        "MAX_CELLS": MAX_CELLS,
        "MAX_CELLS_PROBES": MAX_CELLS_PROBES,
        "_max_nets_observed": max_nets,
        "_max_pins_observed": max_pins_per_net,
    }
    state._kernel_bounds = bounds
    return bounds
