"""init_state: bootstrap GPU state from current positions; routes hard macros,
routes all nets, builds macro->nets CSR, deposits density, computes per-net
HPWL, canonicalizes H_final/V_final, then refreshes proxy components."""

from .density_batch import batched_density_init
from .forward import compute_all_net_hpwl, update_proxy_components
from .route import route_macro, compute_smooth_into_final
from abuplace.kernels.init_route import init_route_all_nets


def init_state(state):
    """Bootstrap GPUState grids/costs from current positions; reuses cached
    hard-macro H/V routing (hard macros never move) so subsequent calls skip
    the ~290-macro Python loop."""
    state.refresh_all_pin_cache()

    # Zero net/density grids only - H_macro/V_macro restored from cache below.
    state.H_net.zero_()
    state.V_net.zero_()
    state.H_final.zero_()
    state.V_final.zero_()
    state.grid_occupied.zero_()

    # Hard macros are static: reuse cached contributions when available.
    if getattr(state, "_hard_H_macro_cache", None) is not None:
        state.H_macro.copy_(state._hard_H_macro_cache)
        state.V_macro.copy_(state._hard_V_macro_cache)
    else:
        state.H_macro.zero_()
        state.V_macro.zero_()
        for i in range(state.nh):
            route_macro(state, i)
        state._hard_H_macro_cache = state.H_macro.clone()
        state._hard_V_macro_cache = state.V_macro.clone()

    # Batched Triton kernel replaces a ~7700-call Python net-routing loop.
    init_route_all_nets(state)

    # Topology-only CSR - build once and cache on state.
    if state.mn_offsets is None:
        state.build_mn_csr()

    # Batched scatter avoids per-macro .item() syncs.
    batched_density_init(state)

    compute_all_net_hpwl(state)

    # Fresh smoothing canonicalizes H_final/V_final and erases incremental FP drift.
    compute_smooth_into_final(state)

    update_proxy_components(state)

    return state
