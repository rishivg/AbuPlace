# AbuPlace

A GPU macro placer that optimizes the competition's proxy cost directly rather than a smooth
surrogate.

[![CI](https://github.com/rishivg/AbuPlace/actions/workflows/ci.yml/badge.svg)](https://github.com/rishivg/AbuPlace/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE.md)

Macro placement puts large fixed-size blocks (SRAMs, IP, analog macros) on a chip canvas.
The search space is discrete and huge, overlap is forbidden, and the three cost terms pull
against each other: wirelength wants macros clustered, density wants them spread, congestion
wants routing channels left open.

```
Proxy Cost = 1.0 * Wirelength + 0.5 * Density + 0.5 * Congestion
```

Written for the [Partcl/HRT Macro Placement Challenge](CHALLENGE.md), where that cost is the
ranking metric. ABU is the area-based utilization behind the congestion term: the mean of the
largest 5% of the smoothed horizontal and vertical routing-demand maps.

## How it works

Most placers optimize a smooth surrogate (analytical density plus weighted HPWL) and only
measure the true cost at the end. Any gap between surrogate and truth is search effort spent
in the wrong direction.

AbuPlace ports the evaluator itself into C bit-for-bit (`get_routing()`,
`__smooth_routing_cong()`, `abu()`) and optimizes that. Two things fall out of it.

Greedy accept/reject stops being a gamble. Each operator scores its candidate exactly and
accepts only when the real proxy drops, so there is no surrogate left to be wrong about.

Scoring can be incremental. Owning the cost code means you can expose `apply`/`revert`/
`commit` over the live routing and density maps and touch only the bins and nets a move
changed. A single-macro probe costs 0.5-2 ms instead of a 5-15 ms full recompute, about 30x
cheaper, which is what makes tens of thousands of exact-scored candidate moves fit in the
runtime budget.

The catch is drift. After many commits the incremental score wanders away from a fresh
recompute, so every stage rebuilds periodically, and before keeping a result it re-scores
from scratch and reverts itself if the honest number doesn't actually beat where it started.

## Pipeline

The benchmark goes in as macros, nets and a canvas, and a legal `[N, 2]` array of macro
centers comes out. In between: Xplace global placement, a tuning probe that reads the post-GP
congestion and gates nearly every knob below it, then refine, polish, escape and operate, and
finally the safety net.

Coarse to fine. Global placement gets the structure right, refine settles it into a good
basin, polish and basin-jump escape local minima, and the operator stack does the fine
cleanup. Every stage calls the exact-proxy oracle to decide whether a move is real:
`congestion.c` for the exact proxy and incremental scoring, `refine.c` for Adam on a smooth
surrogate, `legalize.c` for spiral legalization, and `GPU/forward.py` for the same proxy on
the GPU.

Two things that leaves out.

The whole pipeline runs three times. Xplace's RUDY congestion loss improves global placement
on some designs, but the polish stage, tuned against the default basin, can invert that gain
into a final regression on others, and which designs invert is not predictable. So the
pipeline runs under each setting and the lowest-proxy legal result ships. The validated
baseline runs first and stays a candidate, so this cannot regress.

A fair amount of the code exists only so the placer never fails outright. A crash, one
overlapping macro, an out-of-bounds coordinate or a NaN is a disqualification, which is worse
than a mediocre legal result. Hence the per-trajectory crash guards, the legality gate on
selection, the regression guard, the unconditional legality repair, the bounds clamp and the
NaN sanitizer. All of them are no-ops on a healthy run, where the shipped placement is
byte-identical to what the cascade produced.

[`abuplace/TECH_DEEPDIVE.md`](abuplace/TECH_DEEPDIVE.md) covers each stage and operator.

## Results

Proxy cost, lower is better. Baselines from [An Updated Assessment of Reinforcement Learning
for Macro Placement](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11300304).

| Benchmark | SA | RePlAce | AbuPlace |
|---|---:|---:|---:|
| ibm01 | 1.3166 | 0.9976 | **0.7633** |

The full 17-benchmark IBM sweep runs with `uv run evaluate abuplace/placer.py --all`.

Seeds are pinned, CUDA runs in deterministic mode, and the basin-jump loops use iteration
caps rather than wall-clock deadlines. Several operators still stop on a time budget, so the
last few digits move with host load: two ibm01 runs here gave 0.763301 and 0.763290.

## Quick start

```bash
git clone --recurse-submodules https://github.com/rishivg/AbuPlace.git
cd AbuPlace

# if you already cloned without --recurse-submodules:
git submodule update --init external/MacroPlacement

uv sync --extra abuplace

# one-time: fetch the prebuilt Xplace CUDA extensions (checksum-verified)
scripts/fetch_xplace_binaries.sh

uv run evaluate abuplace/placer.py -b ibm01
```

The three C extensions build from source on first run with `-O3 -march=native`.

There is a Docker path in [`abuplace/README.md`](abuplace/README.md) that pins the Python and
torch versions the prebuilt CUDA extensions need. Full setup is in [`SETUP.md`](SETUP.md).

### Requirements

- Linux, NVIDIA GPU with at least 8 GB VRAM, driver matching the torch cu12 wheels
- Python 3.12 exactly; the prebuilt Xplace extensions are cpython-3.12 ABI
- [`uv`](https://docs.astral.sh/uv/), and about 40 GB free disk for the submodule

## Layout

| Path | What |
|---|---|
| [`abuplace/placer.py`](abuplace/placer.py) | `XplacePlacer`: orchestration, operator stack, `place()` |
| [`abuplace/extensions/`](abuplace/extensions/) | C: exact proxy and `CongState`, Adam refiner, spiral legalizer |
| [`abuplace/GPU/`](abuplace/GPU/) | GPU proxy model, batched polish, basin-jump, device state |
| [`abuplace/kernels/`](abuplace/kernels/) | Triton kernels: HPWL delta, routing emit, scatter |
| [`abuplace/TUNING.md`](abuplace/TUNING.md) | Every environment knob, with defaults |
| [`abuplace/Xplace/`](abuplace/Xplace/) | Vendored Xplace global placer (BSD 3-Clause, modified) |
| [`macro_place/`](macro_place/) | Competition harness: benchmark loading and scoring |
| [`scripts/`](scripts/) | Tier-2 OpenROAD tooling, benchmark conversion, binary fetch |

## License

Apache 2.0, see [`LICENSE.md`](LICENSE.md).

Bundles third-party components under their own licenses, including Xplace (BSD 3-Clause) and
FLUTE (Attribution Assurance License). See [`THIRD_PARTY.md`](THIRD_PARTY.md), which also
carries a notification note relevant to commercial users.
