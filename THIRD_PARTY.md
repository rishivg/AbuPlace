# Third-party components

This project is released under the Apache License 2.0 (see [`LICENSE.md`](LICENSE.md)).
It bundles or depends on the third-party components below, each under its own
license. Those licenses govern the corresponding files, not Apache 2.0.

---

## Xplace -  vendored (modified)

| | |
|---|---|
| Path | [`abuplace/Xplace/`](abuplace/Xplace/) |
| Upstream | https://github.com/cuhk-eda/Xplace |
| License | BSD 3-Clause -  [`abuplace/Xplace/LICENSE`](abuplace/Xplace/LICENSE) |
| Used for | Analytical (eplace/Nesterov electrostatic) global placement, which seeds this placer's refinement cascade |

Xplace is vendored rather than referenced as a submodule because the placer
drives its internals directly.

Modifications. [`abuplace/Xplace/src/calculator.py`](abuplace/Xplace/src/calculator.py)
adds three RUDY congestion losses and a hook that injects them into the Nesterov
gradient; the file carries a notice at the top saying so. Every other file under
`abuplace/Xplace/` is unmodified upstream Xplace.

The prebuilt CUDA extensions (`abuplace/Xplace/cpp_to_py/cpybin/*.so`) are not
tracked in git -  see [`SETUP.md`](SETUP.md) for building them.

## FLUTE -  vendored (unmodified)

| | |
|---|---|
| Path | [`abuplace/Xplace/thirdparty/flute/`](abuplace/Xplace/thirdparty/flute/) |
| Upstream | Dr. Chris C. N. Chu, Iowa State University |
| License | Attribution Assurance License (BSD-derived) -  [`license.txt`](abuplace/Xplace/thirdparty/flute/license.txt) |
| Used for | Rectilinear Steiner minimal tree wirelength inside Xplace |

Only the lookup tables (`POWV9.dat`, `POST9.dat`) are vendored here; they arrive
as part of the Xplace tree.

> Note for commercial users. The FLUTE license is permissive and does not
> prohibit commercial use, but clause 2 states that "users who intend to use the
> Code for commercial purposes will notify Author prior to such commercial use."
> If you plan to use this project commercially, honor that notification.

## TILOS MacroPlacement -  submodule (unmodified)

| | |
|---|---|
| Path | `external/MacroPlacement/` (git submodule, not vendored) |
| Upstream | https://github.com/TILOS-AI-Institute/MacroPlacement |
| Pinned to | https://github.com/partcleda/MacroPlacement (branch `fix-scientific-notation-parsing`) |
| License | See `external/MacroPlacement/LICENSE` after checkout |
| Used for | The proxy-cost evaluator (`plc_client_os.PlacementCost`) and the ICCAD04 benchmark testcases |

Pinned to a specific commit on the `fix-scientific-notation-parsing` branch of
Partcl's fork -  the same source the competition's own repository used. That
branch carries a single three-line regex fix to `plc_client_os.py` so the
netlist parser accepts values in scientific notation (`1.42109e-16`). Without
it, `PlacementCost` raises `ValueError: could not convert string to float` and
15 of the 17 IBM benchmarks fail to load. Do not repoint this at upstream
TILOS.

Apart from that parser fix the evaluator is used as-is and is never
modified -  the competition this placer was written for requires that.

## Benchmarks

`benchmarks/processed/public/*.pt` are pre-processed tensor caches derived from
the TILOS MacroPlacement testcases (ICCAD04 `ibm*` and the NG45/ASAP7 designs).
They are a format conversion of upstream data and carry upstream's terms; the
conversion scripts are in [`scripts/`](scripts/).

## Python dependencies

PyTorch, Triton, NumPy and the rest are ordinary PyPI dependencies declared in
[`pyproject.toml`](pyproject.toml) -  not vendored. Their licenses apply as
published.
