# AbuPlace

GPU macro placer built on Xplace global placement. Entry point is [placer.py](placer.py).

## Quick start (Docker)

The image pins the Python (3.12) and torch (2.10.x) versions the prebuilt binaries need, so
nothing on the host has to match beyond an NVIDIA driver.

```bash
# from the repo root
git submodule update --init external/MacroPlacement   # one-time, see "Benchmark data"

abuplace/build.sh            # build the image, ~5-10 min the first time

abuplace/run.sh -b ibm01     # one benchmark
abuplace/run.sh --all        # all 17 IBM benchmarks
abuplace/run.sh --ng45       # NG45 commercial designs
```

Without the wrappers:

```bash
docker build --network=host -f abuplace/Dockerfile -t abuplace:latest .

docker run --rm --gpus all --network none --shm-size 8g \
    -v "$(pwd)/external:/work/external:ro" \
    -v "$(pwd)/benchmarks:/work/benchmarks:ro" \
    abuplace:latest -b ibm01
```

## Prerequisites

Docker path:

- NVIDIA driver 560 or newer (CUDA 12.6 runtime), check with `nvidia-smi`
- NVIDIA Container Toolkit, so `docker run --gpus all` works
- Docker 23.x or newer, for the BuildKit support that honors `Dockerfile.dockerignore`
- CUDA-capable GPU with at least 8 GB VRAM

Host venv path, faster to iterate on but more setup:

- Python 3.12 exactly; the prebuilt Xplace `.so`s in
  [Xplace/cpp_to_py/cpybin/](Xplace/cpp_to_py/cpybin/) are cpython-3.12 ABI
- [`uv`](https://docs.astral.sh/uv/)
- NVIDIA driver compatible with the torch 2.10 cu12 wheels

Both paths need the benchmark data below.

## Benchmark data

Two inputs must exist at the repo root before running:

```bash
# 1. TILOS evaluator and ICCAD04 testcases (git submodule)
git submodule update --init external/MacroPlacement

# 2. pre-processed .pt benchmark cache, see the top-level SETUP.md
#    (populates benchmarks/processed/)
```

`run.sh` checks for both and points back here if either is missing. The container mounts them
read-only at `/work/external` and `/work/benchmarks`.

## Xplace CUDA extensions

Required, one-time. The Xplace global placer runs through prebuilt CUDA extensions. They are
not tracked in git, being 16 MB of opaque binaries pinned to cpython-3.12 and the CUDA 12
ABI, so they ship as a checksum-verified release asset:

```bash
scripts/fetch_xplace_binaries.sh
ls abuplace/Xplace/cpp_to_py/cpybin/*.so | wc -l   # expect 14
```

If the placer fails at import time this is almost always why.

Two alternatives. You can build from upstream: clone
[Xplace](https://github.com/cuhk-eda/Xplace) at a matching commit, build it (needs `nvcc` and
`cmake`), and copy `cpp_to_py/cpybin/*.so` into `abuplace/Xplace/cpp_to_py/cpybin/`. That is
the only option if your Python or CUDA version differs. Or use Docker, but the image build
expects the binaries in its context, so run the fetch script on the host first.

The three `extensions/*.so` are a separate matter and are supposed to be absent. They compile
from their `.c` sources on first run.

## Host venv path

For iterating on a Python 3.12 box without rebuilding the image:

```bash
uv sync --extra abuplace
uv run evaluate abuplace/placer.py -b ibm01
```

The `abuplace` extras group, which pins torch 2.10.x, is in the top-level
[pyproject.toml](../pyproject.toml).

## Layout

| Path | What |
|------|------|
| [placer.py](placer.py) | `XplacePlacer`, the entry point `evaluate` invokes |
| [_ils_polish_worker.py](_ils_polish_worker.py) | Subprocess worker for parallel polish phases |
| [extensions/](extensions/) | C source for legalize, congestion, refine; built on first run with `-march=native` |
| [GPU/](GPU/) | GPU placement code: basin-jump, batched probes, device state |
| [kernels/](kernels/) | Triton kernels for HPWL, density, routing emit |
| [Xplace/](Xplace/) | Vendored Xplace tree (BSD 3-Clause); `cpybin/*.so` fetched separately |
| [Dockerfile](Dockerfile), [build.sh](build.sh), [run.sh](run.sh) | Docker wrappers |
| [Dockerfile.dockerignore](Dockerfile.dockerignore) | Trims the repo-root build context |

## Run-time knobs

Passed to `run.sh` as environment variables:

- `ABUPLACE_IMAGE_NAME`, default `abuplace:latest`
- `ABUPLACE_NETWORK`, default `none` to match the judges' runtime; use `bridge` to debug
- `ABUPLACE_SHM_SIZE`, default `8g`
- `ABUPLACE_EXTRA_ARGS`, extra `docker run` flags appended verbatim

Any `XP_*` variable `placer.py` reads can be forwarded. [TUNING.md](TUNING.md) lists them all.

```bash
ABUPLACE_EXTRA_ARGS='-e XP_TIER2=0' abuplace/run.sh -b ibm01
```

`build.sh` takes `ABUPLACE_IMAGE_NAME` too.
