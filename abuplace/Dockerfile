# Self-contained image for the AbuPlace macro-placer
# (abuplace/placer.py).
#
# Single-stage on purpose: a multi-stage build with COPY --from=builder
# duplicates the 6+ GB venv across both stages during build, which
# blows up peak disk on tight hosts. Single-stage with all cleanup in
# the same RUN keeps the on-disk footprint to one copy of the venv.
#
# Base: nvidia/cuda 12.6 *base* (NOT runtime). The torch wheels bundle
# their own cuDNN / cuBLAS / cuFFT etc. under
# `.venv/lib/python3.12/site-packages/nvidia/`, so the apt-installed
# copies in the larger `:runtime` image would just be duplicated bytes.
# The driver lib (libcuda.so) is supplied at run time by the NVIDIA
# Container Toolkit via `--gpus all`.
#
# Build context MUST be the repo root (so macro_place/ and submissions/
# are reachable). Use abuplace/build.sh.
#
# Runtime requirements on the host:
#   - NVIDIA driver compatible with CUDA 12.6 (>= 560.x)
#   - NVIDIA Container Toolkit (`--gpus all` works)
#   - external/MacroPlacement and benchmarks/ trees populated and mounted
#     at /work/external and /work/benchmarks (see run.sh).
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=16 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

# System deps (one layer, apt cache cleaned in same RUN):
#   python3.12 + venv  : runtime
#   python3.12-dev     : Python.h for Triton's runtime CUDA-helper JIT
#                        (without it, softmove + other GPU passes fall
#                         back to CPU and quality regresses)
#   gcc + libc6-dev    : compile AbuPlace/extensions/*.so on first run
#                        (lighter than full build-essential)
#   libgomp1           : OpenMP runtime for those .so
#   libcairo2          : DT_NEEDED by Xplace's draw_placement cpybin .so
#                        (gets imported even if visualization isn't used)
#   ca-certificates    : TLS for uv installer + pip
#   curl               : uv installer
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        gcc libc6-dev libgomp1 \
        libcairo2 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://astral.sh/uv/install.sh \
       | env UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /work

# Minimum sources for `pip install -e .[abuplace]` to resolve.
# .dockerignore (sidecar to this file) trims the build context.
COPY pyproject.toml README.md LICENSE.md /work/
COPY macro_place/ /work/macro_place/
COPY abuplace/ /work/abuplace/

# The prebuilt Xplace CUDA extensions are not tracked in git. They must be
# present in the build context — run scripts/fetch_xplace_binaries.sh on the
# host first. Fail loudly here rather than building a silently broken image.
RUN n=$(ls /work/abuplace/Xplace/cpp_to_py/cpybin/*.so 2>/dev/null | wc -l); \
    if [ "$n" -ne 14 ]; then \
        echo "ERROR: expected 14 Xplace .so in the build context, found $n." >&2; \
        echo "Run scripts/fetch_xplace_binaries.sh on the host, then rebuild." >&2; \
        exit 1; \
    fi

# Install + trim + cleanup in ONE layer so the committed layer doesn't
# carry bytes we're about to delete.
#
# Trimmed torch CUDA bundles (yp single-GPU eval doesn't need them):
#   nvidia-nccl-cu12       — 410 MB — multi-GPU collectives
#   nvidia-cusparselt-cu12 — 432 MB — Hopper sparse matmul
#   nvidia-nvshmem-cu12    — 195 MB — multi-GPU shared memory
#   nvidia-cufile-cu12     —   3 MB — GPUDirect Storage
RUN uv venv --python python3.12 /work/.venv \
    && uv pip install --python /work/.venv/bin/python -e ".[abuplace]" \
    && uv pip uninstall --python /work/.venv/bin/python \
        nvidia-nccl-cu12 \
        nvidia-cusparselt-cu12 \
        nvidia-nvshmem-cu12 \
        nvidia-cufile-cu12 \
    && rm -f /work/abuplace/extensions/*.so \
    && find /work/abuplace -name __pycache__ -type d -prune -exec rm -rf {} + \
    && find /work/.venv -name __pycache__ -type d -prune -exec rm -rf {} + \
    && rm -rf /root/.cache /tmp/* /usr/local/bin/uv

ENV PATH=/work/.venv/bin:${PATH}

ENTRYPOINT ["evaluate", "abuplace/placer.py"]
CMD ["--help"]
