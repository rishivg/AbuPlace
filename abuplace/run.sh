#!/usr/bin/env bash
# Run the AbuPlace image with the host's external/ and benchmarks/
# bind-mounted in. All args after the script name are forwarded to the
# placer (e.g. `-b ibm01`, `--all`, `--ng45`).
#
# Env knobs:
#   ABUPLACE_IMAGE_NAME    default: abuplace:latest
#   ABUPLACE_NETWORK       default: none  (matches judge env; set to
#                                        "bridge" or "host" if you need
#                                        the container to reach the net)
#   ABUPLACE_SHM_SIZE      default: 8g    (torch dataloader / multiproc)
#   ABUPLACE_EXTRA_ARGS    extra `docker run` flags appended verbatim
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

IMAGE_NAME=${ABUPLACE_IMAGE_NAME:-abuplace:latest}
NETWORK=${ABUPLACE_NETWORK:-none}
SHM_SIZE=${ABUPLACE_SHM_SIZE:-8g}

if [[ ! -d "$REPO_ROOT/external/MacroPlacement" ]]; then
    echo "[run] ERROR: $REPO_ROOT/external/MacroPlacement not found." >&2
    echo "[run] Initialize it first: git submodule update --init external/MacroPlacement" >&2
    exit 1
fi
if [[ ! -d "$REPO_ROOT/benchmarks/processed" ]]; then
    echo "[run] ERROR: $REPO_ROOT/benchmarks/processed not found." >&2
    echo "[run] See SETUP.md for how to populate the processed benchmark cache." >&2
    exit 1
fi

# shellcheck disable=SC2086  # we want word-splitting on ABUPLACE_EXTRA_ARGS
exec docker run --rm \
    --gpus all \
    --network "$NETWORK" \
    --shm-size "$SHM_SIZE" \
    -v "$REPO_ROOT/external:/work/external:ro" \
    -v "$REPO_ROOT/benchmarks:/work/benchmarks:ro" \
    ${ABUPLACE_EXTRA_ARGS:-} \
    "$IMAGE_NAME" "$@"
