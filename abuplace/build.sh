#!/usr/bin/env bash
# Build the abuplace:latest image from the repo root.
#
# Env knobs:
#   ABUPLACE_IMAGE_NAME  default: abuplace:latest
#
# First build is ~5-10 min on a fast connection (mostly torch + cuda
# wheels). Subsequent builds reuse layers when only the source tree
# under abuplace/ changes.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
IMAGE_NAME=${ABUPLACE_IMAGE_NAME:-abuplace:latest}

# Force buildkit so that Dockerfile.dockerignore is honored.
export DOCKER_BUILDKIT=1

echo "[build] context : $REPO_ROOT"
echo "[build] image   : $IMAGE_NAME"

# --network=host: BuildKit's default sandbox network sometimes lacks
# DNS forwarding on hosts where systemd-resolved binds the resolver to
# specific links (e.g. wifi only). host-network sidesteps that so apt
# + uv can resolve archive.ubuntu.com / astral.sh / pypi.org directly.
exec docker build \
    --network=host \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t "$IMAGE_NAME" \
    "$REPO_ROOT"
