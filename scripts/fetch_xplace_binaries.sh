#!/usr/bin/env bash
# Fetch the prebuilt Xplace CUDA extensions into abuplace/Xplace/cpp_to_py/cpybin/.
#
# These are not tracked in git: they are ~16 MB of opaque binaries pinned to
# cpython-3.12 and the CUDA 12 ABI. They are published as a release asset and
# verified by checksum here.
#
# Usage:
#   scripts/fetch_xplace_binaries.sh
#
# Env:
#   XPLACE_BIN_TAG   release tag to pull from   (default: )
#   XPLACE_BIN_REPO  owner/repo                 (default: derived from git remote)
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST="$REPO_ROOT/abuplace/Xplace/cpp_to_py/cpybin"

TAG="${XPLACE_BIN_TAG:-xplace-bin-v1}"
ASSET="xplace-cpybin-py312-cu12.tar.gz"
EXPECTED_SHA256="323f10921b2c9bb6b0e8ad1e540b20c5234007e66652e9061d4c8438d7170128"

# Derive owner/repo from the git remote unless overridden.
if [[ -n "${XPLACE_BIN_REPO:-}" ]]; then
    REPO="$XPLACE_BIN_REPO"
else
    REPO=$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null \
             | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##')
    [[ -z "$REPO" ]] && { echo "ERROR: no git remote; set XPLACE_BIN_REPO=owner/repo" >&2; exit 1; }
fi

URL="https://github.com/$REPO/releases/download/$TAG/$ASSET"

echo "[fetch] repo    : $REPO"
echo "[fetch] tag     : $TAG"
echo "[fetch] dest    : $DEST"

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
echo "[fetch] downloading $URL"
if ! curl -fSL --retry 3 -o "$TMP/$ASSET" "$URL"; then
    cat >&2 <<MSG

ERROR: could not download $ASSET from $URL

  * If the release does not exist yet, build Xplace from upstream instead:
    https://github.com/cuhk-eda/Xplace  (needs nvcc + cmake), then copy the
    resulting cpp_to_py/cpybin/*.so into $DEST
  * Or use the Docker image, which does not need this step (see abuplace/README.md).
MSG
    exit 1
fi

echo "[fetch] verifying checksum"
ACTUAL=$(sha256sum "$TMP/$ASSET" | cut -d' ' -f1)
if [[ "$ACTUAL" != "$EXPECTED_SHA256" ]]; then
    echo "ERROR: checksum mismatch" >&2
    echo "  expected $EXPECTED_SHA256" >&2
    echo "  actual   $ACTUAL" >&2
    exit 1
fi

mkdir -p "$DEST"
tar xzf "$TMP/$ASSET" -C "$TMP"
cp -f "$TMP/cpybin/"*.so "$DEST/"
[[ -f "$TMP/cpybin/__init__.py" ]] && cp -f "$TMP/cpybin/__init__.py" "$DEST/"

N=$(ls "$DEST"/*.so 2>/dev/null | wc -l)
echo "[fetch] installed $N shared objects into $DEST"
[[ "$N" -eq 14 ]] || { echo "WARNING: expected 14 .so, found $N" >&2; exit 1; }
echo "[fetch] done"
