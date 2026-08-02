#!/usr/bin/env bash
# Overnight tier-2 measurement on mempool_tile_ng45.
#
# Two ORFS runs:
#   1. AbuPlace's placement (your submission)
#   2. TILOS-shipped initial.plc placement (the design's default reference;
#      for mempool_tile this is Cadence-CMP-derived, not strictly RePlAce —
#      but the closest "baseline" we have without writing a custom OpenROAD
#      TCL for true RePlAce-only macro placement).
#
# Output: $OUTDIR contains both ORFS evaluation_summary.json files plus
# raw flow logs.
#
# Estimated wall: 2.5-4 hours.
#
# Usage:
#   nohup ./scripts/overnight_mempool_tier2.sh > /tmp/overnight_mempool.log 2>&1 &

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
export LD_LIBRARY_PATH="$REPO_ROOT/abuplace/Xplace/cpp_to_py/cpybin:${LD_LIBRARY_PATH:-}"

# Point ORFS_ROOT at your OpenROAD-flow-scripts checkout (see scripts/TIER2_RUNBOOK.md).
ORFS_ROOT="${ORFS_ROOT:-$REPO_ROOT/../OpenROAD-flow-scripts}"
if [[ ! -d "$ORFS_ROOT" ]]; then
    echo "ERROR: ORFS not found at $ORFS_ROOT. Set ORFS_ROOT=/path/to/OpenROAD-flow-scripts." >&2
    exit 1
fi

OUTDIR="${OUTDIR:-/tmp/overnight_mempool_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUTDIR"
ORFS_DESIGN_BASE="$ORFS_ROOT/flow"

echo "============================================================"
echo " Overnight tier-2 run: mempool_tile_ng45"
echo " OUTDIR: $OUTDIR"
echo " Started: $(date)"
echo "============================================================"

# ─── Step 1: Generate AbuPlace's placement ───────────────────────
echo
echo "=== $(date) Step 1: AbuPlace placement (XP_CRIT_WEIGHTS=fanout) ==="
XP_CRIT_WEIGHTS=fanout XP_CRIT_ALPHA="${XP_CRIT_ALPHA:-1.0}" \
uv run python scripts/place_and_save.py \
    --placer abuplace/placer.py \
    --benchmark mempool_tile_ng45 \
    2>&1 | tee "$OUTDIR/01_abuplace_place.log"

ABUPLACE_PT="output/placements/abuplace_mempool_tile_ng45.pt"
test -f "$ABUPLACE_PT" || { echo "FAIL: $ABUPLACE_PT missing"; exit 1; }

# ─── Step 2: ORFS finish with AbuPlace placement ────────────────
echo
echo "=== $(date) Step 2: ORFS finish on AbuPlace placement ==="
uv run python scripts/evaluate_with_orfs.py \
    --benchmark mempool_tile_ng45 \
    --orfs-root "$ORFS_ROOT" \
    --placement "$ABUPLACE_PT" \
    --output "$OUTDIR/orfs_abuplace" \
    2>&1 | tee "$OUTDIR/02_orfs_abuplace.log" || \
    echo "WARN: ORFS run 1 exited non-zero; check log + partial results"

# Backup ORFS artifacts before clearing for run 2
for d in results objects logs reports; do
    if [ -d "$ORFS_DESIGN_BASE/$d/nangate45/mempool_tile" ]; then
        cp -r "$ORFS_DESIGN_BASE/$d/nangate45/mempool_tile" \
              "$OUTDIR/run1_abuplace_$d" 2>/dev/null || true
    fi
done

# ─── Step 3: Clear ORFS intermediates so run 2 starts clean ──────
# Keep synth output (1_synth.*) to save 5-15 min via --skip-synthesis;
# clear floorplan onward.
echo
echo "=== $(date) Step 3: clearing post-synth ORFS state for run 2 ==="
for d in objects logs reports; do
    rm -rf "$ORFS_DESIGN_BASE/$d/nangate45/mempool_tile" || true
done
# In results dir, keep 1_synth.* (synthesis output) and delete the rest.
if [ -d "$ORFS_DESIGN_BASE/results/nangate45/mempool_tile/base" ]; then
    find "$ORFS_DESIGN_BASE/results/nangate45/mempool_tile/base" \
         -maxdepth 1 -type f ! -name "1_synth.*" -delete
fi

# ─── Step 4: ORFS finish with shipped default placement ──────────
echo
echo "=== $(date) Step 4: ORFS finish on shipped initial.plc placement ==="
uv run python scripts/evaluate_with_orfs.py \
    --benchmark mempool_tile_ng45 \
    --orfs-root "$ORFS_ROOT" \
    --output "$OUTDIR/orfs_default" \
    --skip-synthesis \
    2>&1 | tee "$OUTDIR/03_orfs_default.log" || \
    echo "WARN: ORFS run 2 exited non-zero; check log + partial results"

# Backup ORFS artifacts
for d in results objects logs reports; do
    if [ -d "$ORFS_DESIGN_BASE/$d/nangate45/mempool_tile" ]; then
        cp -r "$ORFS_DESIGN_BASE/$d/nangate45/mempool_tile" \
              "$OUTDIR/run2_default_$d" 2>/dev/null || true
    fi
done

# ─── Step 5: Print summary ───────────────────────────────────────
echo
echo "============================================================"
echo " DONE: $(date)"
echo "============================================================"
echo
echo "=== AbuPlace summary ==="
cat "$OUTDIR/orfs_abuplace/evaluation_summary.json" 2>/dev/null || \
    echo "(missing — check $OUTDIR/02_orfs_abuplace.log)"
echo
echo "=== default summary ==="
cat "$OUTDIR/orfs_default/evaluation_summary.json" 2>/dev/null || \
    echo "(missing — check $OUTDIR/03_orfs_default.log)"
echo
echo "Outputs: $OUTDIR"
