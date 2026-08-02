#!/usr/bin/env bash
# Extract per-net slack from the existing ORFS run on mempool_tile.
# Output: /tmp/per_net_slack_mempool_tile.json
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
DESIGN="${1:-mempool_tile}"
TECH="${2:-nangate45}"
ORFS_ROOT="${ORFS_ROOT:-$REPO_ROOT/../OpenROAD-flow-scripts}"
if [[ ! -d "$ORFS_ROOT" ]]; then
    echo "ERROR: ORFS not found at $ORFS_ROOT. Set ORFS_ROOT=/path/to/OpenROAD-flow-scripts." >&2
    exit 1
fi

# We need extract_slack.tcl visible inside docker. Copy into the flow's util/.
cp scripts/extract_slack.tcl $ORFS_ROOT/flow/util/extract_slack.tcl

# Paths are /work-relative since docker_shell mounts host flow at /work.
ODB=/work/results/$TECH/$DESIGN/base/5_route.odb
SDC=/work/results/$TECH/$DESIGN/base/5_route.sdc

# Platform LEF/lib paths inside docker (image bundles platforms/).
TECH_LEF=/OpenROAD-flow-scripts/flow/platforms/$TECH/lef/NangateOpenCellLibrary.tech.lef
CELL_LEF=/OpenROAD-flow-scripts/flow/platforms/$TECH/lef/NangateOpenCellLibrary.macro.mod.lef
PLATFORM_LIB=/OpenROAD-flow-scripts/flow/platforms/$TECH/lib/NangateOpenCellLibrary_typical.lib

# Macro LEF/lib from design dir (we copied these in via evaluate_with_orfs.py).
DESIGN_DIR=/work/designs/$TECH/$DESIGN
# Two fakeram macros for mempool_tile.
ADD_LEFS="$DESIGN_DIR/fakeram45_256x32.lef $DESIGN_DIR/fakeram45_64x64.lef"
ADD_LIBS="$DESIGN_DIR/fakeram45_256x32.lib $DESIGN_DIR/fakeram45_64x64.lib"

OUT_JSON_IN_DOCKER=/work/per_net_slack_${DESIGN}.json
OUT_JSON_HOST=$ORFS_ROOT/flow/per_net_slack_${DESIGN}.json

echo "=== Extracting per-net slack from $DESIGN ($TECH) ==="
echo "    odb: $ODB"
echo "    output: $OUT_JSON_HOST"

cd $ORFS_ROOT/flow

# Build one quoted command for docker_shell to run via its inner bash -c.
INNER="cd /work && \
EXTRACT_ODB='$ODB' \
EXTRACT_SDC='$SDC' \
EXTRACT_TECH_LEF='$TECH_LEF' \
EXTRACT_LEFS='$CELL_LEF $ADD_LEFS' \
EXTRACT_LIB_FILES='$PLATFORM_LIB $ADD_LIBS' \
EXTRACT_OUT_JSON='$OUT_JSON_IN_DOCKER' \
EXTRACT_PATH_COUNT=50000 \
openroad -no_init -exit /work/util/extract_slack.tcl"

./util/docker_shell -- "$INNER"

if [ -f $OUT_JSON_HOST ]; then
    cp $OUT_JSON_HOST /tmp/per_net_slack_${DESIGN}.json
    echo
    echo "=== Saved: /tmp/per_net_slack_${DESIGN}.json ==="
    python3 -c "
import json
d = json.load(open('/tmp/per_net_slack_${DESIGN}.json'))
print(f'Total nets with negative slack: {len(d)}')
print('Worst 10 nets:')
for k, v in sorted(d.items(), key=lambda kv: kv[1])[:10]:
    print(f'  {v:9.3f} ns  {k}')
"
else
    echo "ERROR: expected $OUT_JSON_HOST"
    exit 1
fi
