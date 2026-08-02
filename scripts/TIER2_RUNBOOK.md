# Tier-2 evaluation runbook (WNS / TNS / Area)

End-to-end recipe for running AbuPlace's macro placements through the
full OpenROAD PnR flow on NG45 designs and reading WNS / TNS / Area -
the metrics the Grand Prize is judged on
([SCORING.md](../SCORING.md)).

Two scripts do the work:

1. [`scripts/place_and_save.py`](place_and_save.py) -  runs a placer on
   one or more NG45 benchmarks and saves the macro positions as a `.pt`
   tensor.
2. [`scripts/evaluate_with_orfs.py`](evaluate_with_orfs.py) -  feeds the
   `.pt` placement into ORFS (`make`), watches the full flow, parses
   WNS/TNS/Area out of the report files.

## 0. Prerequisites

- Disk: ORFS clone is ~5-10 GB and each design run produces
  ~20-30 GB of intermediate flow artifacts. Plan for 40+ GB free
  before kicking off a run.
- Docker with the NVIDIA Container Toolkit -  confirm `docker info`
  lists the `nvidia` runtime.
- NG45 collateral under `external/MacroPlacement/` -  the submodule must
  be checked out, and `Flows/NanGate45/` and `Enablements/NanGate45` must be
  present.
- Benchmarks .pt under `benchmarks/processed/public/` (shipped in the
  repo -  the scripts read `ariane133_ng45.pt`, etc.).

## 1. Install OpenROAD-flow-scripts (one-time)

```bash
git clone --depth=1 https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts \
    /path/with/40GB/free/OpenROAD-flow-scripts
```

`evaluate_with_orfs.py` defaults to `../OpenROAD-flow-scripts`. If you cloned
elsewhere, pass `--orfs-root /path/to/OpenROAD-flow-scripts` (the shell scripts
read an `ORFS_ROOT` environment variable for the same purpose).

Both the ORFS checkout and Docker's data-root need room -  the
`openroad/orfs:latest` image is ~10 GB and is auto-pulled on the first run. If
your root filesystem is small, point Docker's `data-root` at a larger disk in
`/etc/docker/daemon.json`.

You do not need to build ORFS natively -  Docker mode pulls the
prebuilt image (`openroad/orfs:*`) on first run.

## 2. Run AbuPlace -> produce `.pt` placements

```bash
# from the repo root

# One design (good first smoke test):
uv run python scripts/place_and_save.py \
    --placer abuplace/placer.py \
    --benchmark ariane133_ng45

# All five NG45 designs:
uv run python scripts/place_and_save.py \
    --placer abuplace/placer.py --all
```

Each run prints the proxy cost and validity check, then writes
`output/placements/abuplace_<benchmark>.pt`. NG45 designs are larger
than IBM benchmarks; expect 3-10 minutes per design for the AbuPlace
GP + refinement cascade.

## 3. Run ORFS tier-2 flow

`evaluate_with_orfs.py` uses ORFS's `util/docker_shell` wrapper by
default (omit `--no-docker`). It mounts the design dir, kicks off
`make`, and waits for the final `6_report.json`.

```bash
# Set once per shell so we don't repeat --orfs-root:
ORFS=${ORFS_ROOT:-../OpenROAD-flow-scripts}

# One design:
uv run python scripts/evaluate_with_orfs.py \
    --benchmark ariane133_ng45 \
    --orfs-root "$ORFS" \
    --placement output/placements/abuplace_ariane133_ng45.pt

# All five:
for b in ariane133_ng45 ariane136_ng45 bp_quad_ng45 nvdla_ng45 mempool_tile_ng45; do
    uv run python scripts/evaluate_with_orfs.py \
        --benchmark "$b" \
        --orfs-root "$ORFS" \
        --placement "output/placements/abuplace_${b}.pt"
done
```

A full flow takes 3-8 hours per design end-to-end. The script writes
incremental progress to `output/orfs_evaluation/evaluation_summary.json`
as each design finishes.

## 4. Read the results

After all runs complete, the script prints a final table:

```
Benchmark              Proxy Cost       WNS (ns)    TNS (ns)    Fmax (MHz)    Wire (um)    Area (um²)
---------------------------------------------------------------------------------------------------
ariane133_ng45         0.756123         -0.42       -187.34     385.7         842.34       1.234
…
```

JSON-formatted full results at
`output/orfs_evaluation/evaluation_summary.json`. The fields you score
on:

| Field | Source in ORFS | Notes |
|---|---|---|
| `wns` | `6_report.json` -> finalize/sta WNS | Worst negative slack (ns); less-negative is better |
| `tns` | `6_report.json` -> finalize/sta TNS | Total negative slack (ns); less-negative is better |
| `area` | `6_report.json` -> die area | μm²; smaller is better |

## 5. Score vs baselines

Apply [SCORING.md](../SCORING.md) per design:

```
R_WNS  = WNS_avg  / WNS_sub       (both negative; ratio > 1 means submission better)
R_TNS  = TNS_avg  / TNS_sub
R_Area = Area_avg / Area_sub
Design_Score = (R_WNS^3 x R_TNS^2 x R_Area^1) ^ (1/6)
```

`*_avg` = mean of the SA and RePlAce baseline values for the design.
The submission qualifies for the Grand Prize iff for every design,
WNS_sub >= min(WNS_SA, WNS_RP) and TNS_sub >= min(TNS_SA, TNS_RP).

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `placer file not found` | Wrong relative path; run scripts from repo root. |
| `Benchmark .pt not found` | `benchmarks/processed/public/<name>.pt` missing. The user's MEMORY says we never strip these from `processed/`. |
| `Source directory not found` | An `external/MacroPlacement/Flows/NanGate45/<name>/` subtree got deleted. Repopulate from the TILOS submodule. |
| ORFS docker mode fails to pull image | First run downloads `openroad/orfs:*`; needs internet + ~10 GB disk. |
| Flow OOM-kills | `_set_memory_limit()` caps subprocesses at 100 GB; raise in `evaluate_with_orfs.py:33` if you have more RAM. |
| `placement shape != [num_macros, 2]` | Wrapper's validation tripped -  placer returned wrong shape; check the placer's `place()` return. |

## Reference baselines

[`baselines/ng45_baselines.csv`](../baselines/ng45_baselines.csv) carries the
published SA and RePlAce numbers for the NG45 designs -  core/std-cell/macro
area, power, wirelength, WNS and TNS -  as reported in the TILOS paper
(Innovus postRoute). These are the numbers a Tier-2 result is compared against,
so keep them alongside whatever `evaluate_with_orfs.py` reports.
