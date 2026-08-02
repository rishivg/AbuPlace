# Environment knobs

Every knob the placer reads, with its default. None of these need to be set -
the defaults are what the placer was validated with, and it self-tunes per design
from a congestion probe rather than from configuration. They exist for A/B
experiments, debugging, and adapting to different hardware.

Set them like any environment variable:

```bash
XP_RUDY_BO2=0 uv run evaluate abuplace/placer.py -b ibm01
```

Through Docker, forward them with `ABUPLACE_EXTRA_ARGS`:

```bash
ABUPLACE_EXTRA_ARGS='-e XP_TIER2=0' abuplace/run.sh -b ibm01
```

---

## Pipeline structure

| Knob | Default | Effect |
|---|---|---|
| `XP_RUDY_BO2` | `1` | Master switch for the best-of-N outer loop. `0` runs a single trajectory (~3x faster, usually worse). |
| `XP_RUDY_BO_KERNELS` | `off,rudy,rudy_hv` | Which congestion-loss trajectories to run. `off` is the validated baseline and should stay first -  it is the guaranteed fallback. |
| `XP_RUDY_BO_BUDGET_S` | `2500` | Wall-clock guard, in seconds. A later trajectory is skipped if it likely wouldn't fit, so best-of-N cannot blow the 1-hour cap. |
| `XP_CONG_LOSS` | `0` | Enable Xplace's RUDY congestion loss during global placement. Normally driven by the best-of-N loop, not set by hand. |
| `XP_CONG_LOSS_KERNEL` | -  | Which RUDY kernel: `rudy`, `rudy_hv`, or `rudy_abu`. Only read when `XP_CONG_LOSS=1`. |
| `XP_BJ_BUDGET` | `35.0` | Seconds allotted to the basin-jump escape stage. |

## RUDY congestion loss

Only read when `XP_CONG_LOSS=1`. These shape the differentiable congestion loss
that the `rudy` / `rudy_hv` trajectories add to the Xplace Nesterov gradient
(implemented in `abuplace/Xplace/src/calculator.py`, a local addition to upstream
Xplace).

| Knob | Default | Effect |
|---|---|---|
| `XP_CONG_LOSS_W` | `0.05` | Loss weight, multiplied by the density weight. |
| `XP_CONG_LOSS_BINS` | `32` | Grid resolution for the congestion estimate. |
| `XP_CONG_LOSS_START` | `100` | Iteration at which the loss starts being applied. |
| `XP_CONG_LOSS_RAMP` | `100` | Iteration by which the weight reaches full strength. |
| `XP_CONG_LOSS_EVERY` | `1` | Apply every K-th iteration, with the gradient scaled by K to keep the average effect constant. Raise to cut per-iteration cost. |
| `XP_CONG_LOSS_CAP` | `1.5` | Upper clamp on the loss. |
| `XP_CONG_LOSS_MACROS_ONLY` | `1` | Restrict the loss to macros rather than all movable nodes. |
| `XP_CONG_LOSS_RAMPDOWN_OF` | `0.0` | Overflow at which the weight begins ramping down. `0` disables rampdown. |
| `XP_CONG_LOSS_RAMPDOWN_FLOOR` | `0.2` | Floor the rampdown will not go below. |
| `XP_CONG_LOSS_ADAPT_OF` | `0.0` | Overflow threshold for adapting the weight. `0` disables. |
| `XP_CONG_LOSS_ADAPT_PEAK` | `0` | `1` adapts the weight from the measured congestion peak. |
| `XP_CONG_LOSS_ADAPT_PEAK_LO` | `1.5` | Peak below which the weight is reduced. |
| `XP_CONG_LOSS_ADAPT_PEAK_HI` | `2.5` | Peak above which the weight is raised. |

## Global placement

| Knob | Default | Effect |
|---|---|---|
| `XP_GP_SEED` | per-tier | Xplace RNG seed. Changing it changes the starting basin. |
| `XP_GP_TARGET_DENSITY` | per-tier | Target bin density for the electrostatic solve. Higher packs tighter (better wirelength, worse density and congestion). |

## Adaptive tuning

The refine cascade scales its congestion weight by a measured congestion probe.
These control that response curve.

| Knob | Default | Effect |
|---|---|---|
| `XP_ADAPT_GATE` | `1.85` | Congestion below which the R2/R3 congestion multiplier is scaled down. |
| `XP_ADAPT_R2` | `0.4` | Scale applied to the R2 congestion multiplier on low-congestion designs. |
| `XP_ADAPT_R3` | `0.3` | Same, for R3. |
| `XP_CRIT_WEIGHTS` | -  | Set to `fanout` to bias wirelength toward high-fanout nets, a stand-in for slack-driven weighting. Off by default. |
| `XP_CRIT_ALPHA` | `1.0` | Strength of that bias: `weight = 1 + alpha * log1p(fanout - 1)`. Only read when `XP_CRIT_WEIGHTS=fanout`. |

## Operators

| Knob | Default | Effect |
|---|---|---|
| `XP_SHAKE_TOP_K` | `60` | How many hot hard macros stage 4a perturbs. |
| `XP_SHAKE_STEP_REL` | `0.125` | Shake step, as a fraction of a bin. |
| `XP_SHAKE_BUDGET_S` | `3.5` | Seconds allotted to stage 4a. |
| `XP_SHAKE_VERIFY_N` | `4` | Parallel verification workers for shake candidates. |

## Parallelism and memory

| Knob | Default | Effect |
|---|---|---|
| `XP_ILS_K` | `4` | Subprocess workers in the parallel polish pool. Raise on a machine with more cores. |
| `XP_ILS_OMP` | `4` | OpenMP threads per worker. Keep below 6 -  at or above it, libgomp changes its reduction order, which perturbs the summed cost. |
| `XP_PROBE_CHUNK_BYTES` | `1500 MB` | GPU probe chunk budget, sized for 8 GB-class cards. Lower it if you hit OOM on large designs; raise it on a big card for speed. |

## GPU polish internals

Defaults here were co-tuned with the polish stage; changing them mainly serves
A/B experiments.

| Knob | Default | Effect |
|---|---|---|
| `XP_POLISH_IS_TOPK` | `8` | Commits allowed per independent set. The placer raises it to `999` for stage 2. |
| `XP_POLISH_IS_BBOX` | `0` | Set `1` to add a bbox-overlap check when building the independent-set partition. Off by default: it is conservative, and the false conflicts cost more parallelism than they buy. |
| `XP_POLISH_EARLY_SKIP` | `1` | Skip probes whose pins cross no gcell -  they have exactly zero routing delta. Disabling only costs time. |
| `XP_POLISH_IS_BASIC` | `0` | Use plain axis candidates instead of the gradient-guided "smart" ones. |
| `XP_POLISH_EXPANDED_CANDS` | `0` | `1` or `2` densifies the candidate step multipliers. |
| `XP_POLISH_AUX_STREAM` | `0` | Run density and HPWL on a second CUDA stream, concurrent with the routing kernel. Off by default: on a shared GPU, SM contention can cost more than the overlap saves. |

## Tier-2 (OpenROAD flow)

These affect only the NG45 designs evaluated through OpenROAD. They are
invisible to the Tier-1 proxy -  on the small IBM canvases the clearance guard
disables them automatically.

| Knob | Default | Effect |
|---|---|---|
| `XP_TIER2` | `1` | Master switch for all Tier-2 features. `0` gives a clean Tier-1-only A/B baseline. |
| `XP_CLEARANCE_UM` | `12` | Minimum hard-macro clearance, in µm, matching what the Tier-2 evaluator enforces anyway. `0` disables and leaves the push to the evaluator. |
| `XP_PDN_ROWSNAP` | `1` | Restructure into clean rows when a power-distribution channel would otherwise pinch shut. Prevents `[ERROR PDN-0179]`, at some wirelength cost. |
| `XP_PDN_CHANNEL_UM` | `30` | Target channel width, roughly 2x the NG45 macro halo. |
| `XP_PDN_HGAP` | `12` | Minimum horizontal gap between macros within a row. |

## Debugging

| Knob | Default | Effect |
|---|---|---|
| `XP_DET_TRACE` | `0` | Dump full-precision proxy every iteration, for bisecting determinism breaks. |
| `XP_POLISH_PROFILE` | `0` | Per-phase GPU polish timings. |
| `XP_GPU_HYBRID_VERBOSE` | `0` | Per-stage timings for the hybrid polish. |
