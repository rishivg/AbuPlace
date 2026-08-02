# AbuPlace: how it works

An end-to-end explanation of the `abuplace` macro placer: what each piece does,
how the pieces fit together, and why the design is shaped the way it is.

In short: it seeds a layout with Xplace analytical global placement, then runs a
long refinement cascade alternating between a faithful C reimplementation of the
competition's proxy-cost evaluator, used as an optimization oracle, and a stack
of discrete local-search operators. The pipeline runs three times with different
congestion-loss settings and ships the lowest-proxy legal result. On top of that
sits a robustness net so a valid placement always comes back: no crash, no
overlap, no disqualification.

---

## 1. The problem and the objective

Macro placement positions large fixed-size blocks (SRAMs, IP, analog macros) on
a chip canvas. The competition ranks submissions (Tier 1) by a single proxy
cost computed by the TILOS MacroPlacement evaluator:

```
Proxy Cost = 1.0 x Wirelength + 0.5 x Density + 0.5 x Congestion
```

- Wirelength: normalized half-perimeter wirelength (HPWL) summed over all nets.
- Density: top-10% mean bin occupancy (macros want to spread).
- Congestion: top-5% mean routing demand over a routing grid (`ABU` /
  area-based-utilization metric on smoothed horizontal+vertical demand maps).

These three objectives conflict: wirelength wants macros clustered, density
wants them spread, congestion wants routing channels left open. The search space
is enormous (~10^800 placements), non-convex, with hard zero-overlap constraints
and a hard 1-hour-per-benchmark runtime cap. Everything in the design follows
from those facts.

There is also a Tier 2 (OpenROAD flow on NG45 designs) judged on real timing
(WNS/TNS) and area. `abuplace` adds two Tier-2-only features (orientation sidecar
and PDN clearance push) that are invisible to the Tier-1 proxy.

The entry point is `placer.py :: XplacePlacer.place(benchmark) -> torch.Tensor`,
returning `[N, 2]` macro centers. Everything below hangs off that one call.

---

## 2. The central idea: own the evaluator

The single most important architectural decision is this:

> The proxy-cost function the judges use is reimplemented bit-faithfully in C
> (`extensions/congestion.c`) and used as the optimization oracle.

The header of `congestion.c` says it ports
`plc_client_os.get_routing() + __smooth_routing_cong() + abu(...)` exactly -
the same H/V routing stripes for 2/3-pin nets, star-split for larger nets,
macro routing allocation with partial-overlap correction, the same 1-D smoothing,
and the same top-5% ABU on the concatenated V||H demand array.

Why go to this trouble?

1. No proxy/optimizer mismatch. Most placers optimize a smooth surrogate
   (analytical density + weighted HPWL) and only measure the true proxy at the
   end. Any gap between surrogate and true cost is wasted effort. By optimizing
   the real thing, every accepted move is a guaranteed real improvement.
2. Cheap incremental scoring. Because the placer owns the cost code, it can
   expose an incremental API (`CongState`, below) that rescoring a single
   macro move in ~0.5-2 ms instead of a 5-15 ms full recompute - the enabling
   trick behind the entire discrete-operator stack.
3. Greedy accept/reject is provably non-regressing on the internal score.
   Every operator probes a move, scores it exactly, and accepts only if the proxy
   strictly drops.

There are two C cost engines plus a GPU one, all kept numerically aligned:

| Engine | File | Role |
|--------|------|------|
| `cong_relax_v2` | `extensions/congestion.c` | Exact proxy + greedy soft-macro coordinate-descent ("polish"); also the `CongState` incremental API |
| `refine_adam` | `extensions/refine.c` | Adam gradient descent on a smooth WL+density+congestion+repulsion surrogate (the analytical refiner) |
| GPU polish | `GPU/forward.py`, `GPU/polish.py`, Triton `kernels/` | A GPU re-implementation of the same proxy + a batched coordinate-descent, used for the fast polish and basin-jump |

`GPU/forward.py` carries comments like "matches C `compute_cong`" / "matches C
`abu_top_n` modulo tie-ordering" - the GPU path is deliberately kept consistent
with the C path so results agree across engines.

---

## 3. The pipeline at a glance

`place()` runs these stages in order. (The whole thing is wrapped in a
best-of-3 loop - see §5.)

```
GP            Xplace analytical global placement  (GPU)         -> rough layout
  |
TUNE          probe post-GP congestion, pick adaptive knobs
  |
STAGE 1       REFINE cascade: R1 -> R1-cc -> R2 -> R3 -> R3-cc       (C Adam + C polish)
  |             (+ optional hard "micro-move" prep)
  |
STAGE 2       POLISH: GPU init + 5x perturb-and-repolish + C short refine  (GPU)
  |
STAGE 3       ESCAPE: basin-jump v3 - GPU perturb-polish chain
  |             + CPU operator stack (hardmove/swap/softmove/window-reorder)
  |
STAGE 4       OPERATE: 4a hard-shake . 4b pair-swap . 4c hardmove .
  |             4d softmove . 4e long-shake . 4f window-reorder
  |             4g Klein-4 orientation (Tier-2) . 4h clearance push (Tier-2)
  |
EVAL          report final proxy; regression guard; legality repair; clamp; ship
```

The arc is coarse-to-fine: GP gets the global structure right, the refine
cascade settles it into a good basin, polish and basin-jump escape local minima,
and the operator stack does fine-grained discrete cleanup. Each stage uses the
exact-proxy oracle as the accept gate, so the proxy only ever moves down.

---

## 4. Stage by stage

### 4.0 GP - Xplace global placement (GPU)

The benchmark is converted to LEF/DEF (`_generate_lefdef`) - synthetic layers,
tracks, gcell grid, components, pins, nets - then parsed by the vendored Xplace
C++ parser into its GPU database. Xplace then runs eplace/Nesterov electrostatic
global placement (`_run_one_config`): macros are charges in an electrostatic
field, density is the electric potential (solved by DCT/FFT on a 128x128 bin
grid), and HPWL uses weighted-average wirelength. This produces a near-overlap-
free, low-wirelength continuous layout in a few seconds.

GP parameters scale with design size (`GP_TIERS`): smaller designs get more bins
and iterations; very large ones drop to coarser grids to stay within budget.

The generated LEF/DEF is cached on disk, and the `.plc` benchmark load + Xplace
init + LEF/DEF parse all run on background threads (the C++ parser releases
the GIL), overlapping the GPU-bound GP work.

### 4.1 TUNE - adaptive knob selection

A single exact-proxy probe of the GP output (`_cc_call`) reads the congestion
level `c_probe`. Nearly every downstream decision is gated on it:

- Refiner congestion multiplier `cong_r1` (2.5 / 3.5 / 4.0 by congestion band).
- Whether to run the hard `micro_move` prep (off when congestion is high - the
  fast/exact score drift hurts dense designs).
- Pair-swap budget and polish schedules (§4.4).
- Whether long-shake (4e) runs at all (only in a narrow congestion window).

The philosophy: one uniform algorithm, self-tuned per design from a measurement
of that design, not benchmark-specific hardcoding (which is disallowed).

### 4.2 STAGE 1 - the refine cascade (C)

Three rounds of `refine_adam` (`extensions/refine.c`), each followed by spiral
legalization, interleaved with shallow `cong_relax_v2` "cc" cleanups:

- R1: aggressive Adam (high LR, congestion-weighted) on the smooth surrogate
  -> legalize -> R1-cc shallow exact-proxy polish. R1-cc's congestion reading
  drives an adaptive R2/R3 congestion multiplier on low-congestion designs.
- R2: medium Adam -> legalize.
- R3: small-step Adam -> legalize -> R3-cc cleanup.

`refine_adam` optimizes a differentiable approximation (Gaussian-bell density,
smoothed routing, WA-HPWL, plus a pairwise repulsion term), which is why it can
take gradient steps the discrete operators can't. Legalization
(`extensions/legalize.c`) is a spiral search: macros placed largest-first,
each pushed to the nearest non-overlapping grid site, with a fine step that falls
back to a coarse step on residual overlap (`_legalize_two_pass`).

The post-R3 state is snapshotted as `_refine_proxy` - a regression guard
baseline (§6).

### 4.3 STAGE 2 - the hybrid polish (GPU)

`gpu_hybrid_5perturb_polish` (`GPU/polish_hybrid_5perturb.py`):

1. Init polish: full GPU coordinate descent (`polish_v1`) on soft macros:
   phased step decay with reheats, first-improving accept, best-proxy restore.
2. 5x perturb-and-repolish: Gaussian-shift K random soft macros
   (`K ≈ max(50, 0.087.n_soft)`), re-init, brief repolish, with 5 fixed seeds.
   This is a small multi-start to jump out of the polish's local minimum.
3. C-short refine: a brief `cong_relax_v2` to reconcile the GPU result with
   the C engine's exact score (small formula/FP differences between engines).

The GPU polish's per-iteration speed comes from an independent-set (IS)
partition (`GPU/is_partition.py`, `GPU/eval_is_batched.py`): macros that share
no nets and don't overlap in their candidate moves are mutually independent, so
all their candidate moves can be scored in one batched Triton kernel launch
(`kernels/polish_emit.py`) instead of one launch per macro - a ~28x reduction in
launch overhead. Routing is gcell-quantized, so probes whose pins don't cross a
gcell boundary are skipped entirely (`_compute_pin_crossed_mask`).

If any GPU stage throws (e.g. OOM on a large design), the polish falls back to a
pure-C `cong_relax_v2` schedule.

### 4.4 STAGE 3 - escape via basin-jump v3

`basin_jump_v3` (`GPU/basin_jump_v3.py`) is the heavy escape stage. It chains:

- Phase 1 - GPU perturb-polish chain (`basin_jump_v2`): repeated
  *perturb -> GPU polish -> C-short -> exact n_iter=0 accept gate*, aborting on the
  first reject. Three congestion-adaptive schedules (FULL/TIGHT/SUPER…) trade
  iteration depth against the wall budget.
- Phases 2-5b - CPU operator stack, run under a live `CongState`:
  hardmove ladders (decreasing step sizes), pair-swap, softmove ladders,
  window-reorder, legalize between passes, then iterated soft-only sweeps and a
  bounded full-stack revisit loop (until improvement < 1.5e-4 or 10 iters).

Crucially, the "best" is tracked only by a fresh `cc_call(n_iter=0)` exact
score - never the drifting incremental `CongState` score - and the GPU chain
result is kept as a safety fallback if the operator stack regresses.

### 4.5 STAGE 4 - the discrete-operator stack

Six fine-grained local-search operators, each gated on `c_probe` and each
following the same pattern: *probe candidate moves via `CongState`, gate accepts
on the exact proxy, legalize, reconcile with a short polish, and verify against a
fresh rebuild before keeping the result.*

| Stage | Operator | What it moves |
|-------|----------|---------------|
| 4a | hard-shake | top-K hottest hard macros, 8-dir x small step; parallel-verified via the ILS worker pool |
| 4b | pair-swap | swap same-size hard-macro pairs (hottest pins first) |
| 4c | hardmove | all hard macros, 16 candidates each (axis/diag + knight-like), multi-pass |
| 4d | softmove | all soft macros, 16 candidates (GPU-batched when available) |
| 4e | long-shake | multi-scale (8 dir x 5 scale) relocation of hot macros - only in congestion ∈ [1.85, 2.0) |
| 4f | window-reorder | VNS-style K!-1 permutations of K-nearest soft macros |

Two more run only for Tier 2 (no effect on Tier-1 proxy):

- 4g Klein-4 orientation: for each hard macro, pick the N/FN/FS/S flip that
  minimizes incident-net HPWL, recorded to an `orientations.pt` sidecar.
  Non-destructive (90° rotations are disallowed; all four flips keep the footprint).
- 4h clearance push: pre-push hard macros >=12 µm apart so the placer
  controls the final Tier-2 PDN-channel coordinates instead of the evaluator's
  blind push. Auto-skips the tiny IBM canvases (`_clearance_applies` guards on
  "12 µm must be small vs the canvas").

---

## 5. The `CongState` incremental-scoring engine

`CongState` (a Python wrapper over `cong_state_create/apply/revert/commit` in
`congestion.c`) is what makes the whole operator stack affordable. It holds the
live routing/density/HPWL maps and supports:

- `score()` -> current exact proxy + breakdown,
- `apply(macros, x, y)` -> move macros, incrementally update only the affected
  bins/nets, return the new proxy (and stash one undo),
- `revert()` -> undo the last apply,
- `commit()` -> make it permanent,
- `rebuild()` -> full recompute to flush accumulated floating-point drift.

A single-macro probe costs ~0.5-2 ms incremental vs 5-15 ms for a full rebuild
(~30x faster). That's what lets a stage probe tens of thousands of candidate
moves inside its budget.

The drift caveat - and the discipline around it. After many `commit()`s the
incremental score drifts from a fresh recompute (density-formula edge cases + FP
accumulation). So every operator does two things: (1) `rebuild()` periodically
mid-pass, and (2) before keeping a stage's result, it runs a fresh-rebuild
verify (`_cc_call(n_iter=0)`) and reverts the whole stage if the honest score
doesn't actually beat the pre-stage proxy. This is why the code is littered with
`rebuild-verify … ; revert (skip polish)` branches - they prevent the placer from
fooling itself with stale incremental scores.

---

## 6. Robustness: never get disqualified

A large fraction of the code exists purely so the placer always returns one
valid placement - because a crash, an overlap, an out-of-bounds macro, or a
NaN coordinate is a guaranteed DQ, which is strictly worse than a mediocre legal
result. The defenses, from outer to inner:

- Per-trajectory crash guard (§5 below): any one best-of-3 trajectory can OOM
  or throw without killing the run; the validated `off` baseline always runs first.
- Legality gate on selection: a trajectory is only selectable if it is
  overlap-free and in-bounds. A lower-proxy but invalid trajectory is never shipped.
- `_best_effort_legal_tensor`: if every trajectory fails (e.g. no CUDA at
  all), build a guaranteed-legal placement from the input - spiral legalize, then
  pairwise clearance push, then a shelf-pack grid fallback if overlaps persist.
- Regression guard: if the full cascade somehow ends up worse than the
  post-refine baseline, ship the refine baseline instead.
- Unconditional legality repair: a final pairwise push if any residual hard
  overlap slipped through.
- Bounds clamp: clamp the float32 output so a sub-ULP rounding can't nudge a
  boundary macro outside the canvas.
- NaN/Inf sanitization: replace any non-finite coordinate with the (finite)
  input position; fall back to the legal builder if even that fails.

All of these are happy-path no-ops - on a healthy run the shipped placement is
byte-identical to what the cascade produced; the guards only fire on pathology.

---

## 7. The best-of-3 outer loop (RUDY trajectories)

`place()` re-enters itself up to three times with different congestion-loss
settings during GP (`off`, `rudy`, `rudy_hv`), and keeps the lowest final proxy
among the legal results (`XP_RUDY_BO2`, default on).

Why: the RUDY congestion loss improves the GP output on some designs but the
polish stage - co-tuned to the default-GP basin - can invert that gain into a
final regression on others, and which designs invert is unpredictable. Running
the full pipeline under each setting and keeping the best is non-regressing by
construction (`off` is always a candidate and always runs first), at ~3x wall
time - guarded by a wall-budget check so 3x can't blow the 1-hour cap.

---

## 8. Determinism

The placer pins seeds, sets `cudnn.deterministic`, forces deterministic CUDA
algorithms, and fixes `CUBLAS_WORKSPACE_CONFIG` before CUDA init. The basin-jump
loops deliberately replace wall-clock deadlines with iteration caps so the
output doesn't depend on host clock speed. The goal is bit-reproducible results
(important for verification and for the regression guards to behave consistently).

---

## 9. Engineering performance tricks (the "why it's fast enough" list)

- Two C cost engines + a GPU one, kept numerically aligned, so each phase
  uses the cheapest engine that fits.
- Incremental `CongState` scoring (~30x cheaper per probe).
- GPU independent-set batching in polish (~28x fewer kernel launches).
- Persistent `GPUState` built on a background thread during Stage 1 so its
  ~3.4 s setup overlaps useful work; refreshed in place rather than rebuilt.
- Persistent K-worker ILS subprocess pool (`ILSWorkerPool`,
  `_ils_polish_worker.py`): each worker owns its own libgomp runtime for true
  Kx parallel `cong_relax_v2` (a thread pool would share one OMP team). Reused
  across calls via length-prefixed pickle frames over stdin/stdout.
- Background threading of `.plc` load + LEF/DEF parse + C-lib preload,
  overlapping the GPU-bound GP.
- C extensions compiled at first run with `-O3 -march=native -flto` (and
  `-fopenmp` where parallel) - portable source, host-tuned binary.
- On-disk LEF/DEF cache so re-runs skip regeneration.
- `expandable_segments` CUDA allocator to survive 3 sequential pipelines on
  an 8 GB GPU without fragmenting into OOM.

---

## 10. Why this design wins (summary of the "why")

1. Optimize the real objective, not a surrogate. Reimplementing the exact
   proxy as a fast incremental oracle removes the surrogate/true-cost gap that
   limits most placers, and makes greedy accept/reject provably non-regressing.
2. Coarse-to-fine, multi-engine. Analytical GP gets global structure; Adam
   refinement settles the basin; GPU polish + basin-jump escape local minima;
   discrete operators do exact-scored fine cleanup. Each tool is used where it's
   strongest.
3. Self-tuning, not benchmark-tuning. One algorithm reads a per-design
   congestion measurement and adapts its knobs - general, not overfit (which is
   also a competition rule).
4. Multi-start where it pays. The 5-perturb polish and best-of-3 RUDY loop
   are cheap insurance against the pipeline landing in a bad basin, made
   non-regressing by always keeping a validated baseline.
5. DQ-safety is a first-class feature. Extensive legality/finiteness guards
   mean a valid placement is always returned - the difference between a low
   score and no score at all.
6. Tier-2 awareness for free. Orientation and clearance are handled in
   channels invisible to the Tier-1 proxy, so they help downstream PnR without
   costing proxy rank.

---

### File map (where to look)

| Concern | File(s) |
|---------|---------|
| Orchestration / `place()` | `placer.py` |
| Environment knobs | `TUNING.md` |
| Exact proxy + incremental `CongState` + polish | `extensions/congestion.c` |
| Analytical Adam refiner | `extensions/refine.c` |
| Spiral legalization | `extensions/legalize.c` |
| GPU proxy forward model | `GPU/forward.py` |
| GPU coordinate-descent polish | `GPU/polish.py`, `GPU/polish_hybrid_5perturb.py` |
| Independent-set batched eval | `GPU/is_partition.py`, `GPU/eval_is_batched.py`, `kernels/polish_emit.py` |
| Escape / basin-jump | `GPU/basin_jump_v3.py`, `GPU/basin_jump_v2.py` |
| Persistent GPU state | `GPU/state.py`, `GPU/init.py` |
| Parallel polish workers | `_ils_polish_worker.py` (pool in `placer.py`) |
| Xplace GP engine | `Xplace/` (vendored; `cpybin/*.so` fetched, see the README) |
