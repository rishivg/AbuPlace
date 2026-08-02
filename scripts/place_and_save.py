#!/usr/bin/env python3
"""Run a placer on an NG45 benchmark and save the resulting positions as
a .pt tensor that scripts/evaluate_with_orfs.py can consume via
``--placement``.

Usage
-----
    # Run AbuPlace on ariane133_ng45 and save under output/placements/
    uv run python scripts/place_and_save.py \\
        --placer abuplace/placer.py \\
        --benchmark ariane133_ng45

    # All NG45 designs in one shot
    uv run python scripts/place_and_save.py \\
        --placer abuplace/placer.py --all

    # Custom output dir + custom file naming
    uv run python scripts/place_and_save.py \\
        --placer abuplace/placer.py \\
        --benchmark nvdla_ng45 \\
        --out-dir my_runs/abuplace

Output
------
    output/placements/<placer_stem>_<benchmark>.pt
        torch.Tensor of shape [num_macros, 2] - the centers chosen by the
        placer. This is the file format expected by:
            evaluate_with_orfs.py --placement <this file>
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement


NG45_BENCHMARKS = [
    "ariane133_ng45",
    "ariane136_ng45",
    "bp_quad_ng45",
    "nvdla_ng45",
    "mempool_tile_ng45",
]


def _load_placer(path: Path):
    """Mirror macro_place.evaluate._load_placer: import the file and
    instantiate the first class that defines a ``place`` method."""
    path = path.resolve()
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None:
        raise RuntimeError(f"Failed to load placer from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for attr in vars(mod).values():
        if (
            isinstance(attr, type)
            and attr.__module__ == path.stem
            and callable(getattr(attr, "place", None))
        ):
            return attr()

    raise RuntimeError(
        f"No placer class found in {path}. "
        "Expected a class with a place(self, benchmark) -> Tensor method."
    )


def _benchmark_path(name: str) -> Path:
    """Resolve a benchmark name to its preprocessed .pt under
    benchmarks/processed/public/."""
    return REPO_ROOT / "benchmarks" / "processed" / "public" / f"{name}.pt"


def _try_load_plc_for(name: str):
    """Best-effort plc load mirroring scripts/evaluate_with_orfs.py paths.

    Returns the plc object or None. Used only for the optional proxy print.
    """
    from macro_place.loader import load_benchmark_from_dir
    candidates = []
    if name.endswith("_ng45"):
        base = name[:-len("_ng45")]
        if base == "bp_quad":
            candidates.append(REPO_ROOT / "external/MacroPlacement/CodeElements/SimulatedAnnealingGWTW/test/bp_ng45")
        candidates.append(REPO_ROOT / f"external/MacroPlacement/Flows/NanGate45/{base}/netlist/output_CT_Grouping")
    elif name.endswith("_asap7"):
        base = name[:-len("_asap7")]
        candidates.append(REPO_ROOT / f"external/MacroPlacement/Flows/ASAP7/{base}/netlist/output_CT_Grouping")
    candidates.append(REPO_ROOT / f"external/MacroPlacement/Testcases/ICCAD04/{name}")
    for d in candidates:
        if d.exists():
            try:
                _, plc = load_benchmark_from_dir(str(d))
                return plc
            except Exception:
                continue
    return None


def _save_placement(placement: torch.Tensor, benchmark: Benchmark, out_path: Path):
    """Validate shape/dtype and persist as a CPU float32 tensor."""
    if not torch.is_tensor(placement):
        placement = torch.as_tensor(placement)
    placement = placement.detach().cpu()
    expected = (benchmark.num_macros, 2)
    if tuple(placement.shape) != expected:
        raise RuntimeError(
            f"placement shape {tuple(placement.shape)} != {expected}; "
            "evaluate_with_orfs expects [num_macros, 2]"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(placement.to(torch.float32).contiguous(), out_path)


def run_one(placer, name: str, out_dir: Path, placer_stem: str) -> dict:
    pt_file = _benchmark_path(name)
    if not pt_file.exists():
        raise FileNotFoundError(
            f"Benchmark .pt not found: {pt_file}. "
            f"Available NG45 benchmarks: {NG45_BENCHMARKS}"
        )
    benchmark = Benchmark.load(str(pt_file))
    print(f"\n=== {name} === macros={benchmark.num_macros}")

    t0 = time.time()
    placement = placer.place(benchmark)
    wall = time.time() - t0

    # Save the placement FIRST. The proxy print below is informational and
    # requires a plc object we may not have loaded - its failure must not
    # discard a real placement.
    out_path = out_dir / f"{placer_stem}_{name}.pt"
    _save_placement(placement, benchmark, out_path)
    try:
        rel = out_path.relative_to(REPO_ROOT)
    except ValueError:
        rel = out_path
    print(f"  saved -> {rel} ({wall:.1f}s)")

    # Tier-2 Klein-4 orientation sidecar: int8[num_hard_macros] codes
    # (0=N, 1=FN, 2=FS, 3=S). evaluate_with_orfs.py applies these to plc nodes
    # before generating the macro-placement TCL. Optional - absent => all N.
    orients = getattr(placer, "_last_orientations", None)
    if orients is not None:
        orient_path = out_path.with_name(out_path.stem + "_orientations.pt")
        torch.save(torch.as_tensor(orients, dtype=torch.int8).cpu(), orient_path)
        try:
            orel = orient_path.relative_to(REPO_ROOT)
        except ValueError:
            orel = orient_path
        print(f"  saved -> {orel} (orientations sidecar)")

    # Validate shape / overlap / canvas.
    is_valid, viols = validate_placement(placement, benchmark)
    if not is_valid:
        print(f"  WARNING: placement is invalid ({len(viols)} violations)")
        for v in viols[:5]:
            print(f"    - {v}")

    # Optional proxy print - needs a plc. Try to load one the same way
    # scripts/evaluate_with_orfs.py does for the matching NG45 source dir;
    # silently skip if we can't.
    proxy_val = None
    plc = _try_load_plc_for(name)
    if plc is not None:
        try:
            costs = compute_proxy_cost(placement, benchmark, plc)
            proxy_val = float(costs["proxy_cost"])
            print(
                f"  proxy={proxy_val:.6f}  "
                f"wl={costs['wirelength_cost']:.3f}  "
                f"den={costs['density_cost']:.3f}  "
                f"cong={costs['congestion_cost']:.3f}  "
                f"overlaps={costs['overlap_count']}"
            )
        except Exception as e:
            print(f"  (proxy compute skipped: {type(e).__name__}: {e})")
    else:
        print(f"  (proxy compute skipped: no plc found for {name})")

    return {
        "benchmark": name,
        "proxy_cost": proxy_val,
        "valid": is_valid,
        "runtime_s": wall,
        "out_path": str(out_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run a placer on NG45 designs and save the macro positions "
                    "as a .pt tensor for tier-2 (ORFS) evaluation."
    )
    parser.add_argument(
        "--placer",
        type=Path,
        required=True,
        help="Path to a placer .py file (e.g. abuplace/placer.py).",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        help=f"One NG45 benchmark name. Choices: {NG45_BENCHMARKS}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all NG45 benchmarks sequentially.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "placements",
        help="Output directory for .pt files (default: output/placements).",
    )
    args = parser.parse_args()

    if not args.placer.exists():
        print(f"ERROR: placer file not found: {args.placer}")
        return 1
    if not (args.benchmark or args.all):
        print("ERROR: pass --benchmark <name> or --all")
        return 1

    benchmarks = NG45_BENCHMARKS if args.all else [args.benchmark]
    if not args.all and args.benchmark not in NG45_BENCHMARKS:
        print(
            f"ERROR: --benchmark {args.benchmark!r} not in NG45 set "
            f"{NG45_BENCHMARKS}"
        )
        return 1

    print(f"Loading placer from {args.placer}…")
    placer = _load_placer(args.placer)
    print(f"  placer class: {type(placer).__name__}")
    placer_stem = args.placer.stem  # 'placer' for submissions/<x>/placer.py
    # Disambiguate by parent dir when stem is generic.
    if placer_stem in ("placer", "main"):
        placer_stem = args.placer.parent.name

    results = []
    for name in benchmarks:
        try:
            results.append(run_one(placer, name, args.out_dir, placer_stem))
        except Exception as e:
            print(f"  FAILED on {name}: {e}")
            results.append({"benchmark": name, "error": str(e)})

    print("\n" + "=" * 72)
    print(f"{'Benchmark':<22} {'Proxy':>10} {'Valid':>7} {'Wall (s)':>10}")
    print("-" * 72)
    for r in results:
        if "error" in r:
            print(f"{r['benchmark']:<22} {'ERROR':>10}  {r['error']}")
        else:
            print(
                f"{r['benchmark']:<22} "
                f"{r['proxy_cost']:>10.6f} "
                f"{'yes' if r['valid'] else 'NO':>7} "
                f"{r['runtime_s']:>10.1f}"
            )
    print()
    try:
        _od = args.out_dir.relative_to(REPO_ROOT)
    except ValueError:
        _od = args.out_dir
    print(f"Placements saved to: {_od}/")
    print("\nNext step - run tier-2 ORFS flow with these placements:")
    for r in results:
        if "error" not in r:
            try:
                rel = Path(r["out_path"]).relative_to(REPO_ROOT)
            except ValueError:
                rel = Path(r["out_path"])
            print(
                f"  uv run python scripts/evaluate_with_orfs.py "
                f"--benchmark {r['benchmark']} --placement {rel}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
