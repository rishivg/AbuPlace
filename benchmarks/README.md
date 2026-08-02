# Benchmarks

This directory contains pre-processed benchmark data in PyTorch tensor format.

## Structure

```
benchmarks/
├── processed/
│   └── public/           # Public benchmarks (for competition)
│       ├── ariane133_ng45.pt
│       ├── ariane136_ng45.pt
│       ├── nvdla_ng45.pt
│       └── mempool_tile_ng45.pt
└── metadata/
    └── baseline_scores.json  # Initial placement baseline scores
```

## Benchmark Details

All benchmarks are modern chip designs in NanGate45 technology (45nm):

| Benchmark | Design Type | Macros | Nets | Canvas (mm) | Initial Score |
|-----------|-------------|--------|------|-------------|---------------|
| ariane133 | RISC-V Processor | 133 | 22,584 | 1.43x1.43 | 0.7109 |
| ariane136 | RISC-V Processor | 136 | 23,067 | 1.45x1.45 | 0.7097 |
| nvdla | AI Accelerator | 128 | 40,606 | 2.13x2.13 | 0.7569 |
| mempool_tile | Memory Arch. | 20 | 32,944 | 0.89x0.89 | 0.9610 |

## Loading Benchmarks

```python
from benchmark import Benchmark

# Load a benchmark
benchmark = Benchmark.load('benchmarks/processed/public/ariane133_ng45.pt')

print(f"Design: {benchmark.name}")
print(f"Macros: {benchmark.num_macros}")
print(f"Nets: {benchmark.num_nets}")
print(f"Canvas: {benchmark.canvas_width:.2f} x {benchmark.canvas_height:.2f} mm")
```

## Source Data

These benchmarks are sourced from the TILOS MacroPlacement repository:
- Location: `external/MacroPlacement/Flows/NanGate45/*/netlist/output_CT_Grouping/`
- Format: Protocol buffer (netlist.pb.txt) + initial placement (initial.plc)
- Processing: Converted to PyTorch tensors via `scripts/convert_ibm_benchmarks.py`
  (ICCAD04) and `scripts/convert_asap7_benchmarks.py` (ASAP7/NG45)

## Baseline Scores

Initial placement baseline scores are stored in `metadata/baseline_scores.json`.

These scores represent expert-designed placements and serve as the baseline to beat in the competition.

To win the $20K prize, your algorithm must achieve a higher aggregate score than these baselines.
