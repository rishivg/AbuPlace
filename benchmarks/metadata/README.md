# Benchmark Metadata

This directory contains metadata about benchmarks, including baseline scores.

## baseline_scores.json

Contains initial placement baseline scores for all competition benchmarks.

### Format

```json
{
  "benchmark_name": {
    "num_macros": int,
    "num_nets": int,
    "canvas_width": float,
    "canvas_height": float,
    "initial_placement": {
      "proxy_cost": float,
      "wirelength_cost": float,
      "density_cost": float,
      "congestion_cost": float,
      "overlap_count": int,
      "total_overlap_area": float
    }
  }
}
```

### Example

```json
{
  "ariane133": {
    "num_macros": 133,
    "num_nets": 22584,
    "canvas_width": 1433.406,
    "canvas_height": 1433.406,
    "initial_placement": {
      "proxy_cost": 0.7109,
      "wirelength_cost": 0.0497,
      "density_cost": 0.6067,
      "congestion_cost": 0.7157,
      "overlap_count": 0,
      "total_overlap_area": 0.0
    }
  }
}
```

### Usage

```python
import json

# Load baseline scores
with open('benchmarks/metadata/baseline_scores.json') as f:
    baselines = json.load(f)

# Get baseline for a specific benchmark
ariane133_baseline = baselines['ariane133']['initial_placement']['proxy_cost']
print(f"Ariane133 baseline: {ariane133_baseline:.4f}")
```

### Updating Baselines

To recompute baseline scores:

```bash
python scripts/compute_initial_baselines.py
```

This will evaluate the initial placements and update `baseline_scores.json`.

## Future Extensions

Additional metadata files may include:
- `sa_baselines.json` - Simulated Annealing baseline scores
- `replace_baselines.json` - RePlAce baseline scores
- `submission_results.json` - Competition submission scores
