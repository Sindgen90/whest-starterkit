# Phase 1 experiment protocol

This directory holds reproducible Phase 1 experiment records. The bundled starter examples remain unchanged; experiment code imports them rather than copying or editing them.

## Whole-MLP splits

| Internal name | Official split | MLPs | Use |
| --- | --- | ---: | --- |
| development | `mini` | 100 | algorithm iteration and paired comparisons |
| validation | `full` | 1,000 | frozen-candidate validation only |

`mini` and `full` have disjoint MLP seeds. The runner defaults to development; validation requires an explicit opt-in so it cannot become a tuning loop.

## Commands

```powershell
uv sync
uv run python experiments/run_baselines.py --baseline all --split development
uv run whest validate --estimator estimator.py
uv run whest package --estimator estimator.py
```

The runner writes one immutable JSON record per estimator and a compact baseline table. Records retain paired MLP IDs and log final/all-layer MSE, prefix MSE, FLOPs, runtime, configuration, source hash, and failures.
