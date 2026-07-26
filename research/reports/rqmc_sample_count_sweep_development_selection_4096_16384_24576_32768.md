# RQMC sample-count sweep

Only the sample count changes. Each candidate is a complete independently constructed
rank-1 lattice using the same shift rule and generator multiplier.

## Command actually executed

```powershell
research\scripts\run_rqmc_sample_count_sweep.py --partition development_selection --sample-counts 4096 16384 24576 32768
```

## Result matrix

| partition | samples | final MSE | adjusted score | FLOPs | runtime | win vs 4096 | win vs K3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| development_selection | 4096 | 5.58236e-06 | 5.58236e-07 | 1.73225e+10 | 0.365s | -- | 0.267 |
| development_selection | 16384 | 2.12435e-06 | 5.41163e-07 | 6.929e+10 | 1.56s | 0.833 | 0.633 |
| development_selection | 24576 | 1.04985e-06 | 4.01161e-07 | 1.03935e+11 | 2.32s | 0.967 | 0.900 |
| development_selection | 32768 | 9.20651e-07 | 4.69058e-07 | 1.3858e+11 | 3.12s | 1.000 | 0.967 |

No convex combination, control variate, K2 correction, or K3 correction is included.

## Decision

Increasing the sample count works in raw MSE but does not beat the current
diagonal-K3 residual stack in adjusted score.

- Best raw MSE: `32768` samples at `9.20651e-7`.
- Best adjusted score: `24576` samples at `4.01161e-7`.
- Current K3 development adjusted score: `3.14783e-7`.
- The `24576` design improves raw MSE by `2.80x` versus K3 but spends enough
  additional FLOPs to be `1.27x` worse on adjusted score.

The frozen-validation gate therefore failed. Validation was not opened and no
submission was packaged. The next justified change is a variance-reduction
mechanism at roughly the `16384`--`24576` compute level, not another broad
sample-count sweep.
