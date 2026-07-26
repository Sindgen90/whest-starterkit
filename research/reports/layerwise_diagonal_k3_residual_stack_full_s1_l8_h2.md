# Layerwise diagonal-K3 residual stack

This tests source-layer-aware diagonal K3: one diagonal third-cumulant cube per source layer, no generic off-diagonal tensor factors.

## Command actually executed

```powershell
research\scripts\run_layerwise_diagonal_k3_residual_stack.py --third-scales -1.0 --max-k3-layers 8 --generic-horizons 2 --result-tag full_s1_l8_h2
```

## Result matrix

| config | split | method | final MSE | adjusted score | FLOPs | win vs current K2 combo | a | b |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scale=-1,L=8,H2 | development_selection | stack | 2.94145e-06 | 3.1479e-07 | 2.91091e+10 | 0.667 | 0.106534 | -0.791397 |
| scale=-1,L=8,H2 | validation_frozen | stack | 3.34836e-06 | 3.58338e-07 | 2.91091e+10 | 0.65 | 0.106534 | -0.791397 |

## Decision

Layerwise diagonal K3 works, but does not beat the simpler collapsed diagonal-K3 residual stack.

| candidate | validation final MSE | adjusted score | FLOPs |
| --- | ---: | ---: | ---: |
| collapsed diagonal-K3 residual stack | 3.34838e-06 | 3.58334e-07 | 2.91087e+10 |
| layerwise diagonal-K3 residual stack | 3.34836e-06 | 3.58338e-07 | 2.91091e+10 |

The two estimates are effectively tied in raw MSE.  The layerwise version is fractionally worse on adjusted score because it spends slightly more FLOPs.  This suggests the source-layer diagonal cohorts are nearly collinear with the collapsed diagonal-K3 correction under the current two-scalar residual fit.

Do not package this as a separate submission unless we later need the source-layer instrumentation.  Keep `submissions/diag_k3_residual_stack_s03_l8.tar.gz` as the current K3-flavored submission candidate.
