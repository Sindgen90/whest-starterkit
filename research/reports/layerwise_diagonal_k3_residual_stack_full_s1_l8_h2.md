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

Layerwise diagonal K3 works and has a tiny raw-MSE edge over the simpler collapsed diagonal-K3 residual stack, but it does not beat it on the challenge's adjusted score.

| candidate | validation final MSE | adjusted score | FLOPs |
| --- | ---: | ---: | ---: |
| collapsed diagonal-K3 residual stack | 3.34838e-06 | 3.58334e-07 | 2.91087e+10 |
| layerwise diagonal-K3 residual stack | 3.34836e-06 | 3.58338e-07 | 2.91091e+10 |

The layerwise version improves validation raw MSE by about `1.85e-11` absolute, roughly `0.00055%` relative.  It is fractionally worse on adjusted score because it spends slightly more FLOPs.  This suggests the source-layer diagonal cohorts are nearly collinear with the collapsed diagonal-K3 correction under the current two-scalar residual fit.

For the leaderboard objective, keep `submissions/diag_k3_residual_stack_s03_l8.tar.gz` as the current K3-flavored submission candidate.  If we want a raw-MSE-focused experimental package, the layerwise version is defensible, but it should not replace the collapsed version as the primary adjusted-score candidate.

## Raw-MSE experimental package

Because the layerwise version has a tiny validation raw-MSE edge, it was packaged as a secondary experimental candidate.

Estimator:

`submissions/layerwise_diag_k3_residual_stack_s1_l8_h2/estimator.py`

Package:

`submissions/layerwise_diag_k3_residual_stack_s1_l8_h2.tar.gz`

Exact submission-estimator validation-manifest result:

| final MSE | adjusted score | FLOPs |
| ---: | ---: | ---: |
| 3.348361953063994e-06 | 3.5833909899594703e-07 | 29109229018 |

Validation commands executed:

```powershell
$env:PYTHONIOENCODING='utf-8'; C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe validate --estimator submissions\layerwise_diag_k3_residual_stack_s1_l8_h2\estimator.py --format plain
```

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe research\scripts\evaluate_estimator_on_manifest.py --estimator-path submissions\layerwise_diag_k3_residual_stack_s1_l8_h2\estimator.py --partition validation --output research\results\layerwise_diag_k3_residual_stack_s1_l8_h2_validation_manifest_eval.json
```

```powershell
$env:PYTHONIOENCODING='utf-8'; C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe package --estimator submissions\layerwise_diag_k3_residual_stack_s1_l8_h2\estimator.py --output submissions\layerwise_diag_k3_residual_stack_s1_l8_h2.tar.gz --yes --format plain
```

```powershell
$env:PYTHONIOENCODING='utf-8'; C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe validate-package submissions\layerwise_diag_k3_residual_stack_s1_l8_h2.tar.gz --format plain
```

All checks passed.
