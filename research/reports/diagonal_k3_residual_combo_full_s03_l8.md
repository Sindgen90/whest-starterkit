# Cheap diagonal-K3 residual combo

This tests a cheap K3 residual proxy: retain only per-neuron marginal third cumulants, propagate them with `W^3`, and use them as a first-order Edgeworth mean correction. No generic K3 factors are used.

## Command actually executed

```powershell
research\scripts\run_diagonal_k3_residual_combo.py --third-scales -0.3 --max-k3-layers 8 --result-tag full_s03_l8
```

## Result matrix

| config | split | method | final MSE | adjusted score | FLOPs | win vs cubature | lambda |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| scale=-0.3,L=8 | development_selection | combo | 3.09326e-06 | 3.12645e-07 | 2.74919e+10 | 0.8 | 0.884549 |
| scale=-0.3,L=8 | validation_frozen_lambda | combo | 3.83568e-06 | 3.87685e-07 | 2.74919e+10 | 0.6 | 0.884549 |

## Decision note

Promote only if validation adjusted score beats the current K2+cubature submission candidate (`3.838e-7`) or if raw MSE improves enough to justify a separate experimental submission.
