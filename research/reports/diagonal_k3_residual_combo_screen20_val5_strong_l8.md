# Cheap diagonal-K3 residual combo

This tests a cheap K3 residual proxy: retain only per-neuron marginal third cumulants, propagate them with `W^3`, and use them as a first-order Edgeworth mean correction. No generic K3 factors are used.

## Command actually executed

```powershell
research\scripts\run_diagonal_k3_residual_combo.py --dev-limit 20 --validation-limit 5 --third-scales -0.03 -0.1 -0.3 --max-k3-layers 8 --result-tag screen20_val5_strong_l8
```

## Result matrix

| config | split | method | final MSE | adjusted score | FLOPs | win vs cubature | lambda |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| scale=-0.03,L=8 | development_selection | combo | 2.86465e-06 | 2.89539e-07 | 2.74919e+10 | 0.7 | 0.854228 |
| scale=-0.03,L=8 | validation_frozen_lambda | combo | 3.22345e-06 | 3.25804e-07 | 2.74919e+10 | 0.4 | 0.854228 |
| scale=-0.1,L=8 | development_selection | combo | 2.86267e-06 | 2.89338e-07 | 2.74919e+10 | 0.7 | 0.855477 |
| scale=-0.1,L=8 | validation_frozen_lambda | combo | 3.22233e-06 | 3.25691e-07 | 2.74919e+10 | 0.4 | 0.855477 |
| scale=-0.3,L=8 | development_selection | combo | 2.85903e-06 | 2.88971e-07 | 2.74919e+10 | 0.7 | 0.859207 |
| scale=-0.3,L=8 | validation_frozen_lambda | combo | 3.21907e-06 | 3.25361e-07 | 2.74919e+10 | 0.4 | 0.859207 |

## Decision note

Promote only if validation adjusted score beats the current K2+cubature submission candidate (`3.838e-7`) or if raw MSE improves enough to justify a separate experimental submission.
