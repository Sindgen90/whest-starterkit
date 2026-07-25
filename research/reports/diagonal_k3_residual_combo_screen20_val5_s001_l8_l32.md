# Cheap diagonal-K3 residual combo

This tests a cheap K3 residual proxy: retain only per-neuron marginal third cumulants, propagate them with `W^3`, and use them as a first-order Edgeworth mean correction. No generic K3 factors are used.

## Command actually executed

```powershell
research\scripts\run_diagonal_k3_residual_combo.py --dev-limit 20 --validation-limit 5 --third-scales -0.01 --max-k3-layers 8 32 --result-tag screen20_val5_s001_l8_l32
```

## Result matrix

| config | split | method | final MSE | adjusted score | FLOPs | win vs cubature | lambda |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| scale=-0.01,L=8 | development_selection | combo | 2.86529e-06 | 2.89604e-07 | 2.74919e+10 | 0.7 | 0.853877 |
| scale=-0.01,L=8 | validation_frozen_lambda | combo | 3.22377e-06 | 3.25836e-07 | 2.74919e+10 | 0.4 | 0.853877 |
| scale=-0.01,L=32 | development_selection | combo | 2.86543e-06 | 2.89684e-07 | 2.74982e+10 | 0.7 | 0.85387 |
| scale=-0.01,L=32 | validation_frozen_lambda | combo | 3.22409e-06 | 3.25943e-07 | 2.74982e+10 | 0.4 | 0.85387 |

## Decision note

Promote only if validation adjusted score beats the current K2+cubature submission candidate (`3.838e-7`) or if raw MSE improves enough to justify a separate experimental submission.
