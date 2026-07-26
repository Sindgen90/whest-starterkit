# Layerwise diagonal-K3 residual stack

This tests source-layer-aware diagonal K3: one diagonal third-cumulant cube per source layer, no generic off-diagonal tensor factors.

## Command actually executed

```powershell
research\scripts\run_layerwise_diagonal_k3_residual_stack.py --dev-limit 20 --validation-limit 5 --third-scales -0.3 -1.0 --max-k3-layers 8 --generic-horizons full 2 --result-tag screen20_val5_s03_s1_l8
```

## Result matrix

| config | split | method | final MSE | adjusted score | FLOPs | win vs current K2 combo | a | b |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scale=-0.3,L=8,full | development_selection | stack | 4.04032e-06 | 4.3242e-07 | 2.91111e+10 | 0.6 | 0.138099 | -3.23377 |
| scale=-0.3,L=8,full | validation_frozen | stack | 3.2801e-06 | 3.51056e-07 | 2.91111e+10 | 0.8 | 0.138099 | -3.23377 |
| scale=-0.3,L=8,H2 | development_selection | stack | 4.04052e-06 | 4.32412e-07 | 2.91091e+10 | 0.6 | 0.138101 | -3.23459 |
| scale=-0.3,L=8,H2 | validation_frozen | stack | 3.27999e-06 | 3.51021e-07 | 2.91091e+10 | 0.8 | 0.138101 | -3.23459 |
| scale=-1,L=8,full | development_selection | stack | 4.041e-06 | 4.32492e-07 | 2.91111e+10 | 0.6 | 0.138099 | -0.97038 |
| scale=-1,L=8,full | validation_frozen | stack | 3.27999e-06 | 3.51045e-07 | 2.91111e+10 | 0.8 | 0.138099 | -0.97038 |
| scale=-1,L=8,H2 | development_selection | stack | 4.0412e-06 | 4.32484e-07 | 2.91091e+10 | 0.6 | 0.1381 | -0.970625 |
| scale=-1,L=8,H2 | validation_frozen | stack | 3.27988e-06 | 3.51009e-07 | 2.91091e+10 | 0.8 | 0.1381 | -0.970625 |
