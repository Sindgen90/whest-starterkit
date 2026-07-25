# Finite-memory K3 + signed Hadamard cubature combo

This tests whether the K3-only finite-memory analytic estimator improves the current cubature blend enough to justify its extra charged FLOPs.

## Command actually executed

```powershell
research\scripts\run_k3_cubature_combo.py --third-scale=-0.001 --generic-horizon 2 --max-k3-layer 4
```

## K3 configuration

- `third_scale`: `-0.001`
- `generic_horizon`: `2`
- `max_k3_layer`: `4`

Calibrated lambda: `0.880905` (`raw=0.880905`)

## Result matrix

| split | method | final MSE | adjusted score | FLOPs | win vs cubature |
| --- | --- | ---: | ---: | ---: | ---: |
| development_selection | finite_memory_k3 | 2.09152e-05 | 2.09152e-06 | 3.84459e+09 | nan |
| development_selection | cubature | 3.6341e-06 | 3.6341e-07 | 2.58729e+10 | nan |
| development_selection | combo | 3.06029e-06 | 3.34354e-07 | 2.97175e+10 | 0.8 |
| validation_frozen_lambda | finite_memory_k3 | 2.67884e-05 | 2.67884e-06 | 3.84459e+09 | nan |
| validation_frozen_lambda | cubature | 4.1252e-06 | 4.1252e-07 | 2.58729e+10 | nan |
| validation_frozen_lambda | combo | 3.80356e-06 | 4.1556e-07 | 2.97175e+10 | 0.6 |
