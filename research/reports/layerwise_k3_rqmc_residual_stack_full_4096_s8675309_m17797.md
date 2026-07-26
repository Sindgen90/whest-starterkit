# Layerwise diagonal-K3 + randomized RQMC residual stack

Prediction form: `rqmc + a*(K2-rqmc) + b*(layerwiseDiagK3-K2)`.

## Command actually executed

```powershell
research\scripts\run_layerwise_k3_rqmc_residual_stack.py --result-tag full_4096_s8675309_m17797
```

## Calibration

- `a`: `0.20713219`
- `b`: `-0.17712045`
- equivalent RQMC/K2 lambda if `b=0`: `0.79286781`

## Result matrix

| split | method | final MSE | adjusted score | FLOPs | win vs current RQMC combo |
| --- | --- | ---: | ---: | ---: | ---: |
| development_selection | rqmc | 5.58236e-06 | 5.58236e-07 | 1.73225e+10 | nan |
| development_selection | current_k2_rqmc_combo | 4.48921e-06 | 4.48921e-07 | 1.89392e+10 | nan |
| development_selection | stack | 4.55422e-06 | 4.55422e-07 | 2.05587e+10 | 0.367 |
| validation_frozen | rqmc | 6.01264e-06 | 6.01264e-07 | 1.73225e+10 | nan |
| validation_frozen | current_k2_rqmc_combo | 4.72079e-06 | 4.72079e-07 | 1.89392e+10 | nan |
| validation_frozen | stack | 4.77722e-06 | 4.77722e-07 | 2.05587e+10 | 0.35 |

## Decision note

Do not package this candidate.

The layerwise-K3 residual does not help the randomized RQMC anchor on the full validation split:

| candidate | validation final MSE | adjusted score | FLOPs |
| --- | ---: | ---: | ---: |
| current K2 + RQMC combo | 4.72079e-06 | 4.72079e-07 | 1.89392e+10 |
| layerwise K3 + RQMC stack | 4.77722e-06 | 4.77722e-07 | 2.05587e+10 |
| collapsed diagonal-K3 + cubature stack | 3.34838e-06 | 3.58334e-07 | 2.91087e+10 |
| layerwise diagonal-K3 + cubature stack | 3.34836e-06 | 3.58339e-07 | 2.91092e+10 |

The K3 residual is useful with the signed-Hadamard cubature anchor, but not with this randomized RQMC lattice anchor.  Because the RQMC stack is worse than both the current K2+RQMC combo and the cubature-based K3 stacks, no submission package was created.
