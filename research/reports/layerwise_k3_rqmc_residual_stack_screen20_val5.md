# Layerwise diagonal-K3 + randomized RQMC residual stack

Prediction form: `rqmc + a*(K2-rqmc) + b*(layerwiseDiagK3-K2)`.

## Command actually executed

```powershell
research\scripts\run_layerwise_k3_rqmc_residual_stack.py --dev-limit 20 --validation-limit 5 --result-tag screen20_val5
```

## Calibration

- `a`: `0.2885898`
- `b`: `0.41619309`
- equivalent RQMC/K2 lambda if `b=0`: `0.7114102`

## Result matrix

| split | method | final MSE | adjusted score | FLOPs | win vs current RQMC combo |
| --- | --- | ---: | ---: | ---: | ---: |
| development_selection | rqmc | 4.77811e-06 | 4.77811e-07 | 1.73225e+10 | nan |
| development_selection | current_k2_rqmc_combo | 3.86202e-06 | 3.86202e-07 | 1.89392e+10 | nan |
| development_selection | stack | 3.94755e-06 | 3.94755e-07 | 2.05587e+10 | 0.3 |
| validation_frozen | rqmc | 7.16022e-06 | 7.16022e-07 | 1.73225e+10 | nan |
| validation_frozen | current_k2_rqmc_combo | 5.39471e-06 | 5.39471e-07 | 1.89392e+10 | nan |
| validation_frozen | stack | 5.22746e-06 | 5.22746e-07 | 2.05587e+10 | 0.6 |

## Decision note

Promote only if validation adjusted score beats the current cubature-based K3 residual candidate.
