# Oracle activation-update diagnostic

This isolates the scalar ReLU update error using sampled preactivation cumulants.

## Command actually executed

```powershell
research\scripts\run_activation_update_oracle_diagnostic.py --partition development --limit 8 --samples 8192 --neurons 32 --layers 4 8 16 24 32 --saddlepoint-nodes 48
```

## Mean squared error against empirical E[ReLU(Z)]

| layer | method | MSE | MAE | win vs Gaussian | failure fraction |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | gaussian | 5.28409e-06 | 0.00183601 | nan | 0 |
| 8 | gaussian | 4.76092e-06 | 0.00169956 | nan | 0 |
| 16 | gaussian | 3.10686e-06 | 0.00106202 | nan | 0 |
| 24 | gaussian | 1.97539e-06 | 0.000710063 | nan | 0 |
| 32 | gaussian | 7.90906e-07 | 0.000384358 | nan | 0 |
| 4 | edgeworth_k3 | 1.0989e-06 | 0.00080019 | 0.758 | 0 |
| 8 | edgeworth_k3 | 7.42316e-07 | 0.000662637 | 0.766 | 0 |
| 16 | edgeworth_k3 | 3.70466e-07 | 0.000429907 | 0.527 | 0 |
| 24 | edgeworth_k3 | 2.71503e-07 | 0.000320586 | 0.371 | 0 |
| 32 | edgeworth_k3 | 1.0648e-07 | 0.000174813 | 0.262 | 0 |
| 4 | edgeworth_k34 | 3.34444e-07 | 0.000449152 | 0.848 | 0 |
| 8 | edgeworth_k34 | 1.27688e-07 | 0.000244047 | 0.953 | 0 |
| 16 | edgeworth_k34 | 7.46925e-08 | 0.000157064 | 0.895 | 0 |
| 24 | edgeworth_k34 | 3.70535e-08 | 9.02887e-05 | 0.828 | 0 |
| 32 | edgeworth_k34 | 2.40957e-08 | 5.83056e-05 | 0.672 | 0 |
| 4 | saddlepoint_k34 | 5.82492e-07 | 0.000593332 | 0.835 | 0.0781 |
| 8 | saddlepoint_k34 | 8.89951e-07 | 0.000618126 | 0.821 | 0.0156 |
| 16 | saddlepoint_k34 | 3.29517e-06 | 0.00103871 | 0.693 | 0.00781 |
| 24 | saddlepoint_k34 | 3.0527e-06 | 0.00108349 | 0.592 | 0.00391 |
| 32 | saddlepoint_k34 | 3.30143e-06 | 0.00110957 | 0.508 | 0.0234 |

## Decision rule

If saddlepoint does not beat Gaussian/Edgeworth with oracle cumulants, do not wire it into propagation.  If Edgeworth K3/K4 is also unstable, preserve the current Gaussian/K2 update for robustness.
