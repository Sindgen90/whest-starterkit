# Conditional-Gaussian final-layer RQMC

The only mathematical change from ordinary 24,576-point RQMC is replacing
`mean(ReLU(z_L))` by the Gaussian ReLU expectation fitted from the sampled
final-preactivation mean and variance.

## Command actually executed

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe research\scripts\run_conditional_gaussian_rqmc.py --partition development_selection
```

## Result matrix

| method | final MSE | adjusted score | FLOPs | win vs ordinary RQMC | win vs K3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary RQMC 24576 | 1.04985e-06 | 4.01161e-07 | 1.03935e+11 | -- | -- |
| diagonal-K3 residual stack | 2.94144e-06 | 3.14783e-07 | 2.91086e+10 | -- | -- |
| conditional-Gaussian RQMC 24576 | 1.9294e-06 | 7.37341e-07 | 1.03948e+11 | 0.000 | 0.667 |

## Decision

Reject this correction. It increased mean final-layer MSE by
`8.79559e-07` (83.78%) relative to ordinary 24,576-point RQMC and lost on
all 30 paired development MLPs. Its adjusted score was 83.81% worse than
ordinary RQMC and 134.24% worse than the diagonal-K3 residual stack.

The failure is consistent with moment-closure bias: the final preactivations
are not Gaussian enough for their first two marginal moments to determine the
ReLU mean. This systematic bias overwhelms any variance reduction from
replacing the sampled final ReLU average.

The validation gate did not pass. No validation data was evaluated and no
submission package was created.
