# Learned positive 4k empirical cubature prototype

A fixed 24,576-point Gaussian lattice pool was evaluated on even
development-calibration MLPs. Sixteen deterministic final-neuron snapshots
per MLP formed the training function family. A rank-256 basis, all coordinate
first moments, and all diagonal second moments determined a positive weighted
subset of at most 4,096 nodes. Odd development-selection MLPs were unseen
during rule construction.

## Command actually executed

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe research\scripts\run_learned_cubature_4k.py
```

## Result matrix

| method | split | final MSE | adjusted score | FLOPs |
| --- | --- | ---: | ---: | ---: |
| direct RQMC 4096 | selection | 5.58236e-06 | 5.58236e-07 | 1.73225e+10 |
| diagonal-K3 stack | selection | 2.94144e-06 | 3.14783e-07 | 2.91086e+10 |
| fixed 24k teacher | selection | 1.2838e-06 | -- | 1.03935e+11 |
| selected nodes, uniform weights | selection | 1.239e-05 | 1.239e-06 | 1.66908e+10 |
| learned positive cubature | calibration | 4.57141e-06 | 4.57141e-07 | 1.66908e+10 |
| learned positive cubature | selection | 1.18473e-05 | 1.18473e-06 | 1.66908e+10 |

## Learned rule diagnostics

- Nonzero support: `3939`.
- Effective sample size: `3237.08`.
- Largest weight: `0.000681975`.
- Weight solver: `exact_active_set`.
- Maximum standardized training/moment residual: `8.99281e-15`.
- Maximum absolute Gaussian mean error: `3.85935e-16`.
- Maximum absolute diagonal second-moment error: `9.88765e-16`.

## Decision

The learned rule has selection adjusted score `1.18473e-6`, +276.36% versus
K3, and wins 0.133 of MLPs versus direct 4k RQMC. It is 112.23% worse than
direct 4k RQMC and 822.83% worse than its fixed 24k teacher. Learned weights
improve the same selected nodes with uniform weights by only 4.38%.

It fails the development gate; validation and submission packaging remain
closed. Despite exact training-mode and Gaussian moment matching, the 159.16%
calibration-to-selection degradation identifies function-space overfitting.
