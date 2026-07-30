# Paired Box--Muller versus inverse-CDF RQMC at 4,096 points

Only the uniform-to-Gaussian transformation changed. The rank-1 lattice,
multiplier, random-shift seed policy, sample count, network forward pass,
and equal-weight averaging are identical. No rotation, control variate,
analytic correction, calibration, or learning is included.

Only odd development-selection MLPs were evaluated. Validation and holdout
were not touched.

## Command actually executed

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe research\scripts\run_box_muller_rqmc_4096.py
```

## Result matrix

| method | final MSE | adjusted score | FLOPs | worst-decile MSE | win vs inverse CDF |
| --- | ---: | ---: | ---: | ---: | ---: |
| inverse-CDF RQMC 4096 | 5.58236e-06 | 5.58236e-07 | 1.73225e+10 | 1.36029e-05 | -- |
| Box--Muller RQMC 4096 | 6.19525e-06 | 6.19525e-07 | 1.72643e+10 | 1.59481e-05 | 0.500 |
| K3 residual stack | 2.94144e-06 | 3.14783e-07 | 2.91086e+10 | 8.7337e-06 | -- |

## Paired comparison

- Mean paired MSE difference, Box--Muller minus inverse CDF: `+6.12886e-07`.
- Standard error of the paired difference: `9.54389e-07`.
- Raw MSE gain: `0.9011`.
- Paired win fraction: `0.500`.

## Decision

Box--Muller changed mean final MSE by +10.98% versus inverse-CDF RQMC, with paired win fraction 0.500. It fails the development gate. Validation remains closed; do not add an orthogonal rotation or package this estimator.
