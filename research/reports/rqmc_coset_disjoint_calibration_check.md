# Disjoint calibration check for fixed RQMC coset subsets

Coset subsets were selected only on even development-calibration MLPs and
then evaluated without change on the previously computed odd
development-selection MLPs. Validation and holdout were not touched.

## Command actually executed

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe research\scripts\run_rqmc_coset_calibration_check.py
```

## Frozen-choice matrix

| size | calibrated residues | calibration MSE | selection MSE | selection adjusted | FLOPs | odd in-sample best |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `4` | 5.58221e-06 | 8.77189e-06 | 8.77189e-07 | 1.73225e+10 | `3` / 6.77387e-06 |
| 2 | `2+3` | 2.93279e-06 | 3.37994e-06 | 4.30507e-07 | 3.4645e+10 | `0+1` / 2.78753e-06 |
| 3 | `1+2+3` | 2.13272e-06 | 2.11694e-06 | 4.04456e-07 | 5.19675e+10 | `0+1+2` / 1.66188e-06 |
| K3 | -- | -- | 2.94144e-06 | 3.14783e-07 | 2.91086e+10 | -- |

## Decision

The best disjointly calibrated choice uses 3 cosets `1+2+3` and has selection adjusted score 4.04456e-07, +28.49% versus K3. It does not beat K3, so no coset submission should be packaged from this result.
