# Fifty random six-way partitions of the 24,576-point lattice

Each of 50 deterministic random permutations split the same candidate pool
into six disjoint 4,096-point groups, producing 300 candidate subsets.
These random subsets are not rank-1-lattice cosets. The only genuine cosets
of the fixed lattice are the six `k mod 6` residue classes evaluated earlier.

Candidate selection used only the 30 even development MLPs. The selected
subset was then evaluated unchanged on the 30 odd development MLPs.
Validation and holdout were not touched.

## Command actually executed

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe research\scripts\run_rqmc_random_partition_diagnostic.py
```

## Updated matrix

| method | selection final MSE | adjusted score | equivalent FLOPs |
| --- | ---: | ---: | ---: |
| direct RQMC 4096 | 5.58236e-06 | 5.58236e-07 | 1.73225e+10 |
| true residue coset `4` (disjoint calibration) | 8.77189e-06 | 8.77189e-07 | 1.73225e+10 |
| best of 300 random subsets selected on even MLPs `p34/g0` | 1.15976e-05 | 1.15976e-06 | 1.73225e+10 |
| full fixed RQMC 24576 | 1.04995e-06 | -- | 1.03935e+11 |
| K3 residual stack | 2.94144e-06 | 3.14783e-07 | 2.91086e+10 |

## Selection diagnostics

- Calibration-selected random subset: calibration MSE `6.62132e-06`, selection MSE `1.15976e-05`, selection rank `214/300`.
- Candidate calibration-versus-selection rank correlation: `-0.0629`.
- Median selection MSE of the 50 independently calibration-selected within-partition winners: `1.07024e-05`.
- In-sample best random subset on odd development MLPs `p43/g0`: `5.86337e-06`. This is an oracle-like diagnostic, not a selectable result.
- Per-MLP oracle choosing the best of all 300 subsets after seeing truth: `1.537e-06`.

## Decision

The best of 300 random subsets selected on disjoint calibration MLPs has selection MSE 1.15976e-05, +107.75% versus direct 4096 RQMC, and adjusted score 1.15976e-06, +268.43% versus K3. It fails the development gate. Random partition search should stop; best-of-many oracle values are selection bias rather than a usable rule.
