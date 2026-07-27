# Six-coset diagnostic for the 24,576-point RQMC lattice

The 24,576 pointwise activations were not stored by earlier experiments, so
this diagnostic replayed each frozen development-selection MLP once. It then
formed residue classes `k mod 6` without additional network evaluations.

Fixed-best and oracle rows below are diagnostics selected on the same
development data; they are not untouched-validation results or submission claims.

## Command actually executed

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe research\scripts\run_rqmc_coset_diagnostic.py
```

## Individual cosets

| residue | samples | mean final MSE | median | worst decile |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 4096 | 7.16114e-06 | 4.88615e-06 | 2.6876e-05 |
| 1 | 4096 | 6.89022e-06 | 6.40366e-06 | 1.84561e-05 |
| 2 | 4096 | 8.39688e-06 | 4.54661e-06 | 3.08396e-05 |
| 3 | 4096 | 6.77387e-06 | 4.97781e-06 | 2.08944e-05 |
| 4 | 4096 | 8.77189e-06 | 6.24421e-06 | 2.61173e-05 |
| 5 | 4096 | 7.85451e-06 | 5.69962e-06 | 2.47189e-05 |

## Compression diagnostic matrix

| rule | samples | final MSE | adjusted diagnostic | equivalent FLOPs |
| --- | ---: | ---: | ---: | ---: |
| current direct 4096 lattice | 4096 | 5.58236e-06 | 5.58236e-07 | 1.73225e+10 |
| best fixed 1-coset `3` | 4096 | 6.77387e-06 | 6.77387e-07 | 1.73225e+10 |
| oracle 1-coset | 4096 | 3.07486e-06 | 3.07486e-07 | 1.73225e+10 |
| best fixed 2-coset `0+1` | 8192 | 2.78753e-06 | 3.55051e-07 | 3.4645e+10 |
| oracle 2-coset | 8192 | 1.24272e-06 | 1.58286e-07 | 3.4645e+10 |
| best fixed 3-coset `0+1+2` | 12288 | 1.66188e-06 | 3.17514e-07 | 5.19675e+10 |
| oracle 3-coset | 12288 | 8.53098e-07 | 1.6299e-07 | 5.19675e+10 |
| all six cosets / original 24576 | 24576 | 1.04995e-06 | 4.01201e-07 | 1.03935e+11 |
| diagonal-K3 residual stack | -- | 2.94144e-06 | 3.14783e-07 | 2.91086e+10 |

## Error dependence

Pearson correlation of signed final-neuron errors, flattened across MLPs:

| residue | 0 | 1 | 2 | 3 | 4 | 5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000 | -0.206 | -0.129 | 0.023 | 0.031 | 0.031 |
| 1 | -0.206 | 1.000 | -0.171 | 0.091 | 0.155 | -0.164 |
| 2 | -0.129 | -0.171 | 1.000 | -0.109 | -0.150 | 0.238 |
| 3 | 0.023 | 0.091 | -0.109 | 1.000 | 0.214 | -0.062 |
| 4 | 0.031 | 0.155 | -0.150 | 0.214 | 1.000 | -0.312 |
| 5 | 0.031 | -0.164 | 0.238 | -0.062 | -0.312 | 1.000 |

- Mean off-diagonal Pearson correlation of signed errors: `-0.034733`.
- Mean off-diagonal uncentered error cosine: `-0.034417`.
- Effective rho in the six-way MSE identity: `-0.035117`.
- MSE predicted by `V/6 * (1 + 5*rho)`: `1.04995e-06`.
- Actual MSE of the six-coset average: `1.04995e-06`.

## Decision

No fixed 24k coset beats the current direct 4k lattice, so blindly taking
every sixth point is not justified. The low effective cross-error correlation
shows that most of the 24k gain comes from averaging several differently wrong
4k cosets.

The in-sample best three-coset rule `0+1+2` has adjusted diagnostic
`3.17514e-7`, only 0.87% worse than K3. It was selected on these same
development MLPs, however, so this is a hypothesis rather than an improvement
claim. Its residues must be selected on a disjoint calibration split before a
paired selection or validation evaluation.
