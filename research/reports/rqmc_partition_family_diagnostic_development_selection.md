# Non-learned partition-family diagnostic

The same fixed 24,576-point RQMC lattice was partitioned in several
predeclared ways. No MLP result was used to select a partition, group,
parameter, or seed. Only odd development-selection MLPs were evaluated;
validation and holdout were not touched.

Every six-way partition averages back to the identical 24k estimator.
The diagnostic therefore asks whether one 4k group is unusually accurate
or whether the 24k gain comes from cancellation among six group errors.

## Command actually executed

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe research\scripts\run_rqmc_partition_family_diagnostic.py
```

## Result matrix

| partition family | mean individual 4k MSE | best descriptive group MSE | win vs direct 4k | full six-way MSE | effective rho | largest leave-one-out penalty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| true_residue_cosets | 7.64142e-06 | 6.77387e-06 (g3) | 0.400 | 1.04995e-06 | -0.03512 | 3.57613e-07 (remove g1) |
| contiguous_blocks | 3.75874e-05 | 3.23279e-05 (g1) | 0.200 | 1.04995e-06 | -0.16648 | 2.32748e-06 (remove g0) |
| antithetic_pair_blocks | 2.05736e-05 | 1.77662e-05 (g0) | 0.267 | 1.04995e-06 | -0.13876 | 9.79254e-07 (remove g0) |
| affine_index_blocks | 9.13633e-06 | 7.25807e-06 (g0) | 0.467 | 1.04995e-06 | -0.06210 | 5.01232e-07 (remove g1) |
| bit_reversed_blocks | 6.80552e-06 | 5.4744e-06 (g3) | 0.467 | 1.04995e-06 | -0.01486 | 3.22229e-07 (remove g2) |
| 50 random balanced partitions (mean/median) | 1.06791e-05 / 1.04881e-05 | 8.12609e-06 / 8.16168e-06 | -- | 1.04995e-06 | -0.08074 | 5.95497e-07 |

Reference estimators:

| estimator | final MSE | adjusted score | FLOPs |
| --- | ---: | ---: | ---: |
| direct RQMC 4096 | 5.58236e-06 | 5.58236e-07 | 1.73225e+10 |
| K3 residual stack | 2.94144e-06 | 3.14783e-07 | 2.91086e+10 |
| fixed RQMC 24576 | 1.04995e-06 | -- | 1.03935e+11 |

## Per-group details

### true_residue_cosets

| group | individual MSE | MSE without group | removal delta |
| ---: | ---: | ---: | ---: |
| 0 | 7.16114e-06 | 1.36937e-06 | +3.19421e-07 |
| 1 | 6.89022e-06 | 1.40757e-06 | +3.57613e-07 |
| 2 | 8.39688e-06 | 1.37175e-06 | +3.21796e-07 |
| 3 | 6.77387e-06 | 1.14956e-06 | +9.96065e-08 |
| 4 | 8.77189e-06 | 1.22111e-06 | +1.71153e-07 |
| 5 | 7.85451e-06 | 1.36231e-06 | +3.12362e-07 |

### contiguous_blocks

| group | individual MSE | MSE without group | removal delta |
| ---: | ---: | ---: | ---: |
| 0 | 4.32676e-05 | 3.37743e-06 | +2.32748e-06 |
| 1 | 3.23279e-05 | 2.19959e-06 | +1.14964e-06 |
| 2 | 3.74376e-05 | 2.13554e-06 | +1.08558e-06 |
| 3 | 3.46473e-05 | 2.04296e-06 | +9.9301e-07 |
| 4 | 3.90564e-05 | 2.60537e-06 | +1.55542e-06 |
| 5 | 3.87878e-05 | 2.70782e-06 | +1.65787e-06 |

### antithetic_pair_blocks

| group | individual MSE | MSE without group | removal delta |
| ---: | ---: | ---: | ---: |
| 0 | 1.77662e-05 | 2.02921e-06 | +9.79254e-07 |
| 1 | 2.14834e-05 | 1.84457e-06 | +7.9462e-07 |
| 2 | 2.37508e-05 | 2.02822e-06 | +9.78269e-07 |
| 3 | 1.91329e-05 | 1.63671e-06 | +5.86762e-07 |
| 4 | 2.11774e-05 | 1.68773e-06 | +6.37773e-07 |
| 5 | 2.01309e-05 | 1.75895e-06 | +7.08995e-07 |

### affine_index_blocks

| group | individual MSE | MSE without group | removal delta |
| ---: | ---: | ---: | ---: |
| 0 | 7.25807e-06 | 1.25286e-06 | +2.02905e-07 |
| 1 | 9.43003e-06 | 1.55118e-06 | +5.01232e-07 |
| 2 | 7.56208e-06 | 1.44714e-06 | +3.97188e-07 |
| 3 | 8.92586e-06 | 1.19767e-06 | +1.47715e-07 |
| 4 | 1.12343e-05 | 1.288e-06 | +2.38045e-07 |
| 5 | 1.04076e-05 | 1.5036e-06 | +4.53645e-07 |

### bit_reversed_blocks

| group | individual MSE | MSE without group | removal delta |
| ---: | ---: | ---: | ---: |
| 0 | 8.17365e-06 | 1.21113e-06 | +1.61181e-07 |
| 1 | 6.83418e-06 | 1.23733e-06 | +1.8738e-07 |
| 2 | 5.78258e-06 | 1.37218e-06 | +3.22229e-07 |
| 3 | 5.4744e-06 | 1.30437e-06 | +2.54413e-07 |
| 4 | 6.19181e-06 | 1.22705e-06 | +1.77101e-07 |
| 5 | 8.37651e-06 | 1.32898e-06 | +2.79032e-07 |

## Decision

The best descriptive structured 4k group is `bit_reversed_blocks` group 3 with MSE 5.4744e-06, -1.93% versus direct 4096 RQMC. Its paired win fraction is 0.467, and its paired MSE difference is -1.07962e-07 with standard error 8.82662e-07. The true-coset average has effective error correlation -0.03512; its largest leave-one-out penalty is 3.57613e-07 (5.4% of the reduction from the average individual coset to the six-way mean). The small descriptive improvement is not paired-consistent and was found on this same development data. No individual group or coset is responsible for most of the 24k gain.
