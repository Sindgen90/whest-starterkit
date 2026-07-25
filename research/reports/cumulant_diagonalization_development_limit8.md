# Cumulant-diagonalization diagnostic

This tests whether an activation basis change concentrates third/fourth cumulant energy onto marginal diagonal cumulants.

## Command actually executed

```powershell
research\scripts\run_cumulant_diagonalization_diagnostic.py --partition development --limit 8 --samples 2048 --layers 4 8 16 24 32 --skew-rank 16 --skew-iterations 8
```

## Mean diagonal cumulant fractions

| layer | basis | R3 diag fraction | R4 diag fraction |
| ---: | --- | ---: | ---: |
| 4 | original_neuron | 0.023 | 0.000 |
| 4 | random_orthogonal | 0.000 | 0.000 |
| 4 | pca_rotation | 0.024 | 0.001 |
| 4 | whitened_pca | 0.204 | 0.309 |
| 4 | whitened_random | 0.000 | 0.000 |
| 4 | whitened_skew_power | 0.362 | 0.624 |
| 8 | original_neuron | 0.008 | 0.000 |
| 8 | random_orthogonal | 0.001 | 0.000 |
| 8 | pca_rotation | 0.162 | 0.030 |
| 8 | whitened_pca | 0.299 | 0.341 |
| 8 | whitened_random | 0.000 | 0.000 |
| 8 | whitened_skew_power | 0.487 | 0.716 |
| 16 | original_neuron | 0.003 | 0.000 |
| 16 | random_orthogonal | 0.001 | 0.000 |
| 16 | pca_rotation | 0.477 | 0.103 |
| 16 | whitened_pca | 0.325 | 0.348 |
| 16 | whitened_random | 0.000 | 0.000 |
| 16 | whitened_skew_power | 0.569 | 0.778 |
| 24 | original_neuron | 0.002 | 0.000 |
| 24 | random_orthogonal | 0.000 | 0.000 |
| 24 | pca_rotation | 0.741 | 0.375 |
| 24 | whitened_pca | 0.293 | 0.316 |
| 24 | whitened_random | 0.000 | 0.000 |
| 24 | whitened_skew_power | 0.587 | 0.783 |
| 32 | original_neuron | 0.001 | 0.000 |
| 32 | random_orthogonal | 0.000 | 0.000 |
| 32 | pca_rotation | 0.884 | 0.598 |
| 32 | whitened_pca | 0.286 | 0.322 |
| 32 | whitened_random | 0.000 | 0.000 |
| 32 | whitened_skew_power | 0.628 | 0.837 |

## Decision rule

A diagonal-cumulant estimator deserves implementation only if a basis consistently pushes R3/R4 diagonal fractions into roughly the 0.5-0.8 range.  If whitening/skew-power does not materially improve over raw coordinates, stop this branch.
