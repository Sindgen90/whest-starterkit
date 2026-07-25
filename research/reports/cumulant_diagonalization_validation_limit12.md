# Cumulant-diagonalization diagnostic

This tests whether an activation basis change concentrates third/fourth cumulant energy onto marginal diagonal cumulants.

## Command actually executed

```powershell
research\scripts\run_cumulant_diagonalization_diagnostic.py --partition validation --limit 12 --samples 2048 --layers 4 8 16 24 32 --skew-rank 16 --skew-iterations 8
```

## Mean diagonal cumulant fractions

| layer | basis | R3 diag fraction | R4 diag fraction |
| ---: | --- | ---: | ---: |
| 4 | original_neuron | 0.022 | 0.000 |
| 4 | random_orthogonal | 0.000 | 0.000 |
| 4 | pca_rotation | 0.029 | 0.001 |
| 4 | whitened_pca | 0.232 | 0.331 |
| 4 | whitened_random | 0.000 | 0.000 |
| 4 | whitened_skew_power | 0.401 | 0.673 |
| 8 | original_neuron | 0.007 | 0.000 |
| 8 | random_orthogonal | 0.001 | 0.000 |
| 8 | pca_rotation | 0.170 | 0.017 |
| 8 | whitened_pca | 0.336 | 0.368 |
| 8 | whitened_random | 0.000 | 0.000 |
| 8 | whitened_skew_power | 0.535 | 0.763 |
| 16 | original_neuron | 0.002 | 0.000 |
| 16 | random_orthogonal | 0.000 | 0.000 |
| 16 | pca_rotation | 0.625 | 0.207 |
| 16 | whitened_pca | 0.326 | 0.350 |
| 16 | whitened_random | 0.000 | 0.000 |
| 16 | whitened_skew_power | 0.593 | 0.799 |
| 24 | original_neuron | 0.002 | 0.000 |
| 24 | random_orthogonal | 0.000 | 0.000 |
| 24 | pca_rotation | 0.759 | 0.288 |
| 24 | whitened_pca | 0.280 | 0.299 |
| 24 | whitened_random | 0.000 | 0.000 |
| 24 | whitened_skew_power | 0.607 | 0.790 |
| 32 | original_neuron | 0.001 | 0.000 |
| 32 | random_orthogonal | 0.000 | 0.000 |
| 32 | pca_rotation | 0.850 | 0.483 |
| 32 | whitened_pca | 0.284 | 0.305 |
| 32 | whitened_random | 0.000 | 0.000 |
| 32 | whitened_skew_power | 0.643 | 0.827 |

## Decision rule

A diagonal-cumulant estimator deserves implementation only if a basis consistently pushes R3/R4 diagonal fractions into roughly the 0.5-0.8 range.  If whitening/skew-power does not materially improve over raw coordinates, stop this branch.
