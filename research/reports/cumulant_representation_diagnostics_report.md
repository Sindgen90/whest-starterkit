# Cumulant representation diagnostics

This report covers the non-submission diagnostics requested after the signed Hadamard cubature result:

1. Tucker/K3 mode-rank prerequisite;
2. cumulant-diagonalizing basis evidence test;
3. oracle scalar ReLU activation-update comparison.

No holdout MLPs were evaluated.

## Commands actually executed

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\run_cumulant_mode_rank_diagnostic.py --partition development --limit 12 --samples 2048 --probes 96 --layers 4 8 16 24 32 --ranks 4 8 16 32 64
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\run_cumulant_diagonalization_diagnostic.py --partition development --limit 8 --samples 2048 --layers 4 8 16 24 32 --skew-rank 16 --skew-iterations 8
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\run_cumulant_diagonalization_diagnostic.py --partition validation --limit 12 --samples 2048 --layers 4 8 16 24 32 --skew-rank 16 --skew-iterations 8
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\run_activation_update_oracle_diagnostic.py --partition development --limit 8 --samples 8192 --neurons 32 --layers 4 8 16 24 32 --saddlepoint-nodes 48
```

## 1. Tucker/K3 mode-rank prerequisite

Question: after preserving diagonal and repeated-index terms, is the generic all-distinct third-cumulant residual low-rank in mode space?

Diagnostic: sample activations, form random contractions `T[g,g,.]`, subtract repeated-index contractions, and measure singular energy.

Development subset: 12 MLPs, 2048 samples, 96 probes.

| layer | generic rank-4 | rank-8 | rank-16 | rank-32 | rank-64 | effective rank | generic/full energy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.280 | 0.439 | 0.639 | 0.833 | 0.966 | 28.17 | 0.477 |
| 8 | 0.556 | 0.696 | 0.822 | 0.924 | 0.987 | 8.27 | 0.325 |
| 16 | 0.832 | 0.893 | 0.940 | 0.977 | 0.997 | 2.45 | 0.335 |
| 24 | 0.908 | 0.941 | 0.969 | 0.989 | 0.999 | 1.53 | 0.403 |
| 32 | 0.944 | 0.964 | 0.982 | 0.995 | 1.000 | 1.26 | 0.466 |

Decision: plausible after a warmup.  Layer 4 is too broad for rank 16, but from layer 8 onward rank 32 is strong; from layer 16 onward rank 16 is strong.  Do not build a full Tucker-K3 estimator immediately, but keep a narrow "low-rank generic after warmup layer" prototype on the possible list.

## 2. Cumulant-diagonalizing basis test

Question: can an activation basis concentrate third/fourth cumulants onto diagonal marginal cumulants?

Diagnostic:

`R3(Q) = sum_i kappa_iii(Q^T X)^2 / ||kappa_3(X)||_F^2`

`R4(Q) = sum_i kappa_iiii(Q^T X)^2 / ||kappa_4(X)||_F^2`

The tested bases were raw neuron coordinates, random orthogonal rotations, PCA rotations, whitened PCA coordinates, whitened random rotations, and a simple whitened skewness-power basis.

Development best-basis matrix:

| layer | best basis | R3 diagonal | R4 diagonal |
| ---: | --- | ---: | ---: |
| 4 | whitened_skew_power | 0.362 | 0.624 |
| 8 | whitened_skew_power | 0.487 | 0.716 |
| 16 | whitened_skew_power | 0.569 | 0.778 |
| 24 | whitened_skew_power | 0.587 | 0.783 |
| 32 | whitened_skew_power | 0.628 | 0.837 |

Validation best-basis matrix:

| layer | best basis | R3 diagonal | R4 diagonal |
| ---: | --- | ---: | ---: |
| 4 | whitened_skew_power | 0.401 | 0.673 |
| 8 | whitened_skew_power | 0.535 | 0.763 |
| 16 | whitened_skew_power | 0.593 | 0.799 |
| 24 | whitened_skew_power | 0.607 | 0.790 |
| 32 | whitened_skew_power | 0.643 | 0.827 |

Decision: pass.  The basis-change idea is real in sampled activations.  It is too large to convert into a full submission estimator immediately, but this is now the most promising cumulant-representation direction.

## 3. Oracle scalar ReLU activation update

Question: with near-oracle scalar cumulants for preactivations, does saddlepoint improve the ReLU mean update?

Development subset: 8 MLPs, 32 neurons per selected layer, 8192 preactivation samples.

| layer | Gaussian MSE | Edgeworth K3 | Edgeworth K3+K4 | Saddlepoint K3+K4 | saddle fail |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 5.28e-6 | 1.10e-6 | 3.34e-7 | 5.82e-7 | 7.8% |
| 8 | 4.76e-6 | 7.42e-7 | 1.28e-7 | 8.90e-7 | 1.6% |
| 16 | 3.11e-6 | 3.70e-7 | 7.47e-8 | 3.30e-6 | 0.8% |
| 24 | 1.98e-6 | 2.72e-7 | 3.71e-8 | 3.05e-6 | 0.4% |
| 32 | 7.91e-7 | 1.06e-7 | 2.41e-8 | 3.30e-6 | 2.3% |

Decision: reject saddlepoint for Phase 1.  It has root/convexity failures and becomes worse than Gaussian in deeper layers.  Edgeworth K3+K4 is excellent with oracle cumulants, which means the bottleneck is cumulant representation/propagation, not the scalar activation formula.

## Overall recommendation

Current primary submission candidate remains:

`submissions/signed_hadamard_cubature_6144/estimator.py`

The next small, submission-relevant experiment should be:

calibrate a scalar convex blend of corrected K2 and signed Hadamard cubature 6144.

The next larger research branch, if we choose to spend time on it, should be:

whitened skew-power cumulant basis with diagonal K3/K4 propagation after a warmup layer.

Do not spend Phase 1 time on saddlepoint, characteristic functions, or full high-dimensional QBMM unless the current stronger candidates stop improving.
