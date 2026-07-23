# Stein/RQMC estimator checkpoint

Status: checkpoint before implementing estimator 4 (`rqmc_hermite_active`). Holdout has not been evaluated.

## Files and APIs inspected/reused

- `submissions/rqmc_lattice_002/estimator.py`: frozen ordinary RQMC estimator. Reused its rank-1 lattice structure, random shift seeding, `flops.stats.norm.ppf` Gaussian transform, batched ReLU forward pass, all-layer accumulation, and budget posture.
- `experiments/run_baselines.py`: reused its local dataset access pattern, MLP row adapter, metric definitions, final/all-layer MSE calculations, score-floor adjustment, and depth-prefix MSE convention.
- `research/configs/stein_holdout_manifest.json`: frozen whole-MLP partition manifest. Development IDs are 0-59, validation IDs are 60-79, holdout IDs are 80-99.
- `tests/test_submission_imports.py`: extended to scan `submission_variants/*/estimator.py` and ban unavailable submission imports including NumPy/SciPy/PyTorch/JAX/sklearn/pandas.
- `whest validate`, `whest package`, and `whest validate-package`: used as official submission-contract checks.

The detailed repository audit is in `research/reports/stein_rqmc_repository_audit.md`.

## Implemented estimator variants

### Estimator 1: ordinary RQMC baseline

Path: `submission_variants/rqmc_baseline/estimator.py`

This is a preserved copy of `submissions/rqmc_lattice_002/estimator.py`. It uses 1293 shifted lattice samples, maps uniforms to Gaussians with `flops.stats.norm.ppf`, evaluates every ReLU layer in batch, and returns all-layer activation means.

### Estimator 2: radial Rao-Blackwellized RQMC

Path: `submission_variants/rqmc_radial/estimator.py`

For a positively homogeneous network and `X = R U`, it estimates:

```text
E[f(X)] = E[R] E[f(U)]
```

where `U = Z / ||Z||`, `Z ~ N(0, I)`, and

```text
E[R] = exp(0.5 log 2 + lgamma((d + 1) / 2) - lgamma(d / 2)).
```

The estimator reuses the existing RQMC Gaussian points, normalizes each point to the unit sphere, evaluates the ReLU MLP on sphere points, and multiplies all predicted layer means by `E[R]`.

Runtime validity guard:

- requires an MLP with matrix `weights`;
- rejects nonzero `biases` when present;
- assumes the challenge ReLU forward convention, which is positively homogeneous.

No Hermite controls are applied to radial samples.

### Estimator 3: cross-fitted random-basis Hermite/Stein RQMC

Path: `submission_variants/rqmc_hermite_random/estimator.py`

Current packaged default:

- rank `r = 4`;
- degree 2;
- feature count `M = 14`;
- two independent RQMC folds of 646 samples each;
- projection coefficients;
- shrinkage `0.5`.

The fixed basis is a deterministic subset of dense Walsh-Hadamard columns. It is orthonormal, independent of the MLP, and independent of the evaluated RQMC randomizations.

For `z = Q^T x`, the zero-mean features are:

```text
z_i
(z_i^2 - 1) / sqrt(2)
z_i z_j, i < j
```

The estimator uses two independent randomized RQMC folds. It fits coefficients on fold A and evaluates on fold B, fits on B and evaluates on A, then averages the two residual estimates. No constant feature is subtracted.

### Estimator 4: active-basis Hermite/Stein RQMC

Not implemented yet. Only active-spectrum diagnostics and manual `J_f(x)^T r` tests have been implemented, as required before building the full estimator.

## New shared code

- `research/src/rqmc/points.py`: reusable RQMC lattice, Gaussian transform, independent folds, radial norm formula, sphere normalization, and NumPy reference implementation for tests.
- `research/src/control_variates/hermite.py`: fixed Hadamard basis, degree-2 Hermite feature construction, projection coefficients, and NumPy cross-fitting reference.
- `research/src/active_subspaces/gradients.py`: manual forward pass and manual input-Jacobian-transpose probe for bias-free ReLU MLPs.
- `research/scripts/evaluate_submission_variants.py`: manifest-aware paired evaluator for baseline/radial/Hermite-random.
- `research/scripts/random_hermite_grid.py`: compact rank/shrinkage grid for random-basis Hermite.
- `research/scripts/active_spectrum_diagnostics.py`: pre-estimator active-subspace spectral diagnostic.

## Correctness tests

Command:

```powershell
& 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' -m pytest tests\test_submission_imports.py tests\test_stein_rqmc_methods.py -q
```

Result:

```text
8 passed in 0.45s
```

Covered:

- RQMC independent randomizations are distinct and repeated seeds reproduce;
- fixed Hadamard basis is orthonormal;
- Hermite degree-2 features have empirical zero means and near-identity covariance;
- cross-fitted CV is unbiased and reduces error on a synthetic function in the feature span;
- radial expected norm formula matches known dimensions;
- radial ReLU estimate matches analytic Gaussian-ReLU mean within Monte Carlo tolerance;
- bias-free ReLU MLPs are positively homogeneous;
- manual `J_f(x)^T r` matches finite differences.

## Development results

Partition: frozen development IDs 0-59. Holdout was not touched.

Command:

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\evaluate_submission_variants.py --partition development --methods rqmc_baseline rqmc_radial rqmc_hermite_random
```

This run used the initial Hermite default `r=8`, `shrinkage=0.5`.

| Method | n | Mean final MSE | Median final MSE | Mean adjusted score | Mean FLOPs / MLP | Win frac vs RQMC | Raw MSE gain vs RQMC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RQMC baseline | 60 | 2.43848e-05 | 1.50229e-05 | 2.43848e-06 | 5.46826e+09 | — | — |
| Radial RQMC | 60 | 2.41990e-05 | 1.47171e-05 | 2.41990e-06 | 5.46926e+09 | 0.517 | 1.0077 |
| Hermite random r=8 s=0.5 | 60 | 3.83052e-05 | 2.21044e-05 | 3.83052e-06 | 6.42524e+09 | 0.267 | 0.6366 |

Radial is slightly better on mean MSE, but the paired confidence interval crosses zero. Hermite-random r=8 is materially worse and has outliers.

## Random-basis Hermite development grid

Command:

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\random_hermite_grid.py --partition development
```

Output: `research/results/random_hermite_grid_development.csv`

| Rank | Shrinkage | Mean final MSE | Win frac vs RQMC | Raw MSE gain vs RQMC |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.25 | 3.73141e-05 | 0.267 | 0.654 |
| 4 | 0.5 | 3.73045e-05 | 0.267 | 0.654 |
| 4 | 0.75 | 3.73591e-05 | 0.267 | 0.653 |
| 4 | 1.0 | 3.74782e-05 | 0.267 | 0.651 |
| 8 | 0.25 | 3.77003e-05 | 0.267 | 0.647 |
| 8 | 0.5 | 3.83052e-05 | 0.267 | 0.637 |
| 8 | 0.75 | 3.92031e-05 | 0.283 | 0.622 |
| 8 | 1.0 | 4.03938e-05 | 0.283 | 0.604 |
| 16 | 0.25 | 3.76077e-05 | 0.250 | 0.648 |
| 16 | 0.5 | 3.88123e-05 | 0.250 | 0.628 |
| 16 | 0.75 | 4.10021e-05 | 0.217 | 0.595 |
| 16 | 1.0 | 4.41771e-05 | 0.200 | 0.552 |

All tested random-basis Hermite configurations lost to ordinary RQMC on development. The packaged default was therefore set to the least bad development setting: `r=4`, `shrinkage=0.5`.

## Validation results

Partition: frozen validation IDs 60-79. Holdout was not touched.

Command:

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\evaluate_submission_variants.py --partition validation --methods rqmc_baseline rqmc_radial rqmc_hermite_random --hermite-rank 4 --hermite-shrinkage 0.5
```

| Method | n | Mean final MSE | Median final MSE | Mean adjusted score | Mean FLOPs / MLP | Win frac vs RQMC | Raw MSE gain vs RQMC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RQMC baseline | 20 | 1.99155e-05 | 1.28719e-05 | 1.99155e-06 | 5.46826e+09 | — | — |
| Radial RQMC | 20 | 1.94883e-05 | 1.30108e-05 | 1.94883e-06 | 5.46926e+09 | 0.600 | 1.0219 |
| Hermite random r=4 s=0.5 | 20 | 2.68167e-05 | 1.92563e-05 | 2.68167e-06 | 5.78529e+09 | 0.600 | 0.7427 |

Hermite-random wins 60% of MLPs on validation but loses badly in mean due to outliers. It is not submission-recommended in its current form.

## Active-spectrum estimates

Command:

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\active_spectrum_diagnostics.py --partition development --limit 3 --pilot-samples 8 --probes-per-sample 2
```

Output report: `research/reports/active_spectrum_diagnostics.md`

| MLP id | Gradient columns | E4 | E8 | E16 | Interpretation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 16 | 0.545 | 0.798 | 1.000 | moderate rank-8 concentration |
| 1 | 16 | 0.612 | 0.843 | 1.000 | good rank-8 concentration |
| 2 | 16 | 0.665 | 0.865 | 1.000 | good rank-8 concentration |

This is enough to justify building a small active-basis prototype next, but not enough to claim estimator 4 works.

## Official validation/package status

Commands:

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' validate --estimator submission_variants\rqmc_radial\estimator.py --format plain
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' validate --estimator submission_variants\rqmc_hermite_random\estimator.py --format plain
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' package --estimator submission_variants\rqmc_radial\estimator.py --output submissions\rqmc_radial_1293.tar.gz --yes --format plain
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' validate-package submissions\rqmc_radial_1293.tar.gz --format plain
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' package --estimator submission_variants\rqmc_hermite_random\estimator.py --output submissions\rqmc_hermite_random_r4_shrink05_646x2.tar.gz --yes --format plain
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' validate-package submissions\rqmc_hermite_random_r4_shrink05_646x2.tar.gz --format plain
```

Results:

- `submission_variants/rqmc_radial/estimator.py`: `whest validate` passed.
- `submission_variants/rqmc_hermite_random/estimator.py`: `whest validate` passed.
- `submissions/rqmc_radial_1293.tar.gz`: valid submission archive.
- `submissions/rqmc_hermite_random_r4_shrink05_646x2.tar.gz`: valid submission archive.

Small official subprocess runs also completed with 0/5 failed MLPs for radial and Hermite-random.

## Recommendation

Current submission ranking:

1. Ordinary RQMC remains the safest estimator 1 baseline.
2. Radial RQMC is the only new estimator with a small positive validation signal. Its improvement is modest and statistically weak, but it is cheap, clean, and submission-valid.
3. Random-basis Hermite/Stein should not be submitted yet. Even after compact rank/shrinkage tuning, it loses mean MSE and adjusted score due to outliers.
4. Active-basis Hermite is not implemented yet. The active-spectrum diagnostic makes it the next reasonable experiment.

Next single implementation step:

Implement the active-basis estimator prototype with `rank=4`, `pilot_samples=8`, `output_probes_per_sample=2`, and two 646-sample independent RQMC cross-fit folds. Count pilot forward/reverse/basis/feature/regression FLOPs, then compare only on development against ordinary RQMC and random-basis Hermite with the same feature count.

