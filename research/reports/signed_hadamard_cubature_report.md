# Signed/permuted Hadamard cubature benchmark

This report covers the sigma-point/cubature idea from the attached note:

> Compare fixed 513-point RQMC, spherical-radial cubature, and higher-order sigma points before developing blockwise rematching.

The high-dimensional GenUT branch failed immediately because matching fourth moments with `2d+1` points for `d=256` gives a very large negative center weight.  The useful discovery was instead averaging multiple signed/permuted spherical-radial cubature bases.

No holdout MLPs were evaluated.  Development used `mlp_id=0..59`; validation used `mlp_id=60..79`.

## Commands actually executed

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\run_sigma_point_benchmark.py --partition development
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\run_sigma_point_benchmark.py --partition validation --methods rqmc_513 axis_cubature_512 hadamard_cubature_512 mc_513
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\run_sigma_point_benchmark.py --partition development --methods rqmc_513 axis_cubature_512 hadamard_cubature_512 signed_hadamard_cubature_1024 signed_hadamard_cubature_2048 signed_hadamard_cubature_4096 signed_hadamard_cubature_6144
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\run_sigma_point_benchmark.py --partition validation --methods rqmc_513 signed_hadamard_cubature_4096 signed_hadamard_cubature_6144
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\evaluate_estimator_on_manifest.py --partition validation --estimator-path submissions\signed_hadamard_cubature_6144\estimator.py
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' validate --estimator submissions\signed_hadamard_cubature_6144\estimator.py --format plain
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' package --estimator submissions\signed_hadamard_cubature_6144\estimator.py --output submissions\signed_hadamard_cubature_6144.tar.gz --yes --format plain
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' validate-package submissions\signed_hadamard_cubature_6144.tar.gz --format plain
```

## Method definitions

| method | point rule | points at width 256 |
| --- | --- | ---: |
| `rqmc_513` | randomly shifted rank-1 lattice transformed by Gaussian inverse CDF | 513 |
| `axis_cubature_512` | `±sqrt(d) e_i` spherical-radial cubature | 512 |
| `hadamard_cubature_512` | `±sqrt(d) q_i`, where `q_i` are Hadamard basis vectors | 512 |
| `axis_genut_513` | `0`, `±sqrt(3)e_i`, weights matching marginal fourth moments | 513 |
| `hadamard_genut_513` | Hadamard-basis version of the same high-order sigma rule | 513 |
| `signed_hadamard_cubature_6144` | average of 12 signed/permuted Hadamard cubature bases | 6144 |

## Development matrix

| method | final MSE | adjusted score | win vs RQMC-513 | gain vs RQMC-513 | FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `signed_hadamard_cubature_6144` | 3.88422e-06 | 3.88422e-07 | 1.000 | 24.020 | 2.58720e10 |
| `signed_hadamard_cubature_4096` | 5.81946e-06 | 5.81946e-07 | 1.000 | 16.033 | 1.72480e10 |
| `signed_hadamard_cubature_2048` | 1.01110e-05 | 1.01110e-06 | 1.000 | 9.228 | 8.62401e09 |
| `signed_hadamard_cubature_1024` | 2.52911e-05 | 2.52911e-06 | 0.917 | 3.689 | 4.31200e09 |
| `axis_cubature_512` | 3.88179e-05 | 3.88179e-06 | 0.783 | 2.404 | 2.15599e09 |
| `hadamard_cubature_512` | 3.94255e-05 | 3.94255e-06 | 0.767 | 2.367 | 2.15599e09 |
| `rqmc_513` | 9.33008e-05 | 9.33008e-06 | n/a | n/a | 2.17374e09 |

The raw `2d+1` GenUT variants are omitted from this compact table because they produced final MSE near `5.91e+1`, caused by unstable negative-weight cancellation.

## Validation matrix

| method | final MSE | adjusted score | win vs RQMC-513 | gain vs RQMC-513 | FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `signed_hadamard_cubature_6144` | 4.12520e-06 | 4.12520e-07 | 1.000 | 34.943 | 2.58720e10 |
| `signed_hadamard_cubature_4096` | 1.02314e-05 | 1.02314e-06 | 1.000 | 14.089 | 1.72480e10 |
| `rqmc_513` | 1.44149e-04 | 1.44149e-05 | n/a | n/a | 2.17374e09 |

The submission-file evaluation for `submissions/signed_hadamard_cubature_6144/estimator.py` matched the research result:

| estimator | validation final MSE | adjusted score | final MSE median | worst-decile MSE | FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `signed_hadamard_cubature_6144` | 4.12520e-06 | 4.12520e-07 | 2.94203e-06 | 1.20402e-05 | 2.58729e10 |
| previous `combo_k2_stein_rqmc_4096_ridge232` | 4.82304e-06 | 4.82304e-07 | 4.40217e-06 | 1.12827e-05 | 1.91816e10 |
| previous `combo_k2_rqmc_4096` | 4.99447e-06 | 4.99447e-07 | 4.37722e-06 | 1.19859e-05 | 1.89392e10 |

## Validation and package status

| check | status |
| --- | --- |
| `python -m py_compile submissions\signed_hadamard_cubature_6144\estimator.py` | passed |
| `whest validate --estimator submissions\signed_hadamard_cubature_6144\estimator.py` | passed |
| `whest package --estimator submissions\signed_hadamard_cubature_6144\estimator.py --output submissions\signed_hadamard_cubature_6144.tar.gz` | passed |
| `whest validate-package submissions\signed_hadamard_cubature_6144.tar.gz` | passed |

Package artifact:

`submissions/signed_hadamard_cubature_6144.tar.gz`

## Recommendation

Promote `signed_hadamard_cubature_6144` to the primary experimental submission candidate.  Keep `combo_k2_stein_rqmc_4096_ridge232` as the backup because it has slightly lower FLOPs and a slightly better worst-decile MSE on validation, but the new cubature estimator has better mean and median validation MSE and a better adjusted validation score.

Next narrow experiment: calibrate a scalar convex combination of corrected K2 and `signed_hadamard_cubature_6144`.  It may or may not help, because the cubature estimator already uses almost the entire score-floor compute allowance; the extra K2 FLOPs push it just above the 10% floor.
