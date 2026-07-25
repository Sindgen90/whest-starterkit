# Cheap diagonal-K3 residual stack

This tests a two-scalar residual form: `cubature + a*(K2-cubature) + b*(diagK3-K2)`.

## Command actually executed

```powershell
research\scripts\run_diagonal_k3_residual_stack.py --third-scale=-0.3 --max-k3-layer 8 --result-tag full_s03_l8
```

## Calibration

- `a`: `0.10653088`
- `b`: `-2.6372835`
- equivalent K2/cubature lambda if `b=0`: `0.89346912`

## Result matrix

| split | method | final MSE | adjusted score | FLOPs | win vs current K2 combo |
| --- | --- | ---: | ---: | ---: | ---: |
| development_selection | cubature | 3.6341e-06 | 3.6341e-07 | 2.58729e+10 | nan |
| development_selection | current_k2_combo | 3.05437e-06 | 3.08689e-07 | 2.74896e+10 | nan |
| development_selection | stack | 2.94144e-06 | 3.14783e-07 | 2.91086e+10 | 0.667 |
| validation_frozen | cubature | 4.1252e-06 | 4.1252e-07 | 2.58729e+10 | nan |
| validation_frozen | current_k2_combo | 3.79756e-06 | 3.838e-07 | 2.74896e+10 | nan |
| validation_frozen | stack | 3.34838e-06 | 3.58333e-07 | 2.91086e+10 | 0.65 |

## Submission package

Estimator:

`submissions/diag_k3_residual_stack_s03_l8/estimator.py`

Package:

`submissions/diag_k3_residual_stack_s03_l8.tar.gz`

Validation commands executed:

```powershell
$env:PYTHONIOENCODING='utf-8'; C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe validate --estimator submissions\diag_k3_residual_stack_s03_l8\estimator.py --format plain
```

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe research\scripts\evaluate_estimator_on_manifest.py --estimator-path submissions\diag_k3_residual_stack_s03_l8\estimator.py --partition validation --output research\results\diag_k3_residual_stack_s03_l8_validation_manifest_eval.json
```

```powershell
$env:PYTHONIOENCODING='utf-8'; C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe package --estimator submissions\diag_k3_residual_stack_s03_l8\estimator.py --output submissions\diag_k3_residual_stack_s03_l8.tar.gz --yes --format plain
```

```powershell
$env:PYTHONIOENCODING='utf-8'; C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe validate-package submissions\diag_k3_residual_stack_s03_l8.tar.gz --format plain
```

All four checks passed. The package is a single-file estimator and imports no unsupported third-party library.
