# Corrected K2 + signed Hadamard cubature combo

This tests whether corrected K2 adds enough bias correction to justify its extra FLOPs on top of the 6144-point cubature estimator.

## Command actually executed

```powershell
research\scripts\run_k2_cubature_combo.py
```

Submission-file checks:

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\evaluate_estimator_on_manifest.py --partition validation --estimator-path submissions\combo_k2_signed_hadamard_cubature_6144\estimator.py
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' validate --estimator submissions\combo_k2_signed_hadamard_cubature_6144\estimator.py --format plain
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' package --estimator submissions\combo_k2_signed_hadamard_cubature_6144\estimator.py --output submissions\combo_k2_signed_hadamard_cubature_6144.tar.gz --yes --format plain
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' validate-package submissions\combo_k2_signed_hadamard_cubature_6144.tar.gz --format plain
```

Calibrated lambda: `0.879892` (`raw=0.879892`)

## Result matrix

| split | method | final MSE | adjusted score | FLOPs | win vs cubature |
| --- | --- | ---: | ---: | ---: | ---: |
| development_selection | k2 | 2.0986e-05 | 2.0986e-06 | 1.61664e+09 | nan |
| development_selection | cubature | 3.6341e-06 | 3.6341e-07 | 2.58729e+10 | nan |
| development_selection | combo | 3.05437e-06 | 3.08689e-07 | 2.74896e+10 | 0.8 |
| validation_frozen_lambda | k2 | 2.68007e-05 | 2.68007e-06 | 1.61664e+09 | nan |
| validation_frozen_lambda | cubature | 4.1252e-06 | 4.1252e-07 | 2.58729e+10 | nan |
| validation_frozen_lambda | combo | 3.79756e-06 | 3.83799e-07 | 2.74896e+10 | 0.6 |

## Decision rule

Promote the combo only if validation adjusted score improves over raw signed Hadamard cubature.  Because K2 pushes compute above the 10% floor, a tiny raw-MSE improvement is not enough.

## Decision

Promote the combo.  The validation adjusted score improves from `4.12520e-07` for raw signed Hadamard cubature to `3.83799e-07` in the research combo run, and the actual submission-file evaluation reproduced this at `3.83801e-07`.

Package artifact:

`submissions/combo_k2_signed_hadamard_cubature_6144.tar.gz`
