# K2 + RQMC convex-combo grid

Run ID: `9185da8f-0417-406c-84bf-86d966e39549`

This experiment stayed inside the narrow fallback lane: corrected K2 plus ordinary RQMC, with one globally calibrated convex weight.  No holdout MLPs were evaluated.

## Command actually executed

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\run_k2_rqmc_combo_grid.py --top-k 3
```

Additional same-validation package checks:

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\evaluate_estimator_on_manifest.py --partition validation --estimator-path submissions\combo_k2_rqmc_4096\estimator.py
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\evaluate_estimator_on_manifest.py --partition validation --estimator-path submissions\combo_k2_stein_rqmc_4096_ridge232\estimator.py
```

## Split policy

- Development rows loaded: 60
- Development calibration rows: 30
- Development selection rows: 30
- Validation rows: 20
- Holdout rows touched: 0

## Development top 5

| candidate | lambda | combo MSE | RQMC MSE | K2 MSE | adj score | FLOPs | win vs RQMC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| n5120_offset1000003_mult17797 | 0.838894 | 3.56905e-06 | 4.54727e-06 | 2.0986e-05 | 3.56905e-07 | 2.32698e+10 | 0.833333 |
| n5120_offset271828_mult17797 | 0.758458 | 4.03823e-06 | 5.12059e-06 | 2.0986e-05 | 4.03823e-07 | 2.32698e+10 | 0.766667 |
| n4096_offset1000003_mult17797 | 0.750261 | 4.44811e-06 | 5.79612e-06 | 2.0986e-05 | 4.44811e-07 | 1.89392e+10 | 0.833333 |
| n4096_offset8675309_mult17797 | 0.792423 | 4.48747e-06 | 5.58236e-06 | 2.0986e-05 | 4.48747e-07 | 1.89392e+10 | 0.866667 |
| n5120_offset8675309_mult271829 | 0.807421 | 4.50441e-06 | 5.63405e-06 | 2.0986e-05 | 4.50441e-07 | 2.32698e+10 | 0.733333 |

## Validation results for frozen top configs

| candidate | lambda | combo MSE | RQMC MSE | K2 MSE | adj score | FLOPs | win vs RQMC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| n4096_offset1000003_mult17797 | 0.750261 | 4.99922e-06 | 6.31026e-06 | 2.68007e-05 | 4.99922e-07 | 1.89392e+10 | 0.55 |
| n5120_offset271828_mult17797 | 0.758458 | 5.4744e-06 | 6.88208e-06 | 2.68007e-05 | 5.4744e-07 | 2.32698e+10 | 0.75 |
| n5120_offset1000003_mult17797 | 0.838894 | 5.51106e-06 | 6.61505e-06 | 2.68007e-05 | 5.51106e-07 | 2.32698e+10 | 0.85 |

## Decision

Best validation candidate: `n4096_offset1000003_mult17797` with lambda `0.750261`.
Validation combo final MSE: `4.99922e-06`; adjusted score: `4.99922e-07`.

Same-validation check against the already packaged simple and K2+Stein/RQMC estimators:

| estimator | validation final MSE | adjusted score | FLOPs |
| --- | --- | --- | --- |
| best simple K2+RQMC grid candidate | 4.99922e-06 | 4.99922e-07 | 1.89392e+10 |
| `submissions/combo_k2_rqmc_4096` | 4.99447e-06 | 4.99447e-07 | 1.89392e+10 |
| `submissions/combo_k2_stein_rqmc_4096_ridge232` | 4.82304e-06 | 4.82304e-07 | 1.91816e+10 |

The simple grid candidate is a useful backup but should not replace the K2+Stein/RQMC package as the primary experimental submission.  The existing `submissions/combo_k2_rqmc_4096` package is already essentially this backup family, with a very close lambda (`0.751711` versus the grid winner's `0.750261`).

## Files

- Dev summary: `research/results/k2_rqmc_combo_grid_dev_summary.csv`
- Validation summary: `research/results/k2_rqmc_combo_grid_validation_summary.csv`
- Full JSONL log: `research/results/k2_rqmc_combo_grid.jsonl`
- Packaged simple combo eval: `research/results/combo_k2_rqmc_4096_validation_manifest_eval.json`
- Packaged K2+Stein/RQMC eval: `research/results/combo_k2_stein_rqmc_4096_ridge232_validation_manifest_eval.json`

Top candidates sent to validation: `n5120_offset1000003_mult17797`, `n5120_offset271828_mult17797`, `n4096_offset1000003_mult17797`
