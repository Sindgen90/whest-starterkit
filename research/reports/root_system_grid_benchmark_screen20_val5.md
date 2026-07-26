# Block root-system grid benchmark

This compares block `D4`, `F4`, `D8`, and `E8` root-system grids after signed/permuted Hadamard mixing.

The dev/validation slices here are a quick paired screen, not a full validation rerun.

## Command actually executed

```powershell
research\scripts\run_root_system_grid_benchmark.py --dev-limit 20 --validation-limit 5 --grids d4 f4 d8 e8 --result-tag screen20_val5
```

## Result matrix

| grid | split | method | final MSE | adjusted score | FLOPs | win vs root |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| d4 | development_selection | root | 2.35328e-05 | 2.35328e-06 | 6.66939e+09 | nan |
| d4 | development_selection | k2_root | 1.43003e-05 | 1.43003e-06 | 8.28612e+09 | 0.8 |
| d4 | development_selection | diagk3_stack | 1.44617e-05 | 1.44617e-06 | 9.90508e+09 | 0.8 |
| d4 | validation_frozen | root | 1.64781e-05 | 1.64781e-06 | 6.66939e+09 | nan |
| d4 | validation_frozen | k2_root | 1.26943e-05 | 1.26943e-06 | 8.28612e+09 | 1 |
| d4 | validation_frozen | diagk3_stack | 1.2988e-05 | 1.2988e-06 | 9.90508e+09 | 1 |
| f4 | development_selection | root | 1.77481e-05 | 1.77481e-06 | 1.33387e+10 | nan |
| f4 | development_selection | k2_root | 1.10256e-05 | 1.10256e-06 | 1.49555e+10 | 0.8 |
| f4 | development_selection | diagk3_stack | 1.32928e-05 | 1.32928e-06 | 1.65744e+10 | 0.7 |
| f4 | validation_frozen | root | 7.20271e-06 | 7.20271e-07 | 1.33387e+10 | nan |
| f4 | validation_frozen | k2_root | 6.37505e-06 | 6.37505e-07 | 1.49555e+10 | 0.6 |
| f4 | validation_frozen | diagk3_stack | 7.0924e-06 | 7.0924e-07 | 1.65744e+10 | 0.4 |
| d8 | development_selection | root | 1.73928e-05 | 1.73928e-06 | 1.55618e+10 | nan |
| d8 | development_selection | k2_root | 1.08074e-05 | 1.08074e-06 | 1.71786e+10 | 0.8 |
| d8 | development_selection | diagk3_stack | 1.02447e-05 | 1.02447e-06 | 1.87975e+10 | 0.6 |
| d8 | validation_frozen | root | 8.45692e-06 | 8.45692e-07 | 1.55618e+10 | nan |
| d8 | validation_frozen | k2_root | 6.21653e-06 | 6.21653e-07 | 1.71786e+10 | 1 |
| d8 | validation_frozen | diagk3_stack | 9.10405e-06 | 9.10405e-07 | 1.87975e+10 | 0.6 |
| e8 | development_selection | root | 8.43778e-06 | 1.03446e-06 | 3.33467e+10 | nan |
| e8 | development_selection | k2_root | 7.71713e-06 | 9.91977e-07 | 3.49635e+10 | 0.9 |
| e8 | development_selection | diagk3_stack | 8.91876e-06 | 1.19952e-06 | 3.65824e+10 | 0.7 |
| e8 | validation_frozen | root | 5.18752e-06 | 6.35981e-07 | 3.33467e+10 | nan |
| e8 | validation_frozen | k2_root | 4.17842e-06 | 5.37104e-07 | 3.49635e+10 | 0.8 |
| e8 | validation_frozen | diagk3_stack | 4.25458e-06 | 5.72216e-07 | 3.65824e+10 | 0.6 |

## Known full-validation references

These rows are from prior full validation runs and are included only as a sanity comparator.

| method | final MSE | adjusted score | FLOPs | source |
| --- | ---: | ---: | ---: | --- |
| collapsed diagonal-K3 residual stack | 3.34838e-06 | 3.58334e-07 | 2.91087e+10 | `research/results/diag_k3_residual_stack_s03_l8_validation_manifest_eval.json` |
| layerwise diagonal-K3 residual stack | 3.34836e-06 | 3.58339e-07 | 2.91092e+10 | `research/results/layerwise_diag_k3_residual_stack_s1_l8_h2_validation_manifest_eval.json` |
| K2 + signed-Hadamard cubature 6144 | 3.79759e-06 | 3.83801e-07 | 2.74896e+10 | `research/results/combo_k2_signed_hadamard_cubature_6144_validation_manifest_eval.json` |
| signed-Hadamard cubature 6144 | 4.1252e-06 | 4.1252e-07 | 2.58729e+10 | `research/results/signed_hadamard_cubature_6144_validation_manifest_eval.json` |

## Screen decision

Best root-grid screen row by validation adjusted score: `e8 / k2_root` with final MSE `4.17842e-06`, adjusted score `5.37104e-07`, and FLOPs `3.49635e+10`.

Do not package a root-grid estimator from this screen.  The only competitive root grid is `E8`, but its FLOP cost pushes adjusted score behind the current diagonal-K3 residual stack.
