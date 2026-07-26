# Diagonal-K3 residual recalibration comparison

This keeps the same estimator envelope and only changes residual-family/calibration constants.

## Command actually executed

```powershell
research\scripts\run_diagonal_k3_recalibration_compare.py --result-tag full
```

## Result matrix

| family | calibration | split | final MSE | adjusted score | all-layer MSE | FLOPs | a | b |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| collapsed_s-0.3_L8 | even_final | validation_frozen | 3.3483805e-06 | 3.5833298e-07 | 5.889142e-06 | 2.9108571e+10 | 0.10653088 | -2.6372835 |
| collapsed_s-0.3_L8 | odd_final | validation_frozen | 3.4535541e-06 | 3.6958832e-07 | 5.6590537e-06 | 2.9108571e+10 | 0.13210004 | -1.5386836 |
| collapsed_s-0.3_L8 | all_dev_final | validation_frozen | 3.3764299e-06 | 3.6133475e-07 | 5.7747292e-06 | 2.9108571e+10 | 0.11626737 | -2.045178 |
| collapsed_s-0.3_L8 | all_dev_all_layers | validation_frozen | 4.2507003e-06 | 4.5489636e-07 | 5.2076191e-06 | 2.9108571e+10 | 0.25618813 | -0.82509287 |
| collapsed_s-0.3_L8 | all_dev_late_16_24_32 | validation_frozen | 3.5360445e-06 | 3.7841619e-07 | 5.4443429e-06 | 2.9108571e+10 | 0.16753643 | -1.9186536 |
| layerwise_s-1_L8_H2 | even_final | validation_frozen | 3.348362e-06 | 3.5833759e-07 | 5.889255e-06 | 2.9109106e+10 | 0.10653433 | -0.79139691 |
| layerwise_s-1_L8_H2 | odd_final | validation_frozen | 3.4535468e-06 | 3.6959434e-07 | 5.6591024e-06 | 2.9109106e+10 | 0.13210373 | -0.46170814 |
| layerwise_s-1_L8_H2 | all_dev_final | validation_frozen | 3.376419e-06 | 3.6134022e-07 | 5.7747984e-06 | 2.9109106e+10 | 0.1162715 | -0.61368483 |
| layerwise_s-1_L8_H2 | all_dev_all_layers | validation_frozen | 4.2506451e-06 | 4.5489883e-07 | 5.2076349e-06 | 2.9109106e+10 | 0.25618761 | -0.24763343 |
| layerwise_s-1_L8_H2 | all_dev_late_16_24_32 | validation_frozen | 3.5360225e-06 | 3.7842079e-07 | 5.444405e-06 | 2.9109106e+10 | 0.16753992 | -0.57575093 |

## Current full-validation references

| candidate | final MSE | adjusted score | FLOPs |
| --- | ---: | ---: | ---: |
| current collapsed package | 3.3483805e-06 | 3.583345e-07 | 2.9108694e+10 |
| raw-MSE layerwise package | 3.348362e-06 | 3.583391e-07 | 2.9109229e+10 |

## Decision

Best recalibration by validation adjusted score: `collapsed_s-0.3_L8 / even_final` with final MSE `3.3483805e-06`, adjusted score `3.5833298e-07`, and FLOPs `2.9108571e+10`.

This recovers the existing collapsed package calibration.  The tiny adjusted-score difference versus the packaged reference is a script/package FLOP-counting detail, not a new estimator.  Do not package a replacement.
