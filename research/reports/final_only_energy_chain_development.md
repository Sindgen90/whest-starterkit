# Final-only energy-chain experiment

## Hypothesis

The new public leader's reported `6.32e6` analytical FLOPs are close to the
cost of propagating one uncentered second-moment vector through all 32 layers:

```text
s_l = (W_l * W_l).T @ q_(l-1)
q_l = s_l / 2
mean_l = sqrt(s_l / (2*pi))
```

The estimator returns zeros for layers 1--31 and only the inferred mean for
layer 32. It contains no calibration.

## Commands actually executed

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe -m pytest tests\test_final_only_energy_chain.py tests\test_submission_imports.py -q
$env:PYTHONIOENCODING='utf-8'; C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe validate --estimator submissions\final_only_energy_chain\estimator.py --format plain
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe research\scripts\evaluate_estimator_on_manifest.py --estimator-path submissions\final_only_energy_chain\estimator.py --partition development --output research\results\final_only_energy_chain_development_manifest_eval.json
```

The unit/import tests passed (`4 passed`) and `whest validate` passed.

## Development-selection result

The comparison uses the existing 30 odd-ID development-selection MLPs. The
even-ID development MLPs remain the calibration partition used by earlier
experiments; no energy-chain parameter was fitted.

| method | n | final MSE | adjusted score | FLOPs | win vs K3 stack |
| --- | ---: | ---: | ---: | ---: | ---: |
| cubature | 30 | 3.63410e-06 | 3.63410e-07 | 2.58729e10 | -- |
| current K2/cubature combo | 30 | 3.05437e-06 | 3.08689e-07 | 2.74896e10 | -- |
| diagonal-K3 residual stack | 30 | 2.94144e-06 | 3.14783e-07 | 2.91086e10 | -- |
| final-only energy chain | 30 | 6.72282e-01 | 6.72282e-02 | 6.29222e06 | 0.000 |

Additional energy-chain diagnostics:

- median final MSE: `3.67486e-01`;
- maximum final MSE: `5.44377`;
- mean all-layer MSE: `8.11320e-01`;
- catastrophic/non-finite failures: `0`.

## Decision

Stop this literal recurrence. It is about `2.29e5` times worse in raw MSE than
the diagonal-K3 residual stack and loses on every paired development MLP.
Although its measured `6,292,224` FLOPs closely match the leader's public
`6.32e6` figure, the matching compute signature does not identify the
leader's mathematics.

Frozen validation was not opened for this estimator because the development
gate failed decisively. Do not package or submit this implementation.
