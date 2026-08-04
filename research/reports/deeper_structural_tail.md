# Deeper Structural Tail Scout

Last fourteen and last sixteen were tested in isolation on the balanced first 50 Mini MLPs. The established F-last-eight and K-last-twelve estimators were rerun as exact controls.

| Candidate | Thresholds | Raw final MSE | FLOPs | Full adjusted | Structural vs dense |
|---|---:|---:|---:|---:|---:|
| `F_last8_control` | 3.75, 3.6, 3.5, 3.4, 3.3, 3.2, 3, 3 | 4.9185e-07 | 1.2435e+11 | 2.5929e-07 | 3.3344e-10 |
| `K_last12_control` | 4.75, 4.5, 4.25, 4, 3.75, 3.6, 3.5, 3.4, 3.3, 3.2, 3, 3 | 4.9190e-07 | 1.2060e+11 | 2.5429e-07 | 3.3380e-10 |
| `L_last14_tapered` | 5.25, 5, 4.75, 4.5, 4.25, 4, 3.75, 3.6, 3.5, 3.4, 3.3, 3.2, 3, 3 | 4.9189e-07 | 1.1973e+11 | 2.5331e-07 | 3.3377e-10 |
| `M_last16_tapered` | 5.75, 5.5, 5.25, 5, 4.75, 4.5, 4.25, 4, 3.75, 3.6, 3.5, 3.4, 3.3, 3.2, 3, 3 | 4.9190e-07 | 1.1933e+11 | 2.5323e-07 | 3.3364e-10 |

## Paired Against K

| Candidate | Raw change | Raw delta 95% CI | Adjusted change | Adjusted delta 95% CI | FLOP change |
|---|---:|---:|---:|---:|---:|
| `L_last14_tapered` | -0.002% | [-2.1614e-11, 3.4555e-12] | -0.387% | [-1.9361e-09, -1.8590e-10] | -0.715% |
| `M_last16_tapered` | -0.000% | [-1.4066e-11, 7.6533e-12] | -0.417% | [-2.5436e-09, 2.0010e-10] | -1.055% |

## Parity Check

| Candidate | Parity-0 raw | Parity-0 adjusted | Parity-1 raw | Parity-1 adjusted |
|---|---:|---:|---:|---:|
| `K_last12_control` | 5.6619e-07 | 2.9213e-07 | 4.1762e-07 | 2.1646e-07 |
| `L_last14_tapered` | 5.6618e-07 | 2.9064e-07 | 4.1761e-07 | 2.1598e-07 |
| `M_last16_tapered` | 5.6619e-07 | 2.9082e-07 | 4.1761e-07 | 2.1564e-07 |

## Interpretation

Both L and M preserve raw accuracy and improve adjusted score in both parity directions. M is the stronger FLOP-only result, while its measured effective compute is nearly tied with L because the additional narrow operations approach the local dispatch-overhead floor.

M classifies only about 11 neurons at layer 17 and 18 neurons at layer 18; the remaining neurons are still sampled with ReLU. L classifies about 25 neurons at layer 19 and 33 at layer 20. Across all 50 MLPs, dense 32k diagnostics observed no wrong-sign samples for the neurons classified at layers 17-20.

The result is positive but clearly in diminishing-return territory. M is the standalone winner; extending still earlier should be considered only with a threshold/controller study rather than blindly continuing the same taper.

## Standalone Artifacts

Both implementations reproduce their experiment caches exactly on a real
depth-32/width-256 MLP, pass `whest validate`, and pass the immutable-array
regression test.

Last fourteen:

```text
submissions/structural_tail14_sobol_32768_remote_safe.tar.gz
SHA256: DB7E0EC296BFE167088751865450468C47A8F62FBFB28088987384B0132CC0E3
```

Last sixteen:

```text
submissions/structural_tail16_sobol_32768_remote_safe.tar.gz
SHA256: CF0C8988EC6AFF89243F36DD3A68E3C358B1F59A3F73DCF4F0E971CFD83D752F
```

Each archive contains `estimator.py`, `samples.npz`, and `manifest.json`.
