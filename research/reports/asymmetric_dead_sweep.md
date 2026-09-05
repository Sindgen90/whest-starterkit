# Asymmetric Dead Sweep

All candidates use corrected K2 plus the fixed 32,768-point antithetic Sobol asset. They were evaluated in balanced order on the first 50 Mini MLPs with opposite-parity lambda fitting. Dense sampling is diagnostic only.

## Results

| Candidate | Mechanism | Raw final MSE | FLOPs | Full adjusted | Structural vs dense |
|---|---|---:|---:|---:|---:|
| `M_last16_symmetric_control` | symmetric control | 4.9190e-07 | 1.1933e+11 | 2.5506e-07 | 3.3364e-10 |
| `N_last16_dead_only_same` | dead-only, same thresholds | 4.9180e-07 | 1.1934e+11 | 2.4943e-07 | 2.8209e-10 |
| `O_last16_dead_only_m025` | dead-only, threshold -0.25 | 4.9198e-07 | 1.1760e+11 | 2.4750e-07 | 3.9133e-10 |
| `P_last16_dead_only_m050` | dead-only, threshold -0.50 | 4.9216e-07 | 1.1577e+11 | 2.4413e-07 | 9.1455e-10 |
| `Q_last16_early_dead_m075` | dead-only -0.75 on layers 17-20 | 4.9190e-07 | 1.1810e+11 | 2.5032e-07 | 3.3382e-10 |

## Paired Against M

Intervals are paired 20,000-draw bootstrap intervals for candidate minus M. Negative changes are improvements.

| Candidate | Raw change | Raw delta 95% CI | Structural change | FLOP change | Adjusted change |
|---|---:|---:|---:|---:|---:|
| `N_last16_dead_only_same` | -0.020% | [-5.0302e-10, 2.7374e-10] | -15.45% | +0.014% | -2.209% |
| `O_last16_dead_only_m025` | +0.016% | [-5.6095e-10, 7.6125e-10] | +17.29% | -1.450% | -2.964% |
| `P_last16_dead_only_m050` | +0.053% | [-1.2462e-09, 1.5842e-09] | +174.11% | -2.977% | -4.284% |
| `Q_last16_early_dead_m075` | +0.000% | [-8.5316e-12, 1.3663e-11] | +0.06% | -1.023% | -1.858% |

## Parity Check

| Candidate | Parity-0 raw | Parity-1 raw | Parity-0 adjusted | Parity-1 adjusted |
|---|---:|---:|---:|---:|
| `M_last16_symmetric_control` | 5.6619e-07 | 4.1761e-07 | 2.9250e-07 | 2.1762e-07 |
| `N_last16_dead_only_same` | 5.6647e-07 | 4.1712e-07 | 2.8640e-07 | 2.1245e-07 |
| `O_last16_dead_only_m025` | 5.6625e-07 | 4.1771e-07 | 2.8462e-07 | 2.1038e-07 |
| `P_last16_dead_only_m050` | 5.6739e-07 | 4.1693e-07 | 2.8083e-07 | 2.0744e-07 |
| `Q_last16_early_dead_m075` | 5.6619e-07 | 4.1762e-07 | 2.8665e-07 | 2.1399e-07 |

## Mean Classification Counts

The table shows the mean number of neurons omitted as dead. N/P intentionally classify no neuron as always-active on layers 17-30; those neurons stay in the sampled ReLU path. Layers 31-32 retain the established symmetric classifier.

| Layer | M dead | M active | N dead | P dead | Q dead |
|---:|---:|---:|---:|---:|---:|
| 17 | 4.5 | 4.7 | 4.5 | 8.6 | 11.3 |
| 18 | 8.1 | 7.1 | 8.1 | 13.6 | 16.9 |
| 19 | 11.1 | 10.9 | 11.1 | 18.5 | 22.4 |
| 20 | 15.8 | 14.5 | 15.8 | 23.4 | 27.8 |
| 21 | 20.1 | 22.2 | 20.1 | 28.8 | 20.1 |
| 22 | 27.3 | 26.9 | 27.3 | 37.8 | 27.3 |
| 23 | 34.5 | 33.4 | 34.5 | 45.4 | 34.5 |
| 24 | 39.8 | 40.7 | 39.8 | 50.4 | 39.8 |
| 25 | 47.9 | 48.9 | 47.9 | 58.2 | 47.9 |
| 26 | 53.0 | 52.7 | 53.0 | 64.0 | 53.0 |
| 27 | 56.9 | 57.8 | 56.9 | 68.0 | 56.9 |
| 28 | 63.5 | 61.5 | 63.5 | 74.5 | 63.5 |
| 29 | 65.4 | 66.4 | 65.4 | 75.6 | 65.4 |
| 30 | 71.0 | 71.2 | 71.0 | 80.8 | 71.0 |
| 31 | 75.5 | 76.7 | 75.5 | 75.5 | 75.5 |
| 32 | 78.7 | 75.3 | 78.7 | 78.7 | 78.7 |

## Interpretation

1. **Always-active linearization was not helping accuracy.** N uses the same dead thresholds as M but leaves every non-dead neuron on layers 17-30 in the sampled ReLU path. Its structural-versus-dense MSE is lower, although the raw-MSE direction differs between parity folds and it does not save counted FLOPs.
2. **Q is the robust conservative efficiency point.** It only makes layers 17-20 more aggressive. Raw error and structural error are effectively unchanged while counted FLOPs fall by about one percent.
3. **P is the aggressive score point.** It saves about three percent counted FLOPs and wins adjusted score in both parity folds, but structural error rises substantially and the raw-MSE direction is not parity-stable. It is useful as an efficiency option, not an accuracy improvement.
4. **The dead decision is analytic, not sample-fitted.** Corrected K2 propagates mean and covariance from the known Gaussian input through the submitted weights. A neuron is declared dead when its standardized preactivation `alpha = mu / sigma` is at or below the layer threshold with a negative sign. The Sobol paths are evaluated only after this classification; dense sample signs are used solely to audit the approximation offline.

## Standalone Artifact

Q is packaged as the conservative standalone winner:

```text
submissions/structural_tail16_early_dead_sobol_32768_remote_safe.tar.gz
SHA256: 77D1772D9ABBFC75CA69442548341F2127C47D692C6BA6EE3A8D6CA0B7964D3D
```

The standalone structural prediction reproduces the cached Q prediction exactly
on a real depth-32/width-256 MLP (`max_abs = 0`). It passes `whest validate`, the
immutable-array/import regression tests, and a two-MLP local runner check with no
failed MLPs. The archive contains `estimator.py`, `samples.npz`, and
`manifest.json`.
