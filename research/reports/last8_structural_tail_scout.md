# Last-Eight Structural Tail Scout

The exact 32,768-point antithetic Sobol rule was evaluated on the balanced first 50 Mini MLPs. Current-last-two and C-last-six were rerun as same-process controls.

| Candidate | Thresholds, layers 25-32 | Raw final MSE | FLOPs | Full adjusted | Structural vs dense |
|---|---:|---:|---:|---:|---:|
| `current_last2_t3` | 3, 3 | 4.9154e-07 | 1.3491e+11 | 2.7682e-07 | 2.6409e-10 |
| `C_last6_tapered` | 3.5, 3.4, 3.3, 3.2, 3, 3 | 4.9172e-07 | 1.2735e+11 | 2.6385e-07 | 3.3039e-10 |
| `D_last8_conservative` | 4.25, 4, 3.8, 3.6, 3.4, 3.2, 3, 3 | 4.9178e-07 | 1.2516e+11 | 2.6074e-07 | 3.2067e-10 |
| `E_last8_balanced` | 4, 3.8, 3.6, 3.5, 3.4, 3.2, 3, 3 | 4.9169e-07 | 1.2475e+11 | 2.6002e-07 | 3.2370e-10 |
| `F_last8_aggressive` | 3.75, 3.6, 3.5, 3.4, 3.3, 3.2, 3, 3 | 4.9185e-07 | 1.2435e+11 | 2.5865e-07 | 3.3344e-10 |

## Paired Against C

Confidence intervals are paired 20,000-draw bootstrap intervals for candidate minus C-last-six. Each MLP uses the lambda fitted on the opposite parity.

| Candidate | Raw change | Raw delta 95% CI | Adjusted change | Adjusted delta 95% CI | FLOP change |
|---|---:|---:|---:|---:|---:|
| `D_last8_conservative` | +0.013% | [-1.5654e-10, 3.0120e-10] | -1.176% | [-4.0280e-09, -2.2182e-09] | -1.719% |
| `E_last8_balanced` | -0.006% | [-1.9784e-10, 1.3038e-10] | -1.452% | [-4.9188e-09, -2.8342e-09] | -2.043% |
| `F_last8_aggressive` | +0.027% | [-1.1789e-10, 5.2662e-10] | -1.970% | [-6.3119e-09, -4.1405e-09] | -2.362% |

## Parity Check

| Candidate | Parity-0 raw | Parity-0 adjusted | Parity-1 raw | Parity-1 adjusted |
|---|---:|---:|---:|---:|
| `C_last6_tapered` | 5.6587e-07 | 3.0281e-07 | 4.1757e-07 | 2.2489e-07 |
| `D_last8_conservative` | 5.6606e-07 | 2.9910e-07 | 4.1751e-07 | 2.2239e-07 |
| `E_last8_balanced` | 5.6599e-07 | 2.9821e-07 | 4.1739e-07 | 2.2182e-07 |
| `F_last8_aggressive` | 5.6613e-07 | 2.9731e-07 | 4.1758e-07 | 2.1999e-07 |

## Decision

F reduces FLOPs by 2.36% and measured adjusted score by 1.97% relative to C. Its raw-MSE change is +0.027%, with a confidence interval spanning zero. The adjusted improvement appears in both parity directions and under FLOP-only accounting.

F is therefore the last-eight packaging candidate. E remains the conservative fallback if a future untouched split shows that F's earlier classifications amplify more than expected.

## Classification Detail

At each layer corrected K2 supplies a Gaussian preactivation mean `mu` and standard deviation `sigma`. The z-score `alpha = mu / sigma` defines the classes:

- active: `alpha >= threshold`; propagate the preactivation linearly;
- dead: `alpha <= -threshold`; omit the neuron as zero;
- uncertain: `abs(alpha) < threshold`; evaluate sampled preactivations and apply ReLU.

F uses thresholds `3.75, 3.6, 3.5, 3.4, 3.3, 3.2, 3.0, 3.0` from layers 25 through 32. On the parity-0 evaluation half it retained, on average, 158.3 uncertain neurons at layer 25 and 149.0 at layer 26. Empirical 32k wrong-sign rates there were approximately `1.9e-6` and `4.0e-6`, respectively.

A next correction experiment should preserve the same classes but add Gaussian tail-mass mean corrections: a small positive correction for the clipped negative tail of active neurons and a small positive expected activation for dead neurons. This can potentially reduce accumulated bias without restoring full sampled matrix products.

## Submission Artifact

The promoted estimator is:

```text
submissions/structural_tail8_sobol_32768/estimator.py
```

The folder-mode upload archive is:

```text
submissions/structural_tail8_sobol_32768_remote_safe.tar.gz
SHA256: 15D8A0B249BB9A0AC87CB4AF026E58F50FF4E622D907E4A4E2CDBFF6CBADABC3
```

It contains `estimator.py`, `samples.npz`, and `manifest.json`. `whest validate`
passes, the implementation matches the F experiment cache exactly on a real
depth-32/width-256 MLP, and the immutable-array regression test passes.

The canonical local runner completed all 100 Mini MLPs in `269.52s`:

| Metric | All 100 Mini MLPs |
|---|---:|
| Raw final-layer MSE | `6.29e-7` |
| All-layers MSE | `1.31e-6` |
| Adjusted final-layer score | `3.79e-7` |
| Mean compute utilization | `0.60454269` |
| Failed MLPs | `0 / 100` |
