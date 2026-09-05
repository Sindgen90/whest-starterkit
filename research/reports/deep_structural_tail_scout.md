# Deep Structural Tail Scout

The exact 32,768-point antithetic Sobol asset was used on the first 50 mini-split MLPs. Dense propagation is diagnostic only; every candidate's measured compute includes its own sampled propagation plus corrected K2.

| Candidate | Tail thresholds | Raw final MSE | FLOPs | Full adjusted | Flop-only adjusted | Structural vs dense |
|---|---:|---:|---:|---:|---:|---:|
| `current_last2_t3` | 3, 3 | 4.9154e-07 | 1.3491e+11 | 2.7920e-07 | 2.4361e-07 | 2.6409e-10 |
| `A_last3_t3` | 3, 3, 3 | 4.9157e-07 | 1.3264e+11 | 2.7515e-07 | 2.3949e-07 | 3.5411e-10 |
| `B_last4_conservative` | 3.5, 3.5, 3, 3 | 4.9146e-07 | 1.3115e+11 | 2.7271e-07 | 2.3667e-07 | 2.9331e-10 |
| `C_last6_tapered` | 3.5, 3.4, 3.3, 3.2, 3, 3 | 4.9172e-07 | 1.2735e+11 | 2.6544e-07 | 2.2984e-07 | 3.3039e-10 |

## Paired Comparison

Each MLP uses the lambda fitted on the opposite parity. Confidence intervals are paired 20,000-draw bootstrap intervals for candidate minus current-last-two.

| Candidate | Raw delta | Raw delta 95% CI | Adjusted delta | Adjusted delta 95% CI | FLOP change |
|---|---:|---:|---:|---:|---:|
| `A_last3_t3` | +0.005% | [-6.5516e-10, 7.0423e-10] | -1.450% | [-5.2287e-09, -3.0023e-09] | -1.677% |
| `B_last4_conservative` | -0.016% | [-5.1308e-10, 2.3179e-10] | -2.323% | [-8.1976e-09, -4.9973e-09] | -2.786% |
| `C_last6_tapered` | +0.036% | [-5.5250e-10, 7.6383e-10] | -4.929% | [-1.6762e-08, -1.1120e-08] | -5.597% |

## Interpretation

Candidate C reduces mean FLOPs by 5.60% and full adjusted score by 4.93%. Its raw final MSE changes by only +0.036%, while structural-versus-dense final MSE remains 3.3039e-10.

The raw difference is too small to call an accuracy improvement. The compute reduction is the result: C is the packaging candidate because its adjusted score improves in both parity directions and under both full and FLOP-only accounting.

The parity directions agree on the decision:

| Calibration parity | Current raw | C raw | Current adjusted | C adjusted |
|---:|---:|---:|---:|---:|
| 0 | 5.6623e-07 | 5.6587e-07 | 3.2167e-07 | 3.0461e-07 |
| 1 | 4.1685e-07 | 4.1757e-07 | 2.3673e-07 | 2.2626e-07 |

## Structural Pattern

Candidate C starts at layer 27. Its thresholds taper from 3.5 to 3.0, retaining more uncertain neurons at the earlier layers where a wrong classification has more opportunities to amplify downstream. The observed per-sample wrong-sign rates are about 0.0007% at layer 27 and rise toward 0.004% in the final two layers.

## Submission Artifact

The grader-safe implementation is in:

```text
submissions/structural_tail6_sobol_32768/estimator.py
```

The upload archive is:

```text
submissions/structural_tail6_sobol_32768_remote_safe.tar.gz
SHA256: 0993F2AA39A65314C0019166E43615469B148D9B478AABEFB4367D6120AFD837
```

It contains `estimator.py`, `samples.npz`, and `manifest.json`. `whest validate`
passes, a real depth-32/width-256 setup-and-predict smoke returns finite values,
and the submission's structural prediction matches the experiment cache exactly.

The canonical local runner also completed all 100 Mini MLPs in `260.92s` with
no failures:

| Metric | All 100 Mini MLPs |
|---|---:|
| Raw final-layer MSE | `6.29e-7` |
| All-layers MSE | `1.31e-6` |
| Adjusted final-layer score | `3.84e-7` |
| Mean compute utilization | `0.61133377` |
| Failed MLPs | `0 / 100` |
