# Quadrature and propagated-distribution diagnostics

Experiment started 30 September 2026; report completed 1 October 2026.

## Question and controls

Does the current estimator need more accurate numerical integration, better
propagated means/covariances, or a richer distributional approximation?

The control uses the unchanged published venture baseline with frozen output
calibration. It changes only Gauss-Legendre integration order, from 8 to 16 or
32 nodes. All eight exposed development networks are evaluated with float32
state and four backend threads. No calibration is refitted.

| Quadrature nodes | Average raw final MSE | Change in MSE vs 8 nodes | Budget used | Mean residual seconds | Cap failures |
|---:|---:|---:|---:|---:|---:|
| 8 | 1.380358434e-6 | reference | 2.5824% | 0.4773 | 7/8 |
| 16 | 1.380339256e-6 | -0.001389% | 2.8052% | 0.9166 | 8/8 |
| 32 | 1.380371768e-6 | +0.000966% | 3.2509% | 1.7865 | 8/8 |

Higher orders change output predictions by only about 6e-7 RMS. They do not
meaningfully improve this implementation's accuracy. This does not prove the
integral has zero numerical error, nor rule out other precision improvements.

The residual-time cap is 0.4 seconds. The original 8-node code also exceeded
it in seven runs on this machine today, unlike earlier measurements. These
are numerical comparisons, not valid remote submission scores. The raw data
retains resource-cap failures and failure-aware scores separately.

## Completed distribution results

All four networks completed. The research replica's largest absolute
discrepancy from the original uncalibrated estimator was 4.0532e-6, within
the prespecified 5e-6 tolerance. All three diagnostic unit tests passed.

| Final-layer comparison | Raw MSE or cross-stream risk estimate | Pooled-prediction MSE |
|---|---:|---:|
| Original propagation, no calibration | 4.418990e-6 | same |
| Current calibrated baseline | 1.417601e-6 | same |
| Repair incoming mean only | 2.342338e-7 | same |
| Repair incoming variance only | 4.493100e-6 | 4.503416e-6 |
| Repair both moments | 1.959697e-7 | 2.063045e-7 |
| Repair both, doubled samples per estimate | 1.957061e-7 | 2.063093e-7 |

Sample-dependent rows use the cross-stream estimate explained below; fixed
baselines and the mean-only intervention use ordinary MSE. All values are
averaged over the same four networks. They are raw errors, not adjusted
competition scores. The doubled-sample comparison reuses the same total
draws, regrouped into two streams of 131,072, rather than four of 65,536.

Repairing the final layer's incoming mean alone reduces uncalibrated MSE by
94.7% and reaches about 6x lower error than the current calibrated baseline.
The same qualitative pattern appears on all four networks. Repairing both
moments leaves approximately 1.96e-7 of estimated error; the near-identical
doubled-sample result supports this being a real remaining approximation
error rather than primarily finite-sample noise. This is evidence, not an
exact population oracle or a formal confidence interval.

The mean/variance effects are not an additive decomposition: their errors
can interact and partially cancel. Variance-only repair failing to help does
not mean the covariance is exact or irrelevant. It means it is a poor sole
repair for this particular final readout.

After resetting the full state, propagation error regenerates:

| Reset after layer | Error at first following layer | Final error at layer 16 | Final pooled-prediction MSE |
|---:|---:|---:|---:|
| 4 | 3.024768e-7 | 3.387684e-6 | 3.451377e-6 |
| 8 | 2.773571e-7 | 1.778674e-6 | 1.838343e-6 |
| 12 | 2.328802e-7 | 8.259559e-7 | 8.608649e-7 |

These are cross-stream risk estimates for the uncalibrated suffix. The
reset at layer 12 repairs its mean/covariance, but four further Gaussian
closure updates still produce 8.26e-7 final error. Combined with the local
readout interventions, this points toward reducing repeated mean-update
bias and feeding corrected state forward. It does not yet identify whether
cumulants, a mixture, learned local corrections, or sample residuals will
give the best accuracy per unit of compute.

Machine-readable results:

- `results/quadrature_order_20260930.json`: each paired numerical/cost result.
- `results/distribution_oracles_20260930.json`: full per-network/layer diagnostics.
- `results/distribution_oracles_summary_20260930.json`: aggregate curves.
- `results/distribution_oracles_20260930_mlp*.npz`: reference, propagated, and intervention arrays.

## Oracle interventions

The distribution diagnostic uses the first four already-exposed development
networks. Reference post-ReLU means are the dataset's baked 1e9-sample targets.
The corresponding preactivation means follow from the exact linear relation
`m_l = reference_mean_(l-1) @ W_l` (and zero for the first layer).

Preactivation variances and postactivation checkpoint covariances are estimated
by forwarding 65,536 fresh standard-normal inputs in each of four independent
streams per network: 262,144 samples per network in total. Samples use a fixed
new seed schedule, not the dataset's seed. Batch size is 2,048. This is offline
research and is not constrained to the competition inference budget.

For each neuron define the Gaussian ReLU readout

`G(m,v) = m Phi(m/sqrt(v)) + sqrt(v) phi(m/sqrt(v))`.

Compare the following at every layer, holding the incoming baseline state
fixed. These are local readout interventions, not corrections fed forward:

1. Baseline: `G(propagated_mean, propagated_variance)`.
2. Mean repaired: `G(reference_mean, propagated_variance)`.
3. Variance repaired: `G(propagated_mean, sampled_reference_variance)`.
4. Both repaired: `G(reference_mean, sampled_reference_variance)`.

If both repaired moments leave a substantial error, the Gaussian marginal
readout itself is inadequate. If they remove most of the error, inaccurate
propagated moments are the immediate problem. That latter result would not
vindicate Gaussian propagation: non-Gaussian joint structure can cause the
incorrect covariance updates that produced those bad moments.

Separately, at postactivation layers 4, 8, and 12, reset the entire mean vector
and full covariance matrix to reference/sample estimates and propagate the
remaining layers using the unchanged uncalibrated closure. This measures
whether error regenerates after repairing the state. The Gaussian closure
does not retain higher-order structure after a reset.

Calibration is excluded from interventions because its coefficients compensate
for the original propagation errors. Reusing them after an oracle repair
would confound the diagnostic. The calibrated baseline is reported separately
as the practical accuracy target.

## Sampling uncertainty

Naively squaring an error from sampled variances/covariances mixes model bias
with Monte Carlo noise. Let `e_s` be an output-error vector from independent
stream `s`. The cross-stream estimate averages `mean(e_s * e_t)` over distinct
stream pairs. Its expectation is the squared bias of the finite-sample
estimator, relative to the supplied target. It removes independent variance;
it does **not** remove nonlinear plug-in bias or shared reference-target noise.
It can be negative and is not an exact noise-free oracle measurement.

The report also retains individual-stream MSE, pooled-prediction MSE, and
between-stream variance. For the both-moments intervention, pairs of streams
are pooled to double the variance-estimation sample size as a sensitivity
check. Four networks and four streams form a diagnostic scout, not a reliable
generalization estimate or leaderboard prediction.

## Next discriminating experiment

Measure the local innovation, rather than fitting accumulated output error:

`innovation_l = reference_post_mean_l - G(reference_pre_mean_l, reference_pre_variance_l)`.

Estimate standardized skewness and excess kurtosis of preactivations using
independent streams, and test whether they explain this innovation on networks
excluded from fitting. Compare a Gaussian readout against a regularized
higher-moment correction using the same reference mean and variance. This
would test whether third/fourth moments are sufficient before investing in
their propagation. Oracle higher moments alone would not make an estimator
deployable: those moments must subsequently be estimated within the inference
budget, with dependence between neurons handled adequately.

For a practical candidate, feed corrected means into subsequent linear
layers and recompute covariance updates consistently. Compare output-only
correction, mean feedback, and joint mean/covariance feedback. The existing
frozen output calibration was fitted for a different state trajectory; it
should not simply be assumed valid after feedback. Assess the complete
recursive estimator on networks excluded from fitting, with its calibration
and regularization frozen, and rerun resource-cap validation.

If higher-moment oracle corrections fail to explain the innovation, use that
as evidence to investigate a richer approximation such as a small mixture
or a sample-based residual correction. These are proposed experiments, not
methods validated by the current results.

## Reproduction and artifacts

From `whest-starterkit` in PowerShell:

```powershell
& .venv-phase2/Scripts/python.exe research/phase2/test_quadrature_order.py --arrow ../tmp/phase2_mini_first8.arrow --output research/phase2/results/quadrature_order_20260930.json
& .venv-phase2/Scripts/python.exe research/phase2/diagnose_distribution.py --arrow ../tmp/phase2_mini_first8.arrow --output research/phase2/results/distribution_oracles_20260930.json
& .venv-phase2/Scripts/python.exe research/phase2/summarize_distribution.py research/phase2/results/distribution_oracles_20260930.json --output research/phase2/results/distribution_oracles_summary_20260930.json
& .venv-phase2/Scripts/python.exe -m pytest research/phase2/test_distribution_diagnostics.py -q
```

Run numerical benchmarks sequentially to avoid timing contention. The
distribution script checks agreement of its NumPy research replica against
the original uncalibrated estimator on each full-sized network (maximum
absolute discrepancy must be below 5e-6). Unit tests check the Gaussian ReLU
identity, restart indexing, and the cross-stream risk calculation.

The baseline source and submission archive are unchanged. Reference targets,
sampled oracle covariances, and this research harness must not enter an
inference submission.
