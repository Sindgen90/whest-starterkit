# Phase 2 skewness and kurtosis experiment

Date: 1 October 2026. Offline diagnostic; no new submission is packaged.

## Completed local-readout results

All eight networks completed, with 2,097,152 input draws in total. On the
four evaluation networks excluded from fitting:

| Final-layer readout | Ordinary pooled-prediction MSE | Cross-stream risk estimate |
|---|---:|---:|
| Gaussian | 2.010140e-7 | 1.910508e-7 |
| Gaussian + skewness (K3) | 2.907826e-8 | 1.757327e-8 |
| Gaussian + kurtosis (K4) | 2.147224e-7 | 2.041975e-7 |
| Gaussian + K3 + K4 | **1.123182e-8** | -7.201863e-10 |
| Gaussian + K3 + K4 + K3-squared | 1.124830e-8 | -7.396053e-10 |
| Development-fitted K3/K4 coefficients | 1.108260e-8 | -7.977737e-10 |

The fixed K3+K4 formula reduces the observed pooled-prediction error by 94.4%
(17.9x). Its slightly negative cross-stream estimate means the remaining
squared bias is unresolved at this sampling precision, not that true error
is negative or zero. Doubling samples per stream gives a pooled MSE of
1.122994e-8 and cross estimate -8.751309e-10: the conclusion is unchanged.
The pooled predictions average separate readouts; the double-sample variant
also recomputes the cumulants after pooling the raw moment statistics.

The estimated Gaussian local bias is about 1.91e-7; K3 alone reduces it to
1.76e-8. K4 alone is slightly harmful at the final layer, yet improves the
K3-corrected readout. Their contributions should therefore be evaluated
jointly. A fit offers only a 1.3% pooled-error reduction over fixed K3+K4,
without a convincing resolved bias reduction. Retain the simpler analytical
formula for subsequent diagnostics; no evaluation-set coefficients are fitted.

The result extends across depth:

| Layer | Gaussian cross-stream risk | K3 cross-stream risk | K3+K4 cross-stream risk |
|---:|---:|---:|---:|
| 2 | 1.49736e-7 | 1.96263e-8 | 4.87613e-9 |
| 4 | 2.87402e-7 | 2.84345e-8 | -2.26662e-9 |
| 8 | 2.52247e-7 | 2.84685e-8 | -4.10038e-10 |
| 12 | 2.18434e-7 | 2.13364e-8 | 8.44752e-11 |
| 16 | 1.91051e-7 | 1.75733e-8 | -7.20186e-10 |

These are local oracle readouts, with the reference incoming mean supplied at
each layer. They do not yet measure recursive propagation or adjusted scores.
All seven selected unit tests passed, including independent integration of
the correction terms and verification that mean feedback affects the next
linear layer without accidentally changing the current covariance update.

## Completed mean-only feedback results

On the same four evaluation networks, with two independent streams of
131,072 samples each:

| Full recursive estimator | Ordinary MSE / pooled-prediction MSE | Cross-stream risk estimate |
|---|---:|---:|
| Uncalibrated Gaussian propagation | 4.386181e-6 | same |
| Current calibrated baseline | 1.416969e-6 | same |
| Oracle skewness, mean feedback | 1.346651e-6 | 1.323147e-6 |
| Oracle skewness + kurtosis, mean feedback | 1.256049e-6 | 1.221695e-6 |
| Reference local-bias feedback control | 1.325684e-6 | 1.137306e-6 |

The local 17.9x reduction does not become a similar end-to-end gain when
Gaussian covariance updates are retained. K3+K4 mean feedback gives only an
11.4% reduction in observed error versus the calibrated baseline (13.8% using
the estimated bias comparison), despite receiving oracle shape information.
This is not sufficient evidence to promote a costly higher-moment estimator.

## Exploratory variance controls

The diagonal-variance follow-up worsened the recursive result to 2.852043e-6
observed pooled MSE (2.816581e-6 cross-stream risk). Correcting marginal first
and second moments while leaving the cross-neuron covariance update Gaussian
is not enough to obtain a good joint propagation rule.

A final inexpensive control bypassed covariance propagation: at every layer,
recursively compute the incoming mean from the previous predicted mean, but
supply sampled true incoming variances. Compare Gaussian and K3+K4 readouts;
the first layer uses its analytically known variance and zero higher cumulants.

| Recursive update with sampled incoming variances | Pooled final MSE | Cross-stream risk |
|---|---:|---:|
| Gaussian | 5.983345e-6 | 5.798759e-6 |
| K3+K4 | **1.961240e-7** | 5.864891e-9 |

The K3+K4 result is 6.4x lower in observed error than K3+K4 mean feedback
with propagated covariances. The much smaller cross-stream estimate suggests
substantial remaining sampling noise, but two pooled streams cannot establish
a precise population bias at this scale. This control was chosen after the
preceding results and uses the same exposed evaluation set.

Together these results support a specific direction: scalar K3/K4 information
is useful, but obtaining accurate incoming variances under corrected mean
propagation requires better joint moment/covariance handling. Marginal
corrections alone should not be promoted. Mixed cumulant transport and the
off-diagonal covariance update are the next research targets; they still need
an affordable implementation and new evaluation before any submission claim.

The best eligible Phase 2 method remains the published calibrated covariance
baseline. A subsequent runtime optimization fixes observed local cap failures
while preserving its accuracy; the upload is `submissions/phase2_best_20261001.tar.gz`.
See [RUNTIME_BASELINE.md](RUNTIME_BASELINE.md) for the paired comparison and
package validation. The oracle experiments are not packaged because they use
reference information and unbudgeted sampling. They do not establish a more
accurate uploadable estimator.

## Questions

1. With accurate incoming means and sampled distributional moments, do K3/K4
   explain the scalar ReLU update error in width-1024, depth-16 networks?
2. Does the result transfer to networks excluded from coefficient fitting?
3. Does using sampled shape information in repeated mean updates reduce final
   error, even while retaining the Gaussian covariance approximation?

These are separate from whether higher cumulants can be propagated cheaply.

## Fixed protocol

- Development: first four networks in `phase2_mini_first8.arrow` (IDs 0-3).
- Evaluation: first four in `phase2_mini_validation8.arrow` (IDs 8-11).
- Four independent streams of 65,536 Gaussian inputs per network; 262,144
  total draws per network. Seed schedule uses MLP IDs as identifiers, not the
  dataset's sampling seeds. Batch size 2,048; four backend threads.
- Reference means are the dataset's baked 1e9-sample means. Incoming means
  are obtained by exact linear transformation of the preceding reference mean.
- Variance is estimated about that incoming reference mean. Third and fourth
  cumulants use sample-centered, unbiased k-statistics to reduce sampling bias.
- Main methods have no fitted parameters: Gaussian, K3 only, K4 only, K3+K4,
  and K3+K4 plus the second-order K3-squared term. No clipping or model selection.
- A secondary fit learns two coefficients per layer for the K3 and K4 terms.
  Its normalized ridge penalty is fixed at 0.001, toward analytical weights
  (1,1), with no intercept. Development streams are pooled; coefficients are
  written to disk before the evaluation shard is loaded.
- Evaluation networks were inspected in earlier project experiments. They are
  excluded from this fit; they are not an untouched competition holdout.

The two pooled-stream sensitivity check uses 131,072 samples per estimate,
regrouping the same draws rather than generating additional independent data.

## Correction equations

Let `a = m/sigma`, `s = kappa3/sigma^3` (skewness), and
`k = kappa4/sigma^4` (excess kurtosis). The Gaussian ReLU mean is

`G = m Phi(a) + sigma phi(a)`.

The tested corrections are

```text
d3  = sigma phi(a) * (-a) * s / 6
d4  = sigma phi(a) * (a^2 - 1) * k / 24
d33 = sigma phi(a) * (a^4 - 6a^2 + 3) * s^2 / 72
```

They follow by integrating the H3, H4, and H6 terms of a truncated Hermite
density expansion against ReLU. The implementation is checked against
independent numerical integration, including nonzero means and nonunit scale.
This is a truncation, not an exact distribution representation or a guarantee
of positivity. The K3-squared term need not improve an arbitrary network.

For background, [Balieiro Filho and Rosenfeld, *Testing Option Pricing with
the Edgeworth Expansion*](https://arxiv.org/abs/cond-mat/0401192) applies
skewness/kurtosis expansions to option payoffs. The ReLU-specific equations
above are directly derived and numerically checked in this experiment.

## Error estimates and limits

`cross_stream_risk` averages error-vector inner products between independent
streams. It removes their independent variance contribution in expectation.
It estimates the squared bias of the finite-sample procedure relative to the
baked target; it does not eliminate nonlinear plug-in bias or shared target
noise. Estimates can be negative near the sampling-resolution limit. Such
values must not be described as negative true MSE or proof of zero error.

The JSON also retains ordinary pooled-prediction and individual-stream MSE.
Paired comparisons share samples, and the doubled-sample check assesses
sensitivity to moment-estimation noise. No leaderboard gain can be inferred
directly from these oracle readouts.

## Mean feedback test

Use the same evaluation networks, with two independent pooled moment streams.
At every layer, supply sampled standardized skewness/kurtosis to the correction
formula evaluated at the currently propagated mean and variance. Feed the
corrected mean into the following layer. Preserve the baseline Gaussian
covariance update as a controlled ablation; this is not claimed to be a fully
consistent higher-order covariance closure. No existing output calibration is
applied after the intervention.

Compare K3 feedback, K3+K4 feedback, and feedback of the reference local bias
`reference_post_mean - G(reference_pre_mean, sampled_pre_variance)`.
The latter is an additional diagnostic control. Its reference-state local
correction is carried into a changing approximate trajectory, so it is not a
mathematical lower bound on final error. The first preactivation is exactly
Gaussian, so all first-layer shape/bias corrections are set to zero.

An exploratory follow-up, selected after viewing these same evaluation
networks, also corrects the second raw ReLU moment:

`d_second = variance * phi(a) * (skewness/3 - a*excess_kurtosis/12)`.

The new diagonal variance is `Gaussian_second_raw + d_second - corrected_mean^2`.
Off-diagonal covariance still uses the Gaussian update. This expression is
also checked against numerical integration of the Hermite density expansion.
This follow-up is a diagnosis on an exposed set, not new independent validation.

## Relation to Phase 1

The earlier `research/reports/activation_update_oracle_development_limit8.md`
already found useful scalar K3/K4 corrections with sampled moments. Practical
cumulant propagation was the difficult part. It used 8,192 samples, selected
neurons, and empirical sample readouts, unlike the current independent-stream
comparison against baked targets. These experiments are not a matched test
of width/depth effects. A Phase 2 gain by itself does not establish that the
architecture change caused a qualitative improvement over Phase 1.

## Reproduction

From `whest-starterkit` in PowerShell, run sequentially:

```powershell
& .venv-phase2/Scripts/python.exe -m pytest research/phase2/test_higher_moment_diagnostics.py research/phase2/test_distribution_diagnostics.py -q
& .venv-phase2/Scripts/python.exe research/phase2/diagnose_higher_moments.py --development ../tmp/phase2_mini_first8.arrow --validation ../tmp/phase2_mini_validation8.arrow --output research/phase2/results/higher_moments_20261001.json
& .venv-phase2/Scripts/python.exe research/phase2/test_higher_moment_feedback.py --validation ../tmp/phase2_mini_validation8.arrow --oracles research/phase2/results/higher_moments_20261001.json --output research/phase2/results/higher_moment_feedback_20261001.json
& .venv-phase2/Scripts/python.exe research/phase2/test_higher_moment_feedback.py --validation ../tmp/phase2_mini_validation8.arrow --oracles research/phase2/results/higher_moments_20261001.json --output research/phase2/results/higher_moment_diagonal_feedback_20261001.json --joint-diagonal-only
& .venv-phase2/Scripts/python.exe research/phase2/test_oracle_variance_feedback.py --validation ../tmp/phase2_mini_validation8.arrow --oracles research/phase2/results/higher_moments_20261001.json --output research/phase2/results/higher_moment_oracle_variance_feedback_20261001.json
```

Per-network moment arrays are saved as `higher_moments_20261001_mlp*.npz`
alongside the JSON. They support feedback tests without generating new samples.
The submitted baseline source and archive are not modified.
