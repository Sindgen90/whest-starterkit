# Phase 2 baseline replication — 26 September 2026

Use **`venture_baseline/estimator.py`** as the reproduced public baseline.
It is byte-identical to venture_agent_ai's published Phase 2 submission
329951 source at commit `fb33bb5a66b4b0aed3022e6a70a9274d5957d61d`.
The author's MIT license and original coefficients are preserved. No local
coefficient fitting or seed selection was performed.

Package: **`venture329951_phase2_baseline.tar.gz`**. It contains the estimator,
license, and provenance. It has been validated locally, not submitted to AIcrowd.

## Measured comparison

Same networks, baked 1e9-sample public targets, float32 weights, four CPU
threads, flopscope 0.12.1 and whestbench 0.16.1. Score is the average of
per-network `MSE * max(0.1, FLOPs / 2**41)`. Failures remain in the reports
and are charged as zero predictions with no discount.

| Estimator | Networks 0–7 adjusted MSE | Networks 8–15 adjusted MSE | All 16 adjusted MSE | Budget used |
|---|---:|---:|---:|---:|
| Gain-covariance starter baseline | 4.39423e-7 | 4.39208e-7 | 4.39315e-7 | 2.351% |
| Existing 0.75/0.25 covariance/whitened-MC blend | 2.85377e-7 | 3.17136e-7 | 3.01256e-7 | 9.539% |
| Published calibrated bivariate-covariance estimator | **1.38036e-7** | **1.37341e-7** | **1.37688e-7** | **2.582%** |

The published estimator reduces adjusted error by **54.3% versus the existing
blend** and **68.7% versus gain covariance** on these 16 networks. All runs
passed the local caps. The MC comparator uses the independent seed 20260926;
the new estimator is deterministic and does not use its RNGs numerically.

These are two previously exposed public Mini shards, not a fresh untouched
holdout. Networks 8–15 were not used for any tuning during this replication.
The upstream coefficient-training corpus has not been independently audited
for overlap. This establishes a reproducible local baseline, not a private
leaderboard score or an independent statistical reproduction of the author's
entire evaluation.

## What is being replicated

The method propagates mean and full covariance through the weights. It
computes the bivariate Gaussian ReLU covariance using an exact integral
identity evaluated by eight-node Gauss–Legendre quadrature. A frozen five-term
correction is applied to each returned mean row. The corrected row is **not**
fed back into the propagation state.

An ablation on networks 0–7, disabling only the published calibration, gives
4.12596e-7 adjusted versus 1.38036e-7 with calibration. Thus most of this
baseline's gain comes from its calibrated readout, not the covariance formula
alone. Gaussian closure remains an approximation to the real network law.

Sources:
- [Author's Phase 2 write-up](https://discourse.aicrowd.com/t/phase-2-write-up-exact-bivariate-relu-covariance-plus-per-layer-calibration-3x-over-the-kit-baseline-with-negative-results-submission-329951/18214)
- [Pinned source](https://github.com/agentaiventure-dot/arc-whest-estimator/blob/fb33bb5a66b4b0aed3022e6a70a9274d5957d61d/estimator/estimator.py)
- Local provenance and SHA-256: `venture_baseline/PROVENANCE.md`.

## Verification

- Twelve numerical tests pass across the existing and newly added Phase 2
  components. New checks independently verify first-layer Gaussian mean and
  arccosine covariance identities, nonstandard-depth behavior, Kerdock mutual
  unbiasedness, FWHT multiplication, and the dual ridge application.
- Official `whest validate`: success, including its 4-wide, 2-deep smoke shape.
- Official full-width subprocess: zero failures, 56,786,552,287 FLOPs,
  2.58235% utilization, 4.468 seconds predict wall time, 0.3043 seconds
  residual time, under the 0.4-second cap. This is local timing, not a remote
  performance guarantee. Residual-time headroom is material and should be
  checked on the actual grader.
- The subprocess smoke target uses only 100 MC samples. Its MSE is intentionally
  excluded from every accuracy comparison above.
- Official package validator: success. Archive contains no datasets or test code.

The first plain-output validator invocation hit a Windows console encoding
error while printing its result. Rerunning with `PYTHONIOENCODING=utf-8` and
JSON output succeeded; the estimator source was unchanged.

## What happened to the Puffi/Kerdock route

An executable partial reconstruction is retained in `kerdock_baseline/`, with
its geometry tests and offline asset generator. It generalizes the published
Kerdock construction to width 1024 and implements Puffi's ridge equations with
the existing approximate analytical reference.

All eight tested configurations lost to the existing blend on networks 0–7.
The best tested partial-frame result was 6.27428e-7 adjusted; the default
8-frame/layer-5 control variant was 7.17895e-7. These are **not promoted**.
This small-frame, dense implementation omits the original pruning, full frame
bank, terminal bypass, and unpublished reference coefficients, so this result
does not refute Puffi's original method. It does show that this particular
partial transplant is not a strong Phase 2 baseline.

## Leaderboard implications

The [public board](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards)
retrieved on 26 September showed J2W at about 2.0e-9 adjusted, 1.61e-8 raw,
12.74% utilization; suliman_tadros and Luna about 2.1e-9 adjusted; Puffi
about 2.7e-9. These are public standings, not final winners.

Inference: similar raw errors paired with different utilization show that
arithmetic efficiency matters to the leading ranking. They do not identify
the hidden algorithms. Puffi's public Phase 1 hybrid is a direct research
lead, not proof of anyone's current Phase 2 implementation. Our roughly
1.38e-7 baseline remains far from that frontier; different evaluation cohorts
prevent treating the numerical gap as a precise rank prediction.

## Reproduce

From the starter-kit directory in PowerShell:

```powershell
$env:PYTHONIOENCODING='utf-8'
& ./.venv-phase2/Scripts/python.exe -m pytest research/phase2/test_public_baseline.py -q
& ./.venv-phase2/Scripts/python.exe research/phase2/benchmark_kerdock.py --arrow ../tmp/phase2_mini_first8.arrow --output research/phase2/results/public_baseline_replication_development.json --methods covariance blend venture_uncalibrated venture
& ./.venv-phase2/Scripts/python.exe research/phase2/benchmark_kerdock.py --arrow ../tmp/phase2_mini_validation8.arrow --output research/phase2/results/public_baseline_replication_validation.json --methods covariance blend venture
& ./.venv-phase2/Scripts/whest.exe validate --estimator research/phase2/venture_baseline/estimator.py --format json
& ./.venv-phase2/Scripts/whest.exe run --estimator research/phase2/venture_baseline/estimator.py --runner subprocess --n-mlps 1 --n-samples 100 --seed 20260926 --max-threads 4 --format json
& ./.venv-phase2/Scripts/whest.exe validate-package research/phase2/venture329951_phase2_baseline.tar.gz --format plain
```

Detailed records: `results/public_baseline_replication_development.json`,
`results/public_baseline_replication_validation.json`,
`results/kerdock_development.json`, `results/venture_validate.json`, and
`results/venture_subprocess_smoke.json`.
