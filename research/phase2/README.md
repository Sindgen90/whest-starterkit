# Phase-two reconnaissance and first replication

**Current upload:** [`../../submissions/phase2_best_20261001.tar.gz`](../../submissions/phase2_best_20261001.tar.gz),
a runtime-optimized derivative of the calibrated public baseline. See
[`RUNTIME_BASELINE.md`](RUNTIME_BASELINE.md) for the numerical comparison,
successful package/subprocess validation, and remaining timing limitation.

**Higher-moment follow-up:** [`HIGHER_MOMENT_DIAGNOSTICS.md`](HIGHER_MOMENT_DIAGNOSTICS.md)
tests sampled skewness/kurtosis, disjoint coefficient evaluation, and recursive
mean feedback at the Phase 2 dimensions.

**1 October update:** see [`DISTRIBUTION_DIAGNOSTICS.md`](DISTRIBUTION_DIAGNOSTICS.md)
for the paired 8/16/32-point quadrature control and offline interventions that
separate propagated moment error from Gaussian closure error.

**26 September update:** a byte-identical replication of published Phase 2
submission 329951 is now in `venture_baseline/`. See
[`BASELINE_REPLICATION.md`](BASELINE_REPLICATION.md) for measured comparisons,
provenance, the packaged baseline, and the rejected partial Kerdock/Puffi scout.
The September 5 leaderboard figures below are historical.

Research date: 5 September 2026. This is a measured starting point, not a recovered winning solution.

## What the current leaders disclose

The [live phase-two leaderboard](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/leaderboards) displayed these rounded numbers when retrieved:

| Displayed rank | Entrant | Adjusted score | Raw final MSE | Compute utilization |
|---|---|---:|---:|---:|
| 1 | marius_binner | 3.8e-9 | 1.87e-8 | 20.48% |
| 1 | suliman_tadros | 3.8e-9 | 2.03e-8 | 18.69% |
| 1 | J2W | 3.8e-9 | 1.99e-8 | 19.24% |
| 4 | Puffi | 3.9e-9 | 2.13e-8 | 18.24% |
| 5 | oqaris | 4.3e-9 | 2.06e-8 | 20.90% |
| 8 | hyojun_kwon | 6.4e-9 | 2.83e-8 | 22.67% |

Scores alone do not identify an algorithm. No public current implementation for the top three was found in the sources searched. Their similar error and cost are compatible with several different methods. They establish an accuracy target, not evidence that a particular method is in use. Rounded ties are not necessarily identical underlying scores, and a public listing is not final eligibility or prize ranking.

**Puffi is the most useful direct lead from a current top team.** Their [phase-one disclosure](https://discourse.aicrowd.com/t/phase-1-write-ups-team-puffi/18175) links a [technical appendix](https://discourse.aicrowd.com/uploads/short-url/iDdUNo66HzI7ovRqhQqpLOFFugI.pdf) describing: exact Gaussian radial integration; Kerdock/Walsh directions with antipodes; a shallow A716 analytical reference; a ridge-fitted transport of the layer-9 discrepancy to the output; probe-based pruning and mean filling; and fast matrix products. A716 carries covariance plus selected third/fourth cumulants and mixed third-cumulant history. This is a disclosed phase-one algorithm; it does not establish their current phase-two implementation. Our transport ablation below reproduces only the *idea* of transporting an early discrepancy, using Gaussian gains and whitened samples. It omits A716, Kerdock frames, ridge fitting, and pruning and is not an exact reproduction of Puffi.

**hyojun_kwon provides another related lead.** Their [phase-one write-up](https://discourse.aicrowd.com/t/phase-1-write-up-monetizing-the-approach-to-the-moment-propagation-frontier-an-intra-network-control-variate-and-a-measured-map-of-the-wall-behind-it-submission-326024/18154) studies intra-network control variates and the tradeoff between control depth and the accuracy of its analytical center. That disclosure is not their current eighth-place code either.

**The explicit phase-two reproduction target is much less competitive.** [baghyeonsig, submission 328274](https://discourse.aicrowd.com/t/phase-2-write-up-complementary-errors-beat-either-branch-a-fixed-0-75-0-25-covariance-whitened-antithetic-mc-blend-submission-328274/18200), discloses a 0.75/0.25 covariance/whitened-antithetic-MC blend. The paper reports 4038 samples and a 24.85% full-Mini improvement to 3.044e-7 adjusted. We implemented those equations and sample count. It is a method-level reconstruction, not a byte-identical or seed-identical submission replication. The published method is well behind the current leaders.

Other reusable public research includes [Oishi1029's MIT-licensed estimators and negative results](https://github.com/Oishi1029/arc-whestbench-2026), [SkyeNygaard's Kerdock/MUB submission and verification](https://github.com/SkyeNygaard/arc-whitebox), and [James Henry's cumulant-chain research](https://github.com/jamesrahenry/arc-whitebox-replication). The last explicitly withholds its separate sampling estimator during the contest. These are phase-one references, not ready-made phase-two winners.

## Phase-two facts that change the work

The [official phase-two launch announcement](https://discourse.aicrowd.com/t/phase-2-of-the-arc-white-box-estimation-challenge-is-live/18197) specifies 1024-wide, 16-deep networks, budget 2^41 per MLP, 120 seconds total, 0.4 seconds residual time, and 8 GB solution-process memory. Meaningful numerical work must use flopscope. Residual time is a cap, not an alternative compute channel; effective compute equals metered FLOPs. Bundled data remains permitted. The current announced deadlines are 2 October registration/team freeze, 17 October submissions, and 24 October write-ups, all 23:59 UTC. This supersedes older dates on the challenge overview.

Use the 0.1 multiplier floor from the [current phase-two starter-kit evaluation documentation](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/getting-started/stage-3-run-local.md): score each MLP as MSE times max(0.1, FLOPs / budget), then average. The overview page still displayed a conflicting 0.5 floor when inspected. Our report pins its assumptions explicitly and records failures separately; it does not silently mix those scoring descriptions.

The local conda environment actually contained flopscope 0.8.0rc5 and whestbench 0.12.0rc5. An isolated `.venv-phase2` now uses flopscope 0.12.1 / whestbench 0.16.1, matching the [current starter-kit pins](https://github.com/AIcrowd/whest-starterkit/blob/main/pyproject.toml). Existing phase-one files and environment were preserved. Their old calibration coefficients and depth schedules have not been adopted for phase two.

## What was built and measured

`estimator.py` is standalone submission code using only flopscope and the whestbench estimator contract. Its default is the published blend. `benchmark.py` is an offline harness; its NumPy, Arrow, datasets, and threadpool imports are not packaged with the estimator.

Protocol: first eight networks (IDs 0-7) from the first prepared public Mini Arrow shard at `v2-phase2`; official baked 1e9-sample targets; float32 numerical state; four backend threads; six presets fixed before inspecting results. One common, independent estimator RNG seed (20260905) is used across networks and methods. We deliberately do not reconstruct the dataset's seed protocol from an identity column. This means the result cannot reproduce another submission bit-for-bit. All eight are now exposed development data; there is no untouched validation claim.

| Method | Adjusted MSE | Raw final MSE | Budget used | Local cap failures |
|---|---:|---:|---:|---:|
| Gain-covariance baseline | 4.39423e-7 | 4.39423e-6 | 2.351% | 0/8 |
| Published 0.75/0.25 blend, 4038 samples | **3.62064e-7** | 3.62064e-6 | 9.539% | 0/8 |
| Whitened antithetic MC, 8192 samples | 7.65402e-7 | 5.49497e-6 | 13.929% | 0/8 |
| 8192 samples + layer-1 gain transport | 8.64654e-7 | 5.31048e-6 | 16.282% | 0/8 |
| 8192 samples + layer-4 gain transport | 8.20021e-7 | 5.03645e-6 | 16.282% | 0/8 |
| 8192 samples + layer-8 gain transport | 6.98489e-7 | 4.29011e-6 | 16.281% | 0/8 |

The blend improves on covariance by 17.60% on this scout. A second independent sampling seed (20260906), using unchanged code and the same eight networks, gives 2.95625e-7 adjusted, a 32.72% improvement. Both seeds are retained; neither was selected as the submission seed. These are two estimates, not an ensemble. The sizable variation is a reason to evaluate across more networks and seeds, not to claim a private score. The reported full-cohort published gain is within this range, but this is not a statistical reproduction of that full-cohort result.

All 48 main-scout and 16 second-seed evaluations met the local FLOP, total-time, and residual caps. Main-scout maximum predict wall time was 1.64 seconds and residual time 0.042 seconds. These are local in-process timings, not production timing guarantees. A custom harness is not the remote grader, and scored local output is not a conformance certificate.

The sampler-only cost excludes the analytical pass it does not use. `results/scout.json` records the preliminary run that still included that pass; use `scout_final.json` for comparisons. Code hashes in both final results match the current estimator.

Additional bounded diagnostic: the existing `experiments/layerwise_diagonal_k3_estimator.py` runs at 1024x16. With third_scale=1, horizon=2, cutoff=16, output_coefficient=1, it obtained 3.82411e-7 adjusted on scout MLP 0 versus covariance's 4.50169e-7, using 135.49 billion FLOPs. This is one network only, with altered parameters to remove the phase-one calibration, not evidence to promote it. A separate target-assisted scalar diagnostic on that same MLP reduced covariance raw error only to 1.64501e-6 even with the optimal scalar; that is an oracle diagnostic and was not shipped.

Three independent numerical tests passed: exact first-layer Gaussian integral with asymmetric weights, the opposite-column antithetic identity, and the small-budget whitening fallback. The official contract validator and archive validator passed. The archive contains the estimator only. Nothing has been submitted to AIcrowd.

The official subprocess runner also completed one generated 1024x16 network with zero failures and 9.539% utilization (`results/subprocess_smoke.json`). That smoke test used only 100 Monte Carlo target samples to check execution and transport cheaply; its error value is intentionally excluded from all accuracy comparisons. Re-run with `whest run --estimator research/phase2/estimator.py --runner subprocess --n-mlps 1 --n-samples 100 --seed 20260905 --max-threads 4 --format json`.

## Puffi analytical-reference reconstruction

The next target proposed below has now been implemented. [`PUFFI_REFERENCE.md`](PUFFI_REFERENCE.md) explains the equations, the disclosure gaps, our completion choices, and the disjoint phase-two experiment. [`puffi_reference.py`](puffi_reference.py) carries full covariance, marginal third/fourth cumulants, a paired fourth-order surrogate, and eight generations of mixed third-order terms through nine layers.

On eight networks from a second public Mini shard that was not read until after ridge selection, the unfitted reconstruction reduced layer-9 raw MSE from the covariance closure's 2.62532e-6 to 2.90282e-7. At the final layer it achieved 1.73746e-6 raw and 4.35718e-7 adjusted, a small 0.79% adjusted improvement over covariance. Newly fitted phase-two response coefficients did not generalize to a final-layer improvement and are rejected. Carrying the rich state for 10, 11, or 16 layers lowered raw error but lost after compute charging; the 16-layer form also violated the FLOP and residual caps.

The standalone estimator passes `whest validate`. A full-width subprocess smoke test completed with zero failures, 25.078% utilization, 6.19 seconds wall time, and 0.312 seconds residual time. Its 100-sample target is deliberately excluded from accuracy comparisons. Results and the frozen fitting protocol are in `results/puffi_reference_fit.json`, `results/puffi_subprocess_smoke_summary.json`, and `puffi_protocol.json`.

## What to pursue next

The leading scores are roughly two orders of magnitude below this starting point, but that numerical comparison uses different cohorts and cannot estimate our rank. Public evidence does not justify claiming we have deciphered the top-three algorithms.

The next concrete replication target is now Puffi's *inter-layer control variate around the reconstructed reference*. Use a budgeted structured angular sample at layer 9 and the output, fit the within-network ridge response disclosed in equations A31-A37, and transport the layer-9 sample/reference discrepancy. Full 1024-dimensional Kerdock/MUB quadrature has 2*1024*(1024/2+1) = 1,050,624 nodes, so its unpruned dense forward pass cannot be transplanted wholesale into this budget. Frame selection, cheaper propagation, or a smaller residual sample remain necessary design choices.

Before tuning any constants, reserve untouched networks and multiple RNG streams. Optimize the compute-adjusted objective on each network, count all failed runs, and use the official subprocess/client path before considering a submission. Winning requires closing the accuracy gap; another tiny phase-one pruning optimization will not establish that.

## Re-run

From the starter-kit directory in PowerShell:

```powershell
$env:PYTHONIOENCODING='utf-8'
& ./.venv-phase2/Scripts/python.exe -m pytest research/phase2/test_numerics.py -q
& ./.venv-phase2/Scripts/python.exe research/phase2/benchmark.py --arrow ../tmp/phase2_mini_first8.arrow --output research/phase2/results/scout_final.json --limit 8
& ./.venv-phase2/Scripts/python.exe research/phase2/benchmark.py --arrow ../tmp/phase2_mini_first8.arrow --output research/phase2/results/blend_second_seed.json --limit 8 --methods covariance published_blend --seed 20260906
& ./.venv-phase2/Scripts/python.exe -m pytest research/phase2/test_puffi_reference.py -q
& ./.venv-phase2/Scripts/python.exe research/phase2/fit_puffi_reference.py --development ../tmp/phase2_mini_first8.arrow --validation ../tmp/phase2_mini_validation8.arrow --output research/phase2/results/puffi_reference_fit.json
& ./.venv-phase2/Scripts/whest.exe validate --estimator research/phase2/estimator.py --format plain
& ./.venv-phase2/Scripts/whest.exe validate --estimator research/phase2/puffi_reference.py --format plain
& ./.venv-phase2/Scripts/whest.exe package --estimator research/phase2/estimator.py --output research/phase2/public_blend_replication.tar.gz --yes --format plain
& ./.venv-phase2/Scripts/whest.exe validate-package research/phase2/public_blend_replication.tar.gz --format plain
```

The downloaded input is about 538 MB, from [this public Arrow shard](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026/resolve/v2-phase2/prepared/mini/data-00000-of-00014.arrow). The other Mini shards were not downloaded. `requirements.txt` pins the benchmark's direct dependencies. The estimator defaults to the supplied per-MLP RNG seed in production; the scout deliberately supplies its independent seed through the same API.
