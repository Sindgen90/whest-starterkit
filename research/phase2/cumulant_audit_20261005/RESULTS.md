# Measured results and recommendation

**Retain the existing best estimator and archive.** No new candidate is promoted or packaged; no upload is made.

Four standalone analytical ablations were evaluated without fitting or a parameter grid. All have finite predictions on four identical width-1024, depth-16 exposed public networks (IDs 0-3). Lower MSE/score is better; a positive percent change below denotes improvement. These are baked 1e9-draw targets. The in-process timings in this table are descriptive; they do not establish eligibility.

| Ablation | Raw final MSE | Nominal adjusted | FLOPs (billions) | Change vs baseline | In-process wall / residual (s) |
|---|---:|---:|---:|---:|---:|
| baseline | 9.425604194e-07 | 9.425604194e-08 | 196.317443 | +0.0000% | 8.088 / 0.266 |
| third_memory16 | 9.316557873e-07 | 9.316557873e-08 | 199.249391 | +1.1569% | 8.799 / 0.327 |
| fourth_cross1 | 9.425840780e-07 | 9.425840780e-08 | 204.957756 | -0.0025% | 8.437 / 0.293 |
| fourth_response | 9.409377974e-07 | 9.409377974e-08 | 196.626034 | +0.1722% | 8.688 / 0.296 |
| third_square | 9.462916240e-07 | 9.462916240e-08 | 196.625896 | -0.3959% | 8.373 / 0.270 |

The one-mode cross-pairing candidate is essentially unchanged in accuracy. The pure marginal fourth response has a small 0.17% aggregate improvement and mixed per-network effects. Adding mixed third squares worsens every screened network: formal completeness of one term does not make approximate cumulant inputs or the remaining closure accurate. No variants were combined or retuned. The mixed third-square and marginal-fourth changes update all three covariance planes rather than changing only returned means.

## Compressed-history confirmation

On IDs 8-11, excluded from this candidate selection but exposed in prior work, raw MSE is 9.576359623e-07 for baseline and 9.507456522e-07 for compressed history (0.7195% improvement). Three of four networks improve; one worsens. Both retain the 0.1 multiplier. The original 2% screen threshold was not met; the documented confirmation amendment advances only this cheap candidate to check the small effect. This deviation is explicit and the source stayed frozen.

Two fresh networks use the official He sampler at seeds 20261051 and 20261052. Each receives four independent reference streams of 65,536 draws (524,288 total input draws across both networks). Samples are offline and never enter inference. Cross-stream risk subtracts independent reference sampling variance in expectation. It estimates error conditional on these networks; the t intervals describe reference sampling uncertainty, not uncertainty over all possible weights.

| Fresh seed | Baseline raw cross-stream risk | History raw cross-stream risk | Candidate minus baseline; sampling-only 95% interval |
|---|---:|---:|---|
| 20261051 | 7.595908332e-07 | 7.762899379e-07 | +1.669910e-08 [-8.761041e-09, +4.215925e-08] |
| 20261052 | 8.619913582e-07 | 8.347179941e-07 | -2.727336e-08 [-6.244036e-08, +7.893636e-09] |

Aggregate raw risk is 8.107910957e-07 versus 8.055039660e-07, a 0.6521% point improvement. Estimated adjusted risks are 8.107910957e-08 versus 8.055039660e-08. The paired raw difference is -5.287130e-09, with 95% interval [-1.236382e-08, +1.789560e-09]. It includes zero and one fresh network worsens. The frozen promotion criterion fails. We do not assert that the population objective improved or that this candidate is definitively worse.

## Official execution gates and numerical checks

The unchanged official CLI first checked baseline and compressed history on identical synthetic smoke networks at seed 20261005 (100-draw targets are for execution only). The official CLI’s unchanged SubprocessRunner and scoring function then checked both estimators on the same two fresh networks with their pooled offline targets. Default caps and four threads are unchanged. All eight official predictions passed. The source ledger confirms no evaluator or metering edits.

| Estimator | Official FLOPs | Official residual range (s) | Maximum official prediction wall (s) | Failures |
|---|---:|---:|---:|---:|
| baseline | 196,317,442,558 | 0.281159–0.302611 | 8.721989 | 0/4 |
| third_memory16 | 199,249,391,038 | 0.319062–0.323977 | 8.936399 | 0/4 |

The official adjusted scores against pooled fresh MC targets are 1.137483309e-07 and 1.132196246e-07. They include reference MC noise; use the paired/cross-stream estimates above for the accuracy comparison. Neither is a leaderboard result or a remeasurement of the user-reported 8.62e-8.

33 mathematical/regression tests pass. These cover exact affine third-factor transport, residual reinjection, recentering and Gaussian pairing subtraction, signed fourth-mode transport, response coefficients, zero ablations, input immutability, repeatability, terminal/full-state equivalence, and the preserved estimator’s previous checks. All screened large-network predictions are finite. In the local baseline/history diagnostic, minimum incoming variance exceeds 0.0967 and no marginal mean correction hits its clip bound. Symmetrization/diagonal replacement do not prove PSD or existence of a joint law.

## What was learned

Local oracle K3/K4 readouts remain accurate at the reference sampling resolution, while propagated means, variance and shape information leave a much larger error. Older CP columns can be retained for only 2.932 billion additional FLOPs, avoiding the cost of a whole second dense generation, but this pruning rule does not establish a reliable improvement. The next mathematical target is newly generated distinct third information and mixed fourth 3+1 information, whose first-layer size is comparable to retained paired information. Do not infer from the failed one-mode or mixed-square ablation that all such contributions are irrelevant. Do not add higher cumulant orders or tune coefficients on these references.

## Preserved submission and reproduction

373 pre-existing evidence/evaluator/metering files were rehashed; none changed. The baseline source also has an identical snapshot in baseline_snapshot/. The preserved submission is `C:\Users\nikso\Desktop\Books\AI papers\ARC_Challenge\ARC_Whitebox_Challenge\whest-starterkit\submissions\phase2_joint_full_runtime_20261002.tar.gz`, SHA-256 `89035799f6acfe5dd14948fbedefd37ea2bc35472ce2ce72518c4485676782b3`. The estimator bytes inside the tar match the frozen baseline. Existing official archive validation remains intact; no new tar is created for rejected candidates.

From whest-starterkit, use .venv-phase2/Scripts/python.exe and the scripts `benchmark_cumulant_audit.py`, `diagnose_cumulant_audit.py`, `validate_cumulant_fresh.py`, and `benchmark_joint_official.py`. All experiment outputs are append-only: choose new output names on reruns. Source hashes, fixed network indices/seeds, parameters, exact commands for official checks, raw reports, MC moments/predictions and the preservation ledger accompany this report. See CUMULANT_AUDIT.md for equations and adaptations of Hanin’s paper.
