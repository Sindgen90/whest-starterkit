# Three ordered approaches: comparison and recommendation

**Retain the baseline and its existing successful archive.** All three approaches were implemented separately, tested in the requested order, and assessed with the unchanged official runner. No new estimator has convincing adjusted-score evidence. Two passing prototypes have separately validated **EXPERIMENTAL — NOT RECOMMENDED** archives as requested; the conditional prototype and control fail the execution constraint and have no submission-ready archive. No upload was made.

The prior mathematical audit and local-versus-propagation diagnostics are preserved in cumulant_audit_20261005/. These experiments pursue three different mechanisms without combining them, adding higher cumulant orders, tuning coefficients against public targets, or constructing a dense tensor at challenge width. The source baseline is unchanged at SHA-256 bd4c808d47068dcf2b0515472925598f56c43b2b603a2159b982d4d1f9dbce7e.

These accuracy measurements are paired on exposed public IDs 0,1, with baked 1e9-draw targets. Baseline values are identical across all three runs. Stochastic methods average all declared independent estimator randomizations, rather than selecting a seed. Final raw MSE is the objective because Phase 2 lambda=0. The unchanged multiplier is max(0.1,FLOPs/2^41), with its floor ending at 219,902,325,555.2 FLOPs. The **user-reported leaderboard score 8.62e-8 is separate** from the local baseline mean below. Numerical reference error is not zero, but the extreme surrogate/conditional losses and the failed mixed allocation screen do not justify generating fresh targets. No fresh held-out network was inspected or candidate promoted.

| Method | Raw final MSE | Nominal adjusted | FLOPs, billions | Official failures | Official wall range, s | Official residual range, s |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 9.136408012e-7 | 9.136408012e-8 | 196.317443 | 0/2 | 6.91351–7.14365 | 0.20485–0.21094 |
| 1. Analytic surrogate + residual | 2.090582088e-5 | 2.090582088e-6 | 160.173107 | 0/2 | 0.70258–0.72932 | 0.05293–0.05783 |
| 2. Output-sensitive old sources | 9.090599697e-7 | 9.090599697e-8 | 200.765546 | 0/2 | 7.34674–7.45773 | 0.25273–0.25685 |
| 3. Important direction, 3 nodes | 8.996202224e-7 | **2.417529231e-7, ineligible** | 590.938584 | 2/2 residual | 21.14221–21.33104 | 0.65378–0.66305 |
| Random-direction control | 9.110633153e-7 | **2.442697769e-7, ineligible** | 589.591207 | 2/2 residual | 21.19964–21.43260 | 0.65494–0.65693 |

Nominal adjusted accuracy is reported to show the FLOP tradeoff and is **not a valid official score for failed candidates**. Official failures use zero predictions and multiplier 1; conditional and random-control adjusted smoke scores are both 0.7905057073. All gate runs use identical CLI smoke networks at seed 20261081, four threads, default caps and 100-draw execution-only targets. Their noisy target scores are not substituted for the baked-target accuracy comparison. Gate manifests preserve both raw output and failure flags; success is determined by n_failed_mlps rather than CLI exit status. No in-process timing confers eligibility.

Approach 1 fails its independent residual-variance pilot: variance ratios to plain sampling are 1.058 and 1.054. Actual average error is about 22.88 times baseline; the cheap fixed input dictionary does not capture enough deep-network variability. All fitting and 4608 full-network evaluations are charged. RQMC was not triggered. Across six network/randomization cases, raw MSE ranges 1.71978e-5–2.72955e-5. Exact Gaussian feature centers and independent correction samples avoid control-variate centering bias.

Approach 2 passes a deliberately modest small-network ranking diagnostic: output-sensitive benefit correlations are 0.10/0.338 versus norm ranking -0.153/0.097. Offline error-weighted sensitivity predicts effect magnitudes better, emphasizing that local error size and sensitivity are different quantities. Full-network adjusted gain is only 0.5014%, with one network worse. The frozen norm control achieves 8.999607726e-7 raw and 8.999607726e-8 nominal adjusted at 199.249391 billion FLOPs, beating this more expensive prototype. That control's prior fresh validation is already mixed; these two exposed development networks do not overturn it.

Approach 3 produces a 1.5346% raw gain, versus 0.2821% for the random control. Important-direction adjusted error is 2.6460 times baseline, whereas the actual cost requires 2.6873 times lower raw MSE to compensate. Residual time also violates the 0.4-second cap by more than 0.25 seconds. No batching or additional directions/nodes were tried because the adjusted-accuracy screen is decisively unfavorable. Conditional covariance retains the perpendicular uncertainty and explicit deterministic branches handle its singularity.

Read SURROGATE_REPORT.md, ALLOCATION_REPORT.md and CONDITIONAL_REPORT.md for equations, the papers' established claims versus adaptations, per-network variation, sampling-only diagnostic intervals, numerical checks and limitations. PROTOCOL.md was frozen before new target inspection; REPRODUCE.md gives exact commands, seeds, versions and raw result locations. Standalone sources are in surrogate/, allocation/, conditional/ and random_control/. Oracle repairs and reference tensors occur only in the separate small diagnostic source and NPZs. They never enter a tar.

21 new-method and baseline regression tests pass. The final ledger rechecks all 438 pre-existing files, including prior evidence, archives, evaluator and metering sources; no file changed. baseline_snapshot is byte-identical. The existing archive was independently checked to contain that baseline and passes current validate-package. New archives pass official validate, package and validate-package and contain estimator.py, LICENSE, PROVENANCE.md and manifest.json. Complete source/evidence SHA inventories accompany final_manifest.json and comparison.json. Development API/test corrections are retained in DEVELOPMENT_NOTES.md; no measured-target parameter change occurred.

New validated archives, both experimental and not recommended:

1. `C:\Users\nikso\Desktop\Books\AI papers\ARC_Challenge\ARC_Whitebox_Challenge\whest-starterkit\submissions\phase2_EXPERIMENTAL_surrogate_20261005.tar.gz`
   SHA-256: **69f0c7bb80532cdd1a1383637e793b31d83a48143537a9161f861f83c60be56b**. Source: a965bfd64e6885f30ad23a316d03ebe7016d00968de3952aca0bceff77150b57.
2. `C:\Users\nikso\Desktop\Books\AI papers\ARC_Challenge\ARC_Whitebox_Challenge\whest-starterkit\submissions\phase2_EXPERIMENTAL_allocation_20261005.tar.gz`
   SHA-256: **ad1606520fc1edb964fa0bfa3e81594309c6b5460fb80f70f36be8e1a7bbb599**. Source: a0ccebe7e9716c1822c8364a4cd6255d7bccc5a86081ee5fc97f7a0a3f8a01a1.

Recommended preserved archive:
`C:\Users\nikso\Desktop\Books\AI papers\ARC_Challenge\ARC_Whitebox_Challenge\whest-starterkit\submissions\phase2_joint_full_runtime_20261002.tar.gz`
SHA-256: **89035799f6acfe5dd14948fbedefd37ea2bc35472ce2ce72518c4485676782b3**. Neither the reported leaderboard result nor this archive is replaced.
