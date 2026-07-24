# Active-basis Hermite/RQMC prototype report

Status: development-only prototype. Holdout was not evaluated.

## Implemented method

Path: `submission_variants/rqmc_hermite_active/estimator.py`

Configuration:

- active rank: `4`
- Hermite degree: `2`
- Hermite feature count: `14`
- pilot samples: `8`
- output probes per pilot sample: `2`
- active gradient columns: `16`
- orthogonal iterations: `3`
- cross-fit folds: two independent RQMC randomizations, `646` samples each
- coefficient fitting: Hermite projection with shrinkage `0.5`

The implementation estimates a network-specific active basis from manual ReLU reverse-mode probes:

```text
g = J_f(x)^T r
```

It stores pilot gradients as columns of `G`, starts from the same fixed Hadamard basis used by random-basis Hermite, and applies block power iteration:

```text
Q <- orth(G @ (G.T @ Q)).
```

The final estimate uses the same cross-fitted zero-mean Hermite control-variate estimator as random-basis Hermite.

## Correctness and validation commands

```powershell
& 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' -m pytest tests\test_submission_imports.py tests\test_stein_rqmc_methods.py -q
```

Result:

```text
10 passed in 1.26s
```

Additional tests added in this checkpoint:

- submission-facing Hadamard bases are orthonormal for width `4` and width `256`;
- active power basis is deterministic and orthonormal.

Official checks:

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' validate --estimator submission_variants\rqmc_hermite_active\estimator.py --format plain
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' run --estimator submission_variants\rqmc_hermite_active\estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner subprocess --n-mlps 5 --max-threads 1 --format plain
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' package --estimator submission_variants\rqmc_hermite_active\estimator.py --output submissions\rqmc_hermite_active_r4_p8_probe2_646x2.tar.gz --yes --format plain
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\Scripts\whest.exe' validate-package submissions\rqmc_hermite_active_r4_p8_probe2_646x2.tar.gz --format plain
```

Results:

- `whest validate`: passed;
- subprocess run: 0 failed MLPs on 5 mini MLPs;
- package written: `submissions/rqmc_hermite_active_r4_p8_probe2_646x2.tar.gz`;
- `whest validate-package`: passed.

## Development comparison

Partition: frozen development MLP IDs `0-59`.

Command:

```powershell
$env:PYTHONIOENCODING='utf-8'; & 'C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe' research\scripts\evaluate_submission_variants.py --partition development --methods rqmc_baseline rqmc_hermite_random rqmc_hermite_active --hermite-rank 4 --hermite-shrinkage 0.5 --profile-active-components
```

Summary file: `research/results/stein_rqmc_holdout_summary.csv`

| Method | n | Mean final MSE | Median final MSE | Mean adjusted score | Mean FLOPs / MLP | Budget fraction | Win frac vs RQMC | Raw gain vs RQMC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary RQMC | 60 | 2.43848e-05 | 1.50229e-05 | 2.43848e-06 | 5.46826e+09 | 0.02010 | n/a | n/a |
| Random-basis Hermite r=4 | 60 | 3.73045e-05 | 2.21185e-05 | 3.73045e-06 | 5.78529e+09 | 0.02127 | 0.267 | 0.6537 |
| Active-basis Hermite r=4 | 60 | 3.70653e-05 | 2.17625e-05 | 3.70653e-06 | 5.88647e+09 | 0.02164 | 0.267 | 0.6579 |

Paired active versus same-feature random-basis Hermite:

- mean active-minus-random final MSE difference: `-2.39132e-07`;
- median active-minus-random final MSE difference: `-1.65263e-07`;
- active win fraction versus random: `0.567`;
- active raw MSE gain versus random: `1.00645`.

Interpretation: active basis is very slightly better than random basis at the same feature count, but the gain is tiny and nowhere near enough to rescue the method. Both Hermite variants are much worse than ordinary RQMC on development.

## Active component FLOP profile

Mean per MLP, from the same development run:

| Component | Mean FLOPs / MLP |
| --- | ---: |
| pilot sampling | 2.34520e+05 |
| pilot forward | 3.3619968e+07 |
| pilot reverse / `J^T r` | 6.7108864e+07 |
| basis estimation | 2.22010e+05 |
| evaluation fold sampling | 3.4404132e+07 |
| Hermite feature construction | 2.664104e+06 |
| evaluation forward passes | 5.419040768e+09 |
| regression / cross-fit residuals | 3.29176320e+08 |
| profiled total | 5.886470686e+09 |

The active overhead beyond same-rank random-basis Hermite is roughly `1.01e+08` FLOPs / MLP, mostly pilot forward/reverse and basis work. Both methods remain below the `0.1` score multiplier floor, so the adjusted score is still raw final MSE multiplied by `0.1`.

## Recommendation

Do not submit active-basis Hermite in this form.

The prototype satisfies the implementation and validation requirements, and it slightly beats the same-feature random basis. But it fails the success criteria because it does not beat ordinary RQMC after counting pilot and reverse-pass costs, and it retains severe outliers.

Next recommended action: do not evaluate active on holdout. Return to the stronger path: ordinary RQMC/radial RQMC and low-risk combinations. If we want to obey the "two reasonable implementations before stopping" rule strictly, the only active follow-up I would allow is one small second attempt with the same rank/pilot design but weaker shrinkage `0.25`; if that also misses, stop active-basis development for Phase 1.
