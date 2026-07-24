# Hybrid checkpoint-correction first report

Run id: `01c1965f-581e-4232-8540-b3fc379e5a58`

Partition: `development` only. Holdout was not evaluated.

## Implemented

- inspected/reused the full-covariance K2 propagator from `examples/03_covariance_propagation.py`;
- inspected/reused the shifted lattice RQMC generator and batched particle propagation pattern from `submission_variants/rqmc_baseline/estimator.py`;
- implemented checkpoint logging at layers 4, 8, 12, 16, 20, 24, 28, 32;
- implemented raw mean replacement, scale/offset correction, and fixed rank-4 Hadamard directional correction;
- counted analytic propagation, particle forward, and correction FLOPs with separate flopscope budget contexts.

## Development results

| Method | Mean final MSE | Mean all-layer MSE | Mean adjusted score | Mean FLOPs | Direct-32 info gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| k2_only | 7.77514e-05 | 5.43373e-05 | 7.77514e-06 | 1.61664e+09 | 12.81 |
| direct_rqmc_32 | 0.000996169 | 0.00284625 | 9.96169e-05 | 1.35882e+08 | nan |
| hybrid_raw_replace | 0.000996169 | 0.00192745 | 9.96169e-05 | 1.75254e+09 | 1 |
| hybrid_scale_offset | 0.000525557 | 0.000294342 | 5.25557e-05 | 1.75256e+09 | 1.895 |
| hybrid_rank4_projected | 0.000534434 | 0.000306072 | 5.34434e-05 | 1.75261e+09 | 1.864 |

Summary CSV: `research\results\hybrid_checkpoint_summary.csv`

The direct-32 information-gain column is a same-particle diagnostic, not an equal-FLOP comparison. The hybrid methods spend the 32-particle budget plus full K2 propagation. Because all methods here remain below the competition score multiplier floor of `0.1`, the adjusted-score comparison is still driven by raw final MSE, but no hybrid improvement is claimed against equal-FLOP direct RQMC.

## Context from existing development baselines

From the existing development run in `research/results/stein_rqmc_holdout_summary.csv`:

| Baseline | Mean final MSE | Mean adjusted score | Mean FLOPs |
| --- | ---: | ---: | ---: |
| ordinary RQMC, 1293 particles | 2.43848e-05 | 2.43848e-06 | 5.46826e+09 |
| active-Hermite RQMC r=4 | 3.70653e-05 | 3.70653e-06 | 5.88647e+09 |

Both are much better than the checkpoint-corrected 32-particle hybrids in this first attempt.

## Checkpoint diagnostics

- Innovation stability across three independent 32-particle randomizations on the diagnostic subset: mean norm CV 0.153, mean pairwise cosine -0.014.
- Scale/offset energy ratio, scale/offset method: 0.05516.
- Rank-4 directional energy ratio after scale/offset: 0.01487.
- Combined unshrunk scale/offset+rank4 energy ratio: 0.07017.

## Answers to the first-report questions

1. Are checkpoint RQMC innovations stable across randomizations? Partly, but not enough for an aggressive correction; the diagnostic above shows material randomization noise.
2. Are they mostly scale, offset, or directional? The scale/offset component captures only a small slice of innovation energy; most innovation is directional/noisy.
3. Does rank-4 projection capture most of the innovation energy? No. The fixed rank-4 directions capture only a small additional part.
4. Does checkpoint correction outperform direct averaging of the same particles? See the info-gain column. Values above one favor the hybrid over direct 32-particle averaging.
5. Does the correction remain useful after four or more downstream layers? The final-layer result is the relevant rollout test; if final info gain is below or near one, downstream propagation did not preserve a useful correction.
6. Is block size four reasonable? It is a reasonable starting point mechanically, but this first fixed-basis correction is too weak/noisy to justify learned closure yet.

## Recommendation

Do not train a learned blockwise closure yet. The deterministic checkpoint method should first beat direct 32-particle RQMC consistently; this run is the gate for that decision.
