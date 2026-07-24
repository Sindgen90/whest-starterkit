# Hybrid checkpoint-correction first report

Run id: `09b476a2-f87d-4780-869f-d729c8ed5e9c`

Partition: `development` only. Holdout was not evaluated.

## Implemented

- inspected/reused the full-covariance K2 propagator from `examples/03_covariance_propagation.py`;
- inspected/reused the shifted lattice RQMC generator and batched particle propagation pattern from `submission_variants/rqmc_baseline/estimator.py`;
- implemented checkpoint logging at layers 4, 8, 12, 16, 20, 24, 28, 32;
- implemented raw mean replacement, scale/offset correction, fixed rank-4 Hadamard directional correction, and downstream-sensitive rank-4 correction;
- counted analytic propagation, particle forward, basis estimation, and correction FLOPs with separate flopscope budget contexts.

## Development results

| Method | Mean final MSE | Mean all-layer MSE | Mean adjusted score | Mean FLOPs | Direct-32 info gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| k2_only | 7.77514e-05 | 5.43373e-05 | 7.77514e-06 | 1.61664e+09 | 12.81 |
| direct_rqmc_32 | 0.000996169 | 0.00284625 | 9.96169e-05 | 1.35882e+08 | nan |
| hybrid_raw_replace | 0.000996169 | 0.00192745 | 9.96169e-05 | 1.75254e+09 | 1 |
| hybrid_scale_offset | 0.000525557 | 0.000294342 | 5.25557e-05 | 1.75256e+09 | 1.895 |
| hybrid_rank4_projected | 0.000534434 | 0.000306072 | 5.34434e-05 | 1.75261e+09 | 1.864 |
| hybrid_downstream_rank4_projected | 0.000561087 | 0.000342443 | 5.61087e-05 | 1.84072e+09 | 1.775 |

Summary CSV: `research\results\hybrid_checkpoint_summary.csv`

## Checkpoint diagnostics

- Innovation stability across three independent 32-particle randomizations on the diagnostic subset: mean norm CV 0.153, mean pairwise cosine -0.014.
- Scale/offset energy ratio, scale/offset method: 0.05516.
- Rank-4 directional energy ratio after scale/offset: 0.01487.
- Combined unshrunk scale/offset+rank4 energy ratio: 0.07017.

## Downstream-sensitive basis follow-up

The downstream-sensitive rank-4 basis used block power iteration on the next four linear layers. It had mean final MSE `5.61087e-04`, versus `5.34434e-04` for the fixed rank-4 Hadamard basis. It also increased mean basis-estimation FLOPs from `1.024e+03` to `8.811e+07` per MLP.

Paired against the fixed rank-4 basis:

- downstream-minus-fixed mean final MSE difference: `2.66530e-05`;
- downstream-minus-fixed median final MSE difference: `1.15788e-05`;
- downstream win fraction versus fixed: `0.233`;
- downstream gain versus fixed: `0.9525`.

## Answers to the first-report questions

1. Are checkpoint RQMC innovations stable across randomizations? Partly, but not enough for an aggressive correction; the diagnostic above shows material randomization noise.
2. Are they mostly scale, offset, or directional? The scale/offset component captures only a small slice of innovation energy; most innovation is directional/noisy.
3. Does rank-4 projection capture most of the innovation energy? No. The fixed rank-4 directions capture only a small additional part.
4. Does checkpoint correction outperform direct averaging of the same particles? See the info-gain column. Values above one favor the hybrid over direct 32-particle averaging.
5. Does the correction remain useful after four or more downstream layers? The final-layer result is the relevant rollout test; if final info gain is below or near one, downstream propagation did not preserve a useful correction.
6. Is block size four reasonable? It is a reasonable starting point mechanically, but both the fixed-basis and downstream-sensitive corrections are too weak/noisy to justify learned closure yet.

## Recommendation

Do not train a learned blockwise closure. The deterministic checkpoint method uses the same 32 particles more effectively than direct 32-particle averaging, but it fails against K2 and stronger RQMC baselines, and the downstream-sensitive basis made the rank-4 correction worse.
