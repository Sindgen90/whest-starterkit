# Fourth-order representation study — 7 October 2026

**Recommendation: retain the incumbent.** No candidate met the frozen adjusted-score advancement criterion. The user-reported leaderboard value 8.62e-8 is not used as a local target. Sources, prior results and archives were preserved; the preservation check is recorded separately. No AIcrowd upload.

## Paired full-width development and unchanged official gates

| Method | Execution status | Public raw MSE | Nominal adjusted | GFLOPs | Official max wall s | Official max residual s | Official smoke adjusted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_snapshot | run/pass | 9.13641e-07 | 9.13641e-08 | 196.317 | 6.95803 | 0.200644 | 6.78403e-05 |
| qr | failed official | 9.17408e-07 | 1.85881e-07 | 445.557 | 11.0261 | 0.406246 | 0.89047 |
| block8 | failed official | 1.34738e-06 | 3.21734e-07 | 525.092 | 23.1066 | 1.28458 | 0.89047 |
| block16 | failed official | 1.34666e-06 | 8.11394e-07 | 1324.96 | 49.4455 | 4.82889 | 0.89047 |
| qblock8 | failed official | 9.16656e-07 | 3.95432e-07 | 948.627 | 24.8459 | 1.36679 | 0.89047 |
| qblock16 | failed official | None | None | None | 52.6346 | 5.04808 | 0.89047 |
| qrblock8 | failed official | 9.1737e-07 | 4.66511e-07 | 1118.27 | 26.4129 | 1.46468 | 0.89047 |
| qrblock16 | infeasible FLOPs | None | None | 2279.62 | None | None | None |
| selective_block8 | run/pass | 1.11861e-06 | 1.36023e-07 | 267.402 | 8.11401 | 0.344145 | 8.25387e-05 |
| selective_block8_compat | run/pass | 1.11861e-06 | 1.36023e-07 | 267.402 | 8.13654 | 0.343994 | 8.25387e-05 |

Public means are baked 1e9-draw targets on identical IDs 0,1. Nominal adjusted = raw MSE × max(0.1,F/2^41), and is **not an eligible score for a failing method**. Official smoke scores use 100 samples on different seed-20261071 networks and are execution evidence only; failure uses the official fallback predictions/scoring. Do not compare smoke scores to baked-target scores. The full per-network and layerwise means remain in the raw JSON/NPZ files. Q+R/block16 has no full-width accuracy result. Q/block16 exhausted its budget before final output; no final MSE is claimed.

Current installed versions: whestbench 0.16.1, flopscope 0.12.1. Unchanged spec: F=2^41, wall 120 s, residual 0.4 s, memory 8192 MB, four threads, lambda=0. Setup timeout 5 s and predict-timeout field 30 s are recorded in environment.json; actual official outcomes, rather than assumptions about those fields, govern eligibility. The score floor ends at 219.902325555 GFLOPs, 12.014% above the 196.317442558-GFLOP incumbent.

## Representation, transport and nonlinear errors

The common-oracle study has 372 cells: widths 16/32, two independent weight seeds each, depth 16, source checkpoints 0/7/14, four independent streams of 32768 inputs each. It compares A–G and every requested block-size/partition combination. Four-stream cross-products of linear errors remove k-statistic sampling noise in expectation. Delete-one-stream jackknife intervals use t3 and are approximate with only four streams. One-step nonlinear quantities retain plug-in bias; their intervals must not be read as rigorous confidence bounds on the exact nonlinear closure. Network-to-network variation is separate from input-sampling uncertainty.

| Oracle family | d4 risk | Q risk | R risk | Weight-probe risk | One-step mean risk | One-step C risk |
| --- | --- | --- | --- | --- | --- | --- |
| A | 4218.46 | 975.333 | 3956.53 | 158.075 | 0.00102423 | 0.0131907 |
| B | 4218.46 | 941.559 | 1449.49 | 158.075 | 0.00102423 | 0.136167 |
| C | 277.754 | 128.788 | 132.422 | 30.8376 | 0.000443076 | 0.0184474 |
| D | 1955.72 | 525.73 | 719.809 | 88.3079 | 0.000281436 | 0.0208543 |
| E | 2231.33 | 502.756 | 758.065 | 79.1765 | 0.000637826 | 0.05438 |
| F | 171.045 | 80.5894 | 81.3504 | 17.0202 | 0.000306832 | 0.0133905 |
| G | 0 | 0 | 0 | 0 | 7.21237e-06 | 0.000370364 |

The D/E/F rows average all requested sizes and partition controls; this is a descriptive overview, not a weighted scientific ranking across units or networks. Detailed normalized per-network/layer comparisons and uncertainty are in oracle_results_v2.json. A uses the legacy Gram plane; its general four-linear probes refer to its underlying paired-Q representation, since the legacy Gram approximation alone does not define a complete transported tensor. B retains exactly the same source projection as A, so its marginal fourth contraction is identical. C changes representation. G removes fourth representation/transport loss but still has nonlinear response-expansion error.

The frozen clustering algorithm uses only baseline-propagated first-layer C: smallest available pivot, highest absolute normalized-correlation neighbors, stable index tie breaks, fixed through depth. The predeclared aggregate E/F contraction criterion chose **correlation**, with normalized risks contiguous=0.358502, random=0.377848, correlation=0.283180. No oracle cumulants choose individual clusters; no coefficient is fitted.

Completing paired-Q transport does not recover discarded 3+1, 2+1+1 or four-distinct information. Q+R markedly reduces actual-weight contraction errors, while full cores can reduce them further. These oracle gains do not guarantee recurrence gains. The complete-fourth oracle still has one-step nonlinear error; generic core generation and bivariate response truncation are independent limitations. See AUDIT.md for exact formulas, multiplicities, overlap subtraction, and the specified nonlinear closure.

## Small recurrent ablations

| Method | Completed networks | Final cross-stream mean risk | Improvement % |
| --- | --- | --- | --- |
| baseline_diagnostic | 4 | 0.0473817 | 0 |
| exact_q_control | 4 | 0.0310541 | 34.4597 |
| qr | 4 | 0.0307247 | 35.1549 |
| block8_b4 | 4 | 0.0125106 | 73.596 |
| block8_b8 | 4 | 0.0095952 | 79.7491 |
| block8_b16 | 4 | 0.135878 | -186.773 |
| qblock8_b4 | 4 | 0.0396731 | 16.2692 |
| qblock8_b8 | 4 | 0.0335148 | 29.2664 |
| qblock8_b16 | 4 | 0.023587 | 50.2191 |
| qrblock8_b4 | 4 | 0.0264082 | 44.265 |
| qrblock8_b8 | 4 | 0.0287776 | 39.2644 |
| qrblock8_b16 | 4 | 0.0223104 | 52.9134 |
| block8_b8_repair | 4 | 503442 | -1.06252e+09 |

All primary recurrent runs retain full covariance and the unchanged third-source algorithm. The third tensor can change through changed fourth-informed moments, but its factorization, finite history and reinjection are unchanged. Small-width aggregate gains are highly variable across four networks and are not full-width performance guarantees. The exact-Q control changes paired off-diagonal transport while omitting R responses; it is distinct from the common-oracle physical-Q test B.

The explicitly offline missing-third diagnostic uses oracle fourth anchors and compares complete versus repeated-only source T in generic core generation. Its aggregate contraction risks were 0.0522024 versus 0.0475430, respectively. More third information did not improve this truncated core update on average. This narrow result does not prove older CP history has no value; it prevents attributing the present failures solely to missing third entries. No oracle-filled state enters an estimator.

## Moment consistency and numerical stability

Full per-layer small-network S minima, violation sums, symmetry, marginal slack and covariance eigenvalues are in recurrence_results_v2.json. The separate D,b8 repair makes the quadratic-feature matrices PSD at the repaired state but is catastrophically worse in recurrence: aggregate final mean risk 503441.969 versus 0.00959520 uncorrected. Its covariance became indefinite, reaching minimum eigenvalue −1.52062e8 on one network. Local fourth-moment consistency does not repair an inconsistent lower-order closure. Negative K itself is valid and was never projected PSD.

Initial small failures and legacy kernels are retained in recurrence_results.json/source_history. Some batched einsums triggered incorrect symmetry inference across batch axes and were replaced by equivalent metered matrix products. Later float32 affine covariance asymmetry was handled by explicit symmetrization of WᵀCW, without changing the SDK or disabling validation. The corrected small study completed all 52 cases. Independent tests also verify Gaussian-tree leading order, CP-core contraction and the new Gaussian B31 anchor.

large_covariance_results.json is an explicitly exploratory public-ID-0 diagnostic with fresh 4×8192 reference inputs and checkpoints 0/7/14. It reports covariance cross-stream risks, approximate sampling intervals, eigenvalues and block consistency. Failed budget cases retain only completed checkpoints. It is separate from the paired two-network final-mean development screen.

## Cost, memory and the single selective schedule

Initial block16 all-to-all transport took 19.167 s per call. The equivalent batched-matrix transport took 1.828 s, versus 0.505 s for block8. It increased the metered copy/reshape arithmetic and all such work is counted. Process peak working set in that fresh bounded profiler was 3.403 GB; small storage was not treated as proof of cheap propagation. Original and revised kernel profiles, individual operations/calls, copies, construction, nonlinear updates, third restriction, and optional consistency repair are retained.

Global QR/block16 mandatory affine kernels alone require 2,279,623,227,392 FLOPs over 14 state transitions, exceeding the 2,199,023,255,552 cap before covariance, third state, nonlinear updates, preparation or final contractions. Tagged-Q checks give the same relevant kernel costs. This implementation is infeasible; no dense 1024^4 allocation was made.

Process memory observations in *_gate_memory.json are read-only OS process-tree observations every 0.5 s, including helper/CLI/backend/worker. Each process supplies its lifetime peak; sampled tree sums can miss short-lived children and are not exact tensor-allocation accounting. In-process screen peak working sets are cumulative within a cohort process and are explicitly labeled. Shape bounds and fresh bounded profiles complement these observations. Neither timing type replaces official residual/wall outcomes.

Only one selective schedule was tested: D,b8 enriches source layers 0 and 7, then generates the next global Q from current full-covariance/marginal/bivariate moments. Discarded cross-block source cumulants are not copied back. The schedule and family were frozen in selective_selection.json before its full-width confirmation; D,b8 had the strongest small mean improvement. A vectorized partition implementation preserves the same greedy rule. No schedule or damping sweep was run. Terminal-only marginal contractions avoid unused Q/R/core work; a small independent comparison differed by at most 3.73e-9 in returned means.

The packaged selective_block8_compat source caps block width at n for the official width-four/depth-two contract check. It has identical arithmetic, outputs and metered FLOPs at width1024; compatibility.json and the separate paired full-width/gate results document this necessary contract fix. It is a packaging version of the same schedule, not another selected experiment. The original source and its gate remain preserved.

No primary candidate met the ≥2% paired adjusted improvement criterion on both development networks; fresh held-out weight-seed confirmation and a promising-method random-partition control were therefore not triggered. This is a deliberate frozen stopping rule, not evidence of held-out accuracy.

## Packaging and preservation

| Validated EXPERIMENTAL archive (not recommended) | Archive SHA-256 | Estimator SHA-256 |
| --- | --- | --- |
| C:\Users\nikso\Desktop\Books\AI papers\ARC_Challenge\ARC_Whitebox_Challenge\whest-starterkit\submissions\phase2_EXPERIMENTAL_selective_block8_compat_20261007.tar.gz | 43273f389b437b8629a0ab99282ea9ad712ef5c6d10c858163e3e1bd5561dded | 033c8c06adb55abe21c5137993f36d61b19dda33ca70fc3f223bb8faaa7c6fad |

The incumbent archive remains `submissions/phase2_joint_full_runtime_20261002.tar.gz`, SHA-256 `89035799f6acfe5dd14948fbedefd37ea2bc35472ce2ce72518c4485676782b3`. Incumbent source SHA-256 `bd4c808d47068dcf2b0515472925598f56c43b2b603a2159b982d4d1f9dbce7e`. The 528-file pre-study preservation ledger covers previous research, archives and official SDK/metering sources; preservation_verification.json records the final check.

Source hashes and raw-file hashes are in inventory.json; full statuses are in experiment_matrix.csv. Reproduction commands and seed rules are in REPRODUCE.md and PROTOCOL.md. No new best was promoted and nothing was uploaded.

## Additional isolated controls

Exact-transport controls retain the identical nonlinear update but complete the global paired-Q off-diagonal contraction. These are small-network diagnostics only; no full-width source was selected from them.

| Exact affine model | Completed networks | Final mean risk |
| --- | --- | --- |
| qr | 4 | 0.0400299 |
| qblock8 | 4 | 0.0368667 |
| qrblock8 | 4 | 0.0269829 |

Local bivariate fourth-generation test: common oracle mu,C,T and model-specific fourth input, compared against independent post-ReLU references. This separates generation of marginal/Q/R cumulants from accumulated state error. Risks are nonlinear plug-in diagnostics, with four-stream intervals in local_nonlinear_results.json.

| Oracle input family | Generated marginal K4 risk | Generated Q risk | Generated R risk |
| --- | --- | --- | --- |
| A | 32601.3 | 2094.28 | 2359.33 |
| C | 33914.1 | 2257.06 | 2154.39 |
| D8 | 33051.9 | 2283.22 | 2242.43 |
| E8 | 33463.3 | 2342.64 | 2297.08 |
| F8 | 33883.5 | 2244.73 | 2151.71 |
| G | 33888.6 | 2239.69 | 2134.99 |

Full-width exploratory covariance/stability checks on public ID0 only; partial checkpoints for budget failure.

| Method | Observer status | Last checkpoint | Last C risk | Min C eigenvalue | Min block S eigenvalue |
| --- | --- | --- | --- | --- | --- |
| baseline_snapshot | run | 14 | 2.89568e-07 | -6.27793e-07 | inf |
| qr | run | 14 | 2.90111e-07 | -5.73119e-07 | inf |
| block8 | run | 14 | 6.32567e-07 | -6.14379e-07 | -0.205665 |
| block16 | run | 14 | 6.29816e-07 | -6.21851e-07 | -0.228695 |
| qblock8 | run | 14 | 2.9012e-07 | -5.97037e-07 | -0.290411 |
| qblock16 | failed | 14 | 2.9007e-07 | -5.96554e-07 | -0.317502 |
| qrblock8 | run | 14 | 2.90114e-07 | -5.73102e-07 | -0.290666 |
| selective_block8 | run | 14 | 2.96007e-07 | -5.48084e-07 | -0.149395 |

## Same-seed local/subprocess timing

| Source | Official runner | Failed/2 | GFLOPs | Max wall s | Max residual s |
| --- | --- | --- | --- | --- | --- |
| baseline_snapshot | local | 0 | 196.317 | 7.05874 | 0.204853 |
| baseline_snapshot | subprocess | 0 | 196.317 | 6.95803 | 0.200644 |
| selective_block8 | local | 0 | 267.402 | 8.18261 | 0.353819 |
| selective_block8 | subprocess | 0 | 267.402 | 8.11401 | 0.344145 |
| qr | local | 2 | 445.557 | 10.9637 | 0.410683 |
| qr | subprocess | 2 | 445.557 | 11.0261 | 0.406246 |

These CLI runs use identical source hashes, seed20261071, deterministic contest generation, sample count, thread limit and caps. Only the runner differs. In-process/remote execution paths have different framework/backend work and timing accounting; measured residuals are not interchangeable. This experiment establishes the practical discrepancy, without claiming to isolate every source of framework overhead. Only subprocess outcomes gate packaging/promotion.

