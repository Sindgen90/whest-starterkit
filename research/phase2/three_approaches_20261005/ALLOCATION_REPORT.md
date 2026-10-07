# Approach 2: output-sensitive retention of older third sources

**Experimental, not recommended.** Small-network repair rankings justify testing this fixed prototype, but the full-network effect is weak and mixed. It passes official execution and packaging. No fresh-reference or promotion trigger is met.

The retained factorization is

\[T_{ijk}={1\over3}\sum_r(A_{ir}B_{jr}B_{kr}+B_{ir}A_{jr}B_{kr}+B_{ir}B_{jr}A_{kr}).\]

The baseline drops an older source after affine transport and gain scaling, then injects a new identity-factor source representing only repeated slices P_ij=T_iij and d_i=T_iii. We keep 16 columns across older generations and subtract their repeated slices before injecting the residual new source. Thus retention changes distinct-index information while preserving the newly computed P and d exactly in algebra. It does not repair the unrepresented newly generated distinct tensor.

Let D_r equal one older CP tensor with **all entries having repeated indices removed**. A representation-loss norm bound is E_r=||A_r|| ||B_r||^2; this bounds ||D_r||, but is not an estimate of the error relative to the unknown physical tensor. For a probe q, define a=q^TA_r, b=q^TB_r, b2=(q^2)^T(B_r^2), ab=(q^2)^T(A_r B_r), ab2=(q^3)^T(A_r B_r^2). Its distinct contraction is exactly

\[D_r[q,q,q]=a(b^2-b2)-2(ab)b+2(ab2).\]

Eight final-coordinate probes are propagated backward through supplied weights and diagonal-Gaussian approximate ReLU gains. The terminal response curvature is F'''_j/6. Rank columns by

\[s_r=\left(\sum_{j\in probes}[(F'''_j/6)D_r[q_j,q_j,q_j]]^2\right)^{1/2}.\]

This separates known loss size E_r from an approximate unit sensitivity s_r/E_r. Probe scores approximate direct terminal third-response changes; they omit indirect changes through later covariance, response nonlinearities and future source generation. They are not derivatives of the complete closure. The future diagonal Gaussian state is only a weight-based sensitivity heuristic; full covariance remains in the actual propagation. All forward gain preparation, backward probes, rankings, retained-factor operations and state updates are metered. No full state Jacobian is constructed.

[Becker and Rannacher's paper](https://doi.org/10.1017/S0962492901000010) motivates weighting local error contributions by output sensitivity. Proposition 2.2, equations (2.17)–(2.19), includes primal and adjoint residuals with a cubic remainder; equation (2.22) gives a simplified weighted-residual representation. The publisher supplied its abstract; direct full-PDF retrieval timed out, so these sections were checked in the [indexed primary paper text](https://citeseerx.ist.psu.edu/document?doi=db4afae538ad519fc2d31a9f5fb56e7a76b2496b&repid=rep1&type=pdf). [Becker's author lecture](https://www.imo.universite-paris-saclay.fr/~francois.dubois/organisation/cnam-20-decembre-2006/becker.pdf), slides 15–21, also gives Galerkin dual-weighted residual identities and distinguishes residual size from dual weights. Our CP selection is an adaptation: moment closure has no proved Galerkin orthogonality, known residual to the true hierarchy, or computed exact adjoint. Its ranking is a heuristic, without the paper's error guarantees.

The frozen small diagnostic uses width 16, depth 8, seeds 20261061/62 and post-layer index 2. For each discarded CP column separately, form D_r and estimate c_r=<T_true,D_r>/||D_r||^2 from four independent streams of 32768 inputs. Replace the missing component by c_r D_r, keeping all existing repeated slices and marginal cumulants unchanged. Assessment uses four additional independent streams of 32768 inputs. These are **offline partial oracle repairs**, not inference data or trained coefficients. A dense 16^3 tensor exists only in this small diagnostic; no dense third tensor appears at width 1024. The repair affects the next propagation and subsequent state; later source dropping follows the baseline.

| Weight seed | Norm / benefit Spearman | Sensitivity / benefit Spearman | Offline error-weighted / benefit Spearman | Offline error-weighted / effect magnitude |
|---|---:|---:|---:|---:|
| 20261061 | -0.15294 | 0.10000 | 0.19118 | 0.65588 |
| 20261062 | 0.09706 | 0.33824 | 0.37941 | 0.81176 |

The sensitivity-versus-norm improvement exceeds the frozen 0.2 correlation threshold on both networks; absolute benefit correlations remain modest. Effects can have either sign. Eight of 16 individual repairs improve the first network and 11 of 16 the second. Sampling-only 95% paired intervals exclude zero for 15 and 14 effects respectively; individual estimates/intervals are saved. Baseline raw cross-stream risks are 0.00427452 and 0.00192562 at this small width, a much harder regime for this closure than width 1024. These rank diagnostics cannot establish a transferable accuracy gain or which full-network error sign is correct.

The separately frozen standalone prototype retains 16 columns ranked by s_r. It uses no oracle coefficient, local repair target, fitted output calibration or reference file. The tensor-norm control is the unchanged prior third_memory16 source, not retuned.

| Method on public IDs 0,1 | ID 0 raw MSE | ID 1 raw MSE | Mean raw MSE | Nominal adjusted | FLOPs |
|---|---:|---:|---:|---:|---:|
| Baseline | 8.910184250e-7 | 9.362631775e-7 | 9.136408012e-7 | 9.136408012e-8 | 196,317,442,558 |
| Norm control | 8.701441780e-7 | 9.297773671e-7 | 8.999607726e-7 | 8.999607726e-8 | 199,249,391,038 |
| Output-sensitive | 8.780561847e-7 | 9.400637546e-7 | 9.090599697e-7 | 9.090599697e-8 | 200,765,545,718 |

Output-sensitive mean improvement is **0.50138%** versus baseline, with one network worsening. It is about 1.011% worse than the cheaper norm control. This fails the frozen >=2% and both-networks screen. No fresh targets were generated. The prior norm-control fresh evidence is already mixed and remains unchanged; its two development improvements are not grounds for promotion.

In-process mean wall/residual seconds are 7.39711/0.25259 for the prototype, 7.16274/0.23644 for norm control, 6.95726/0.20282 for baseline. Official unchanged subprocess passes **2/2** prototype predictions: walls 7.45773/7.34674, residuals 0.25273/0.25685 seconds, same exact metered FLOPs. Default caps and four threads are unchanged. Official validate, package and validate-package pass; allocation_archive.json records tar/source hashes and membership. Execution-smoke target scores are not accuracy evidence.

All large-network and all 32 repaired small-network predictions are finite. Tests independently verify the distinct contraction and that repeated tensor entries of a repair are zero, plus the complete output contract and input immutability. Limited output probes, a crude future-state model, a proxy for representation loss instead of physical truncation error, and missing generated third/fourth information limit this adaptation. A full adjoint or broader sweep was not attempted. Preserve this source as an experimental comparison; retain the successful baseline.

Evidence: allocation_diagnostic.json and two diagnostic NPZs, allocation_screen.json and per-network prediction NPZs, allocation_gate*.json, allocation_validate*.json, allocation_packaging_manifest.json, allocation_archive.json. Sources: allocation/estimator.py and offline repair_diagnostic/estimator.py. Offline sample RNG SeedSequence is [20261005, weight_seed, stream, purpose], purpose 1 for projection and 2 for assessment. Exact commands and hashes accompany REPRODUCE.md and final_manifest.json.
