# Joint covariance and higher-moment propagation

Research date: 1 October 2026. This is a local exploratory comparison, not a
leaderboard result. Lower mean squared error (MSE) is better. The full joint
two-node method improves accuracy but FAILED the official subprocess overhead
check. In-process timing results alone are insufficient for promotion.

## What the submitted baseline does

The input is standard Gaussian. At each layer we represent the incoming
activation vector by its mean mu and full covariance C. For z = W^T x,
the linear transformation is m = W^T mu and V = W^T C W. These two identities
are exact for the moments supplied, even when x is not Gaussian.

The approximation happens at ReLU. Treat z as jointly Gaussian with those
moments. With sigma_i = sqrt(V_ii), a_i = m_i/sigma_i, its ReLU mean is
sigma_i phi(a_i) + m_i Phi(a_i). Bivariate Gaussian integrals supply the full
post-ReLU covariance, using numerical quadrature. Reuse that mean and
covariance in the next layer. Full covariance matters because each downstream
variance contains cross terms W_ia W_ja C_ij, not just diagonal variances.

The existing venture-derived estimator also applies a frozen five-feature
calibration to reported means. That adjustment is OUTPUT ONLY. It is never
fed into the next layer, and no coefficients are trained during prediction.

There is no backward pass or repeated optimization of the same network.
The recurrence is a forward chain:

    estimated state at layer l -> linear mixing -> ReLU -> state at layer l+1

Errors can accumulate through this chain: covariance affects the next
variance, variance affects ReLU's mean, and the mean affects later layers.
This is why correcting a single reported mean can help a score without
repairing the internal distribution, and why inconsistent mean/variance
corrections can make later predictions worse.

## What the new method changes

We retain mean and full covariance, plus selected joint cumulants:

- Marginal third cumulant k3_i = E[(x_i-mu_i)^3] measures asymmetry.
- Marginal fourth cumulant k4_i = E[(x_i-mu_i)^4] - 3 C_ii^2 measures departure
  from Gaussian fourth moments. Standardized versions are skewness and
  excess kurtosis; the implementation propagates the unstandardized values.
- P_ij = cum(x_i,x_i,x_j) retains cross-neuron third moments.
- Q_ij = cum(x_i,x_i,x_j,x_j) retains paired fourth cumulants.

For a Gaussian reference Z and a function f, the truncated cumulant correction
is E[f(z)] approximately E[f(Z)] + k3 E[f'''(Z)]/6 + k4 E[f''''(Z)]/24.
For ReLU, the derivatives are interpreted through Gaussian smoothing. In
particular the mean correction is

    - k3*a*phi(a)/(6*sigma^2)
    + k4*(a^2-1)*phi(a)/(24*sigma^3).

We apply corresponding corrections to the first four raw ReLU moments, then
recenter them to obtain mean, variance, third and fourth cumulants. Corrections
are bounded by 0.25 sigma^k for raw moment order k; this is a numerical guard,
not a guarantee of a valid joint distribution.

The covariance update also changes. Gaussian anchors are calculated for
Cov(h_i,h_j), Cov(h_i^2,h_j), and Cov(h_i^2,h_j^2). Mixed third/fourth response
terms correct these three matrices. Together they update C, P and Q from the
same approximate joint state. This is substantially different from adding
skewness/kurtosis corrections to marginal means alone.

Full tensors would be prohibitive at width 1024: roughly 4 GiB for third order
and 4 TiB for fourth order in float32. Instead the third tensor is represented
as sums of symmetrized factors A,B,B, retaining a finite history of sources.
The one-history version drops older independent sources but generates the
next source from the corrected current state. Fourth transport projects the
paired plane with squared weights and an adjusted diagonal contraction.
The off-diagonal fourth plane remains an approximation; keeping the projected
Gram matrix does not make the full fourth-tensor transport exact.

These response terms are inspired by the [Puffi A716 write-up, appendix A.7](https://discourse.aicrowd.com/uploads/short-url/iDdUNo66HzI7ovRqhQqpLOFFugI.pdf).
This implementation is our closure, not a reproduction of its published
submission. It combines those ideas with Gaussian pair integrals from the
[venture_agent_ai baseline](https://github.com/agentaiventure-dot/arc-whest-estimator).

## Experiment design and findings

Two public networks (0 and 1) are used for development. Optional fixed
five-feature output calibrations are fitted there with ridge 0.001 and frozen
before reading evaluation targets. Four networks (8-11) are used for the
initial evaluation. They were exposed in earlier work: these are not pristine
holdouts. A later frozen paired check uses networks 4-7 and 12-15, excluded
from this scout but also previously exposed in earlier research.

No oracle moments, simulations, public targets or inference-time fitting are
used inside the estimator. Monte Carlo moments are used only in separate
diagnostics to check the quality of its estimated state.

On development networks 0 and 1, final incoming relative variance RMSE fell
from about 5.7% for the Gaussian state to about 1.0% for the full joint state.
This supports improvement in propagation itself, although it is only a
two-network diagnostic against sampled moments.

Initial evaluation, networks 8-11:

| Method | Raw final MSE | Nominal adjusted MSE | Timing result |
|---|---:|---:|---|
| Current calibrated Gaussian baseline | 1.416969e-6 | 1.416969e-7 | Previously passed |
| Full joint, one-history, four nodes, no calibration | 9.575858e-7 | 9.575858e-8 | 4/4 overhead failures |
| Joint first 8 layers, calibrated | 1.556464e-6 | 1.556464e-7 | Passed scout |
| Joint first 12 layers, calibrated | 1.362669e-6 | 1.362669e-7 | Passed scout |

The full four-node method uses 206,999,915,514 FLOPs, below the 10% score floor
of the 2^41 budget, but its average measured residual overhead is 0.43644 s,
above the 0.4 s cap. Its nominal score is therefore NOT an eligible score.
The eight-layer prefix loses the gain. Twelve layers retain only a modest
improvement, which needs the additional frozen comparison.

Increasing third-history length to two lowered raw MSE further (8.352144e-7)
but raised FLOPs enough to worsen the nominal adjusted objective, even before
accounting for overhead failures. Four versus eight quadrature nodes changed
MSE negligibly. A final two-node check tests whether reducing quadrature work
can preserve the richer state within the runtime cap.

## Reproduction and evidence

- `joint_cumulant.py`: research estimator, with explicit variants.
- `benchmark_joint_cumulant.py`: development/evaluation scout, saves all runs.
- `results/joint_cumulant_20261001.json`: initial one/two-history results.
- `results/joint_cumulant_batched_20261001.json`: vectorized quadrature comparison.
- `results/joint_cumulant_optimized_20261001.json`: optimized full/8/12-layer scout.
- `build_joint_candidate.py`: freezes the twelve-layer calibrated candidate.
- `benchmark_joint_frozen.py`: paired comparison using the frozen source.
- `test_joint_cumulant.py`: independent bivariate integration, tensor slice
  identities, first Gaussian layer, optimized/reference equivalence, and
  an explicit check that calibration leaves the propagated state unchanged.

This is a moment closure, not an exact propagated density. Dropped tensor
components, finite third-history, truncation and covariance consistency remain
limitations. Local score gains do not establish hidden-test improvement.

## Final frozen comparison

Reducing quadrature from four nodes to two preserved the useful improvement
and resolved the measured overhead problem. On evaluation networks 8-11,
raw MSE was 9.575842e-7 versus 9.575858e-7 with four nodes. All six two-node
development/evaluation runs passed; maximum residual was 0.349542 seconds.
The final candidate applies NO fitted calibration.

We then froze a standalone source and compared it to the current submission
on eight additional networks (4-7 and 12-15). Neither the equations nor the
coefficients were changed during this comparison.

| Frozen source | Raw final MSE | Adjusted final MSE | Local failures |
|---|---:|---:|---:|
| Current venture runtime baseline | 1.336424354e-6 | 1.336424354e-7 | 0/8 |
| Twelve-layer joint closure, calibrated | 1.257183246e-6 | 1.257183246e-7 | 0/8 |
| Full joint closure, two nodes, no calibration | **9.238894887e-7** | **9.238894887e-8** | **0/8** |

The full candidate improved all eight networks individually; aggregate error
fell approximately 30.9%. It uses 202,869,442,542 FLOPs (about 9.23% of budget),
so its score multiplier remains 0.1. The standalone source omits diagnostic
collection, hence slightly fewer FLOPs than the scout. Mean residual was
0.337955 seconds, maximum 0.346854 seconds; prediction wall time was at most
8.10 seconds. These are local measurements, not a remote timing guarantee.

Six mathematical/equivalence tests passed and the official `whest validate`
accepted the standalone source. The frozen eight-network comparison is in
`results/joint_full_frozen_20261001.json`; the scout is in
`results/joint_cumulant_q2_20261001.json`. The final source is
`joint_full_candidate/estimator.py`, SHA-256
`3539c3eaa325a5ffa92279fb3e39034ba5b36d73c97612a7ecb0680289bca8d9`.

To reproduce the frozen comparison:

```powershell
& .venv-phase2/Scripts/python.exe -m pytest research/phase2/test_joint_cumulant.py -q
& .venv-phase2/Scripts/python.exe research/phase2/benchmark_joint_frozen.py --candidate joint_full_candidate --output research/phase2/results/joint_full_frozen_repeat.json
$env:PYTHONIOENCODING='utf-8'
& .venv-phase2/Scripts/whest.exe validate --estimator research/phase2/joint_full_candidate/estimator.py --format json
& .venv-phase2/Scripts/whest.exe run --estimator research/phase2/joint_full_candidate/estimator.py --runner subprocess --n-mlps 1 --n-samples 100 --seed 20260926 --max-threads 4 --format json
```

The 100-sample subprocess run is an execution smoke test only. Its noisy
Monte Carlo MSE must not be compared to the billion-sample public reference
targets used in the accuracy experiments above.

**Subprocess result:** the two-node full candidate failed the residual cap:
0.614115 seconds, versus a limit of 0.4 seconds. Predict wall time was 10.2797
seconds and FLOPs matched the in-process benchmark. No numerical exception
occurred. The failure is retained in `results/joint_full_subprocess_20261001.json`.
This candidate must not be presented as submission-ready on the basis of
the in-process successes.

The one-node full variant also failed the official subprocess cap: residual
0.525379 seconds, predict wall time 8.00097 seconds, 200,816,801,301 FLOPs.
Its report is `results/joint_q1_subprocess_20261001.json`. Since it failed the
execution gate, it was not promoted or given an additional accuracy sweep.
Neither failure was discarded or replaced with a favorable rerun.

The official worker loads a separate process and executes under its own
BudgetContext, including output materialization. The precise cause of the
timing gap has not been isolated; it would be incorrect to attribute it to a
specific mechanism without further profiling. Subsequent performance work
should use this official subprocess gate early, rather than infer eligibility
from in-process timing. No benchmark or metering code was changed.

The twelve-layer fallback also failed the official subprocess cap: residual
0.712354 seconds, predict wall time 12.63955 seconds, 166,071,553,182 FLOPs.
Its report is `results/joint_prefix12_subprocess_20261001.json`.

**Final decision:** preserve all research sources and results, but do not
package or promote any of these variants for upload. The previous archive
`../../submissions/phase2_best_20261001.tar.gz` remains the latest candidate
with a previously successful official subprocess check. Its narrow timing
margin is documented in `RUNTIME_BASELINE.md`.

The actionable result is that joint higher moments improve propagation and
accuracy in this Phase 2 regime. The remaining implementation work is to
reduce state-update overhead under the official runner while retaining the
full joint state. Improving quadrature precision is not the priority.

## Runtime follow-up, 2 October 2026

The original sources, results, subprocess failures and upload archives above
are preserved. A separate derivative starts from `joint_full_candidate/estimator.py`
and is saved as `joint_full_runtime/estimator.py`, SHA-256
`bd4c808d47068dcf2b0515472925598f56c43b2b603a2159b982d4d1f9dbce7e`.

The fresh official local/subprocess comparison measured 0.267779 / 0.350419 s
residual for the original, both passing under current conditions. Materialization,
weight layout and retained JSON/GC diagnostics did not isolate the cause of the
historical failure. No runner or metering code was changed.

The derivative specializes terminal covariance/third/fourth calculations to
diagonals and the first ReLU moment, moves kernel coefficients onto response
vectors, removes unused/repeated operations and reuses dead state buffers via
public metered operations. It retains the full joint state in every preceding
layer, the two quadrature nodes, one-source history and no calibration.

All ten paired exposed public networks pass the declared numerical equivalence
checks. Largest all-layer prediction difference is 3.814697e-6; final MSE on the
same frozen eight networks is 9.238858483e-7 versus 9.238894887e-7 for the original.
The difference is rounding drift. All 18 tests and official validation pass.

Crucially, the unchanged official subprocess execution gate passes **6/6**
predictions across three seeds and both first/repeated worker calls. Maximum
residual is **0.298400 seconds**, mean 0.294966 seconds, at 196,317,442,558 FLOPs.
The full evidence, timing limitations and reproduction commands are in
`JOINT_RUNTIME_OPTIMIZATION.md`. The accuracy report is
`results/joint_full_runtime_accuracy_20261002.json`; official gate reports are
`results/joint_full_runtime_final_gate_20261002*`; preservation/source audit is
`results/joint_full_runtime_audit_20261002.json`.

This derivative now passes the local official gate. It was not promoted from
in-process timing, and no new upload archive was created. The previous
`submissions/phase2_best_20261001.tar.gz` remains the latest packaged artifact.

At the user's subsequent request, the tested derivative was packaged separately
as `submissions/phase2_joint_full_runtime_20261002.tar.gz`. Official package
validation passed with no issues; estimator bytes match the frozen source and
all manifest hashes were verified. The previous archive is unchanged. Archive
SHA-256: `89035799f6acfe5dd14948fbedefd37ea2bc35472ce2ce72518c4485676782b3`.
See `results/joint_full_runtime_package_audit_20261002.json`. No upload was made.

## Paired fourth-transport diagnostic, 3 October 2026

The successful runtime source and archive remain frozen. A separate diagnostic
completes the six cross-pairings within the retained fourth-tensor projection:
`Q' = G + 2H - 2D`, where `G=(W*W).T @ Q @ (W*W)`,
`H_ab=(w_a*w_b).T @ Q @ (w_a*w_b)`, and
`D=(W*W).T @ diag(diag(Q)) @ (W*W)`. The existing marginal formula is already
correct for that projection; only its off-diagonal transport is changed.
Independent dense-tensor tests verify the contraction.

A fixed width-32, depth-16 pilot on four synthetic networks with four independent
262,144-sample streams each improved aggregate final mean risk by 1.51%, but
worsened final relative incoming-variance RMSE by 0.52%. Local oracle transport
loss improved early and was mixed later. The predeclared requirement to improve
both final metrics by at least 5% was not met, so no dominant-mode approximation,
history/damping sweep, or submission promotion followed. Small-network closure
errors are large and this does not settle the width-1024 case.

Eight tests passed; the artifact audit verified 81 pre-existing files unchanged.
Details and reproduction are in `FOURTH_TRANSPORT_DIAGNOSTIC.md`, with raw
evidence in `results/fourth_transport_diagnostic_20261003.json` and
`results/fourth_transport_audit_20261003.json`. Any future deployable change
continues to require the unchanged official subprocess execution gate.
