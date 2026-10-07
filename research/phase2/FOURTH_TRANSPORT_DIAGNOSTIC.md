# Complete paired fourth-order transport: fixed diagnostic

3 October 2026. The successful standalone estimator and submission archive
remain frozen. This experiment completes one identifiable omission within the
paired fourth-tensor projection; it does not change the submitted implementation.

**Result:** completing the projection improved aggregate final mean error by
1.51%, but worsened final incoming relative variance RMSE by 0.52% on four fixed
small networks. The predeclared requirement to improve both metrics by at least
5% was not met. No dominant-mode variant, older-history variant or damping sweep
was pursued, and no new candidate was promoted or packaged.

## Derivation and independent check

Let the retained tensor satisfy `U[i,i,i,i]=Q[i,i]` and
`U[i,i,j,j]=Q[i,j]` for `i != j`, including all six permutations. All other index
patterns are zero. For columns `w_a,w_b` of the next weight matrix, write

\[
S=W\odot W,\qquad G=S^\top Q S,
\]
\[
H_{ab}=(w_a\odot w_b)^\top Q(w_a\odot w_b),\qquad
D=S^\top\operatorname{diag}(\operatorname{diag}Q)S.
\]

The complete paired plane is

\[
Q'_{ab}=G_{ab}+2H_{ab}-2D_{ab}.
\]

For each `i<j`, the six permutations contribute

\[
Q_{ij}\left(W_{ia}^2W_{jb}^2+W_{ja}^2W_{ib}^2
             +4W_{ia}W_{ja}W_{ib}W_{jb}\right).
\]

The current Gram formula retains the first two terms. The missing term can be
computed directly as `2*(H-D)` using `Q_off = Q-diag(diag(Q))`, which is the
contraction `einsum('ij,ia,ib,ja,jb->ab',Q_off,W,W,W,W)` times two. On the
diagonal, `H_aa=G_aa`, so the existing formula `3*G_aa-2*D_aa` is already
correct within this projection. Both variants explicitly retain that same
diagonal; the experiment changes only the off-diagonal incoming plane.

`test_fourth_transport.py` independently materializes the retained dense
fourth tensor at input width 4 and transforms all four indices to output width
3. Four random signed symmetric sources agree with the new paired contraction.
Other tests check unchanged diagonals, identity/diagonal-source limits, exact
Gram-replica parity, and translation-invariant joint k-statistics. **8 tests
pass**; log: `results/fourth_transport_tests_20261003.txt`.

## Fixed experiment and measurement

Before looking at results, fix four synthetic He-Gaussian networks of width 32
and depth 16, seeds **3103, 4103, 5103, 6103**. Compare only the preserved Gram
closure and complete paired transport. History 1, two quadrature nodes, all
16 rich layers, clipping, and all other response/state formulas remain the same.
The research replica's Gram means and diagnostic traces match the frozen
runtime source exactly. Predictions are calculated before reference sampling;
no sampled moments enter the recurrence and no coefficients are fitted.

For each network, use four independent Monte Carlo streams of 262,144 inputs
(1,048,576 samples total per network). Record all-layer means and incoming
variances, plus paired fourth cumulants at source layers 1,4,8,12,15 and their
next-layer preactivations. Joint fourth k-statistics remove finite-sample
cumulant bias; their diagonal agrees with the usual marginal fourth k-statistic.

The reported mean error is the independent-stream cross-product risk, which
removes reference sampling variance. The paired MSE difference has a Student-t
95% interval over independent sampling streams. That interval concerns Monte
Carlo noise for these fixed networks, not uncertainty over new networks.
Incoming variance RMSE uses a pooled plug-in reference; fixed-normalizer paired
variance differences and independent-stream spread are saved as additional
checks. First-layer relative variance RMSE is about 0.144%, consistent with
reference noise for an exactly Gaussian incoming distribution.

## Results

Lower is better in both columns. Variance RMSE is relative, as a fraction.

| Seed | Gram final mean risk | Complete final mean risk | Mean-error improvement | Gram final variance RMSE | Complete final variance RMSE |
|---|---:|---:|---:|---:|---:|
| 3103 | 0.001839916 | 0.001805030 | 1.90% | 0.859043 | 0.827500 |
| 4103 | 0.026745453 | 0.026261537 | 1.81% | 2.380073 | 2.386258 |
| 5103 | 0.002218007 | 0.002123280 | 4.27% | 0.868773 | 0.893383 |
| 6103 | 0.009294598 | 0.009301738 | -0.08% | 1.488366 | 1.514205 |
| Aggregate | **0.010024493** | **0.009872896** | **1.51%** | **1.530744** | **1.538663** |

Aggregate variance RMSE worsens **0.517%**. Three of four networks improve
final means, but only one improves final variance. The aggregate final paired
mean-MSE change is `-1.515973e-4`, with sampling-only 95% interval
`[-1.526108e-4, -1.505838e-4]`. Thus the small mean gain is well resolved against
sampling noise, but does not repair the propagated variance or meet the
predeclared criterion. Layerwise gains are mixed; for example mean risk worsens
at layers 3-5,9,11-12, while variance RMSE improves at layer 14 but worsens at
layers 15-16. All layerwise data and intervals are retained.

An additional **offline oracle one-step** comparison applies each transport to
the same sampled source Q and compares to sampled next-layer Q. It measures
transport loss separately from recurrence error and never feeds the prediction:

| Source -> next layer | Gram off-diagonal risk | Complete off-diagonal risk | Improvement |
|---|---:|---:|---:|
| 1 -> 2 | 0.009540969 | 0.007783463 | 18.42% |
| 4 -> 5 | 0.194551720 | 0.181334157 | 6.79% |
| 8 -> 9 | 0.059014767 | 0.058302183 | 1.21% |
| 12 -> 13 | 3.089185913 | 3.130111449 | -1.32% |
| 15 -> 16 | 1.959538443 | 1.962708105 | -0.16% |

Completing pairings clearly helps this local measure early. Its loss is still
nonzero because actual fourth tensors also contain the omitted 3+1, 2+1+1 and
four-distinct-index patterns. Retaining the exact paired projection need not
improve its approximation to the full physical transport at every layer.

These small networks have much larger closure errors than the successful
width-1024 public comparison: aggregate final relative variance RMSE is around
153%. They are a mathematical diagnostic, not evidence of competition accuracy,
and do not rule out useful effects in a limited full-width follow-up. The study
does establish that this omission alone is not a reliable variance repair in
the tested recurrence. No synthetic-network MSE is compared to the public score.

## Compute constraint and decision

The frozen source uses **196,317,442,558 FLOPs**. The 0.1 multiplier floor ends
at **219,902,325,555.2 FLOPs**, leaving **23,584,882,997.2** additional FLOPs,
or **12.014%** of its current computation.

At width 32, the traced Gram recurrence uses 21,473,264 FLOPs and complete
transport uses 56,434,064, a 2.63x total cost in this diagnostic. Exact missing
pairing contraction has O(width^4) work, so this implementation is for small
diagnostics rather than deployment. These traced costs include the full terminal
plane needed for diagnosis; the submission specializes its terminal layer.
No in-process timings are interpreted as execution eligibility.

If a later limited diagnostic established useful gains, signed dominant modes
of `Q_off` would give a concrete approximation: for
`Q_off ~= sum_r lambda_r u_r u_r.T`, approximate the missing plane by
`2*sum_r lambda_r*(W.T @ diag(u_r) @ W)^2` elementwise, retaining the existing
exact projected diagonal. Both positive and negative modes matter. This costs
O(rank*width^3) plus mode extraction. No rank or coefficients were selected
here because the complete variant failed the decision criterion. Any future
deployable derivative must still pass the unchanged official subprocess gate.

## Evidence, preservation and reproduction

- `fourth_transport.py`: complete paired-projection contraction.
- `fourth_transport_candidate/estimator.py`: separate diagnostic replica.
- `diagnose_fourth_transport.py`: fixed pilot and reference statistics.
- `results/fourth_transport_diagnostic_20261003.json`: all four networks,
  all layerwise results, local transport loss and frozen decision.
- `results/fourth_transport_diagnostic_20261003_seed*.npz`: weights, stream
  references, predictions, variances, source Q and reference Q checkpoints.
- `results/fourth_transport_audit_20261003.json`: summary and unchanged-file audit.

The preservation audit verifies 81 pre-existing source/result/archive files.
One additional snapshot entry is the newly created research replica before it
was edited; both its initial-copy and final research hashes are recorded. The
frozen source remains
`bd4c808d47068dcf2b0515472925598f56c43b2b603a2159b982d4d1f9dbce7e`, and the
successful submission archive remains
`89035799f6acfe5dd14948fbedefd37ea2bc35472ce2ce72518c4485676782b3`.
All prior failures and archives are preserved.

Run from the starterkit root, using new output names for repeats:

```powershell
& .venv-phase2/Scripts/python.exe -m pytest research/phase2/test_fourth_transport.py -q --disable-warnings
& .venv-phase2/Scripts/python.exe research/phase2/diagnose_fourth_transport.py --output research/phase2/results/fourth_transport_diagnostic_repeat.json
& .venv-phase2/Scripts/python.exe research/phase2/summarize_fourth_transport.py --report research/phase2/results/fourth_transport_diagnostic_20261003.json --preserved research/phase2/results/fourth_transport_preserved_20261003.json --output research/phase2/results/fourth_transport_audit_repeat.json
```
