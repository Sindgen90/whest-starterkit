# HOPE merge-and-fold experiments

Date: 2026-08-03
Branch: `codex/fourier-walsh-4k-experiments`
Dataset: phase-1 mini, first 50 MLPs, depth 32, width 256

## Method

For an oriented victim-anchor pair `(i, j)`, the merge uses the K2 activation
second-moment Gram matrix `G`:

```text
c_ij = G_ij / G_jj
h_i approximately c_ij h_j
W_next[j, :] <- W_next[j, :] + c_ij W_next[i, :]
```

The victim is removed from sampled propagation. Candidate damage is ranked by
the HOPE-style downstream residual risk:

```text
risk(i -> j) = (G_ii - G_ij^2 / G_jj) * downstream_path_energy_i
```

All merge plans are derived from weights and K2 moments. Targets are never used
for selection. Predictions use 32,768 fixed antithetic Sobol points unless a
candidate explicitly changes the sample count.

## 1. Fixed merge counts

Merges were disjoint within each layer and applied in layers 17-30.

| Candidate | Raw final MSE | FLOP-only adjusted | Counted FLOPs |
|---|---:|---:|---:|
| Dead-only control | 4.921611e-7 | 2.080238e-7 | 115.541B |
| 1 merge/layer | 4.930340e-7 | 2.084375e-7 | 115.575B |
| **2 merges/layer** | **4.920364e-7** | **2.073324e-7** | **115.188B** |
| 4 merges/layer | 4.927335e-7 | 2.062618e-7 | 114.420B |
| 8 merges/layer | 4.986561e-7 | 2.059380e-7 | 112.903B |

Two merges per layer slightly improved mean raw MSE, although its paired raw
change was not significant: `-0.025%`, 95% bootstrap CI `[-0.430%, +0.353%]`.
Eight merges produced the best counted-FLOP score but caused a visible raw loss.

## 2. Global HOPE allocation

The global controller selected 28 disjoint merges across all eligible layers by
residual risk per estimated sampled FLOP saved. Its mean allocation was:

```text
layer 17: 11.06
layer 18:  9.02
layer 19:  5.88
layer 20:  1.86
layer 21:  0.18
layers 22-30: approximately zero
```

| Candidate | Raw final MSE | FLOP-only adjusted | Counted FLOPs |
|---|---:|---:|---:|
| Dead-only control | 4.921611e-7 | 2.080238e-7 | 115.541B |
| Fixed 2/layer | 4.920364e-7 | 2.073324e-7 | 115.188B |
| **Global 28** | **4.921570e-7** | **2.067165e-7** | **114.816B** |

Global-28 preserved raw MSE (`-0.00083%`, CI `[-0.00388%, +0.00204%]`) and
robustly improved FLOP-only adjusted score by `0.628%`, CI
`[-0.734%, -0.535%]`.

## 3. Residual mean correction

The merge residual mean `mu_i - c_ij mu_j` was propagated through downstream
weights and diagonal ReLU gains, at scales 0.5 and 1.0.

| Candidate | Raw final MSE |
|---|---:|
| Global-28, no correction | 4.921570419e-7 |
| Mean correction 0.5 | 4.921573870e-7 |
| Mean correction 1.0 | 4.921574862e-7 |

The through-origin regression already preserves almost all relevant mean. Both
corrections were numerically neutral and slightly worse, while adding remote
operations. Mean correction was rejected.

## 4. Residual variance correction

Residual variance was propagated diagonally with squared weights. Its ReLU mean
effect used `phi(alpha) / (2 sigma)` and variance transmission used
`Phi(alpha)^2`.

| Candidate | Raw final MSE |
|---|---:|
| Global-28, no correction | 4.921570419e-7 |
| Mean plus variance 0.5 | 4.921574002e-7 |
| Mean plus variance 1.0 | 4.921576588e-7 |

The correction again had negligible magnitude and worsened the point estimate.
It was rejected.

## 5. Combination study

| Candidate | Raw final MSE | FLOP-only adjusted | Counted FLOPs |
|---|---:|---:|---:|
| Dead-only control | 4.921611e-7 | 2.080238e-7 | 115.541B |
| **Global-28, 32,768** | **4.921570e-7** | **2.067165e-7** | **114.816B** |
| Global-28, 33,792 | 5.030913e-7 | 2.179997e-7 | 118.348B |
| Global-28, 34,816 | 4.877354e-7 | 2.169715e-7 | 121.881B |
| Global-28 plus extra-4 pruning | 4.968105e-7 | 2.059107e-7 | 113.289B |

The 34,816-point grid improved raw MSE by about 0.90% but cost too much. Extra-4
pruning improved the counted-FLOP score but worsened raw MSE by about 0.94%.
The plain global-28 merge remained the balanced algorithmic winner.

## Effective-compute blocker

The research planner materializes K2 covariance matrices and constructs pair
pools in NumPy. Mean plan construction time was 0.252 seconds per MLP. At the
challenge penalty of `1e11` FLOP-equivalents per residual second, this costs
about 25.2B effective FLOPs while global-28 saves only 0.73B counted FLOPs.

Consequently, global-28's measured effective adjusted score was about 18.9%
worse than control. This is an implementation-channel failure, not an accuracy
failure, but it means the current method should not be packaged as an estimator.

## Decision and next implementation target

No submission estimator was built from this version. A viable implementation
must select merges without host-side all-pairs planning. The global allocation
is stable enough to justify a specialized follow-up:

1. restrict mergeable layers to 17-20 with a static `[11, 9, 6, 2]` budget;
2. compute pair residuals in `flopscope.numpy`;
3. materialize only a small top-pair payload, not fourteen full covariance
   matrices;
4. keep uncorrected merge folding and 32,768 samples.

That follow-up must reduce additional residual time below roughly 7 ms per MLP
to beat the current estimator's effective score at the observed 0.73B FLOP
saving.

## Optimized follow-up

The specialized remote planner was subsequently implemented with the static
`[11, 9, 6, 2]` layer budget. It computes pair residuals in
`flopscope.numpy`, keeps four anchors per victim, and transfers only the top 256
pairs from each of layers 17-20.

On 50 MLPs:

| Candidate | Raw final MSE | Effective adjusted | Counted FLOPs | Effective compute |
|---|---:|---:|---:|---:|
| Dead-only control | 4.921611e-7 | 2.401447e-7 | 115.541B | 133.272B |
| **Optimized static merge** | **4.921553e-7** | **2.394487e-7** | **114.824B** | **133.044B** |

Host planning fell from 252 ms to 0.53 ms per MLP. The effective adjusted score
improved by 0.290%; its paired 95% bootstrap interval was
`[-0.634%, +0.035%]`. The FLOP-only gain was 0.621%, with interval
`[-0.724%, -0.529%]`, and won on all 50 MLPs.

This optimized version warranted a grader-ready experimental estimator in
`submissions/structural_tail16_hope_merge_static_sobol32768/`.

## Artifacts

- `experiments/run_hope_merge_scout.py`
- `results/hope_merge_merge_limit50.{json,npz}`
- `results/hope_merge_global_limit50.{json,npz}`
- `results/hope_merge_mean_limit50.{json,npz}`
- `results/hope_merge_variance_limit50.{json,npz}`
- `results/hope_merge_combine_limit50.{json,npz}`
- `experiments/run_hope_merge_optimized_scout.py`
- `results/hope_merge_optimized_limit50.{json,npz}`
