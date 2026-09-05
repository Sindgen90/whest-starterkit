# HOPE Follow-up Under flopscope 0.10.0

## Setup

- Dataset: `aicrowd/arc-whestbench-public-2026@v1-phase1`, Mini split, first 50 MLPs.
- Evaluator libraries: `flopscope==0.10.0`, `whestbench==0.14.0`.
- Sampling: 32,768 antithetic Sobol points unless a residual sample count is named.
- Baseline: corrected K2 plus structural dead-neuron pruning over the last 16 layers.
- HOPE-28: centered one-anchor merge folding in human-numbered layers 17-20, with
  `11, 9, 6, 2` merges respectively.
- Scores use the existing parity-cross-fitted final-layer combination coefficient.

All figures below are means over the same 50 MLPs. FLOPs and effective compute are
billions per MLP. Lower is better for every numeric column.

## Main Result

| Estimator | Raw final MSE | Adjusted score | FLOP-only score | FLOPs (B) | Effective (B) |
|---|---:|---:|---:|---:|---:|
| float64 structural control | 4.921611e-7 | 2.465471e-7 | 2.128354e-7 | 118.201 | 136.860 |
| float64 HOPE-28 | 4.921553e-7 | 2.460809e-7 | 2.115886e-7 | 117.509 | 136.767 |
| float32 structural control | 4.921613e-7 | 2.428868e-7 | 2.089817e-7 | 116.071 | 134.966 |
| **float32 HOPE-28** | **4.921565e-7** | **2.415959e-7** | **2.077336e-7** | **115.378** | **134.251** |

Moving the analytic K2 pass from float64 to float32 changes HOPE-28 raw MSE by only
`+0.00023%`, saves `2.131B` counted FLOPs and `2.515B` effective operations, and
improves the measured adjusted score by `1.82%`. Relative to the float32 structural
control, HOPE-28 saves another `0.693B` FLOPs and improves adjusted score by `0.53%`
without a meaningful raw-MSE penalty.

The float32 HOPE-28 estimator is therefore the promoted candidate.

## Six Experiments

### 1. Evaluator rebaseline

The previous float64 estimator was rerun under the new cost model. HOPE-28 still
helped, but its effective-adjusted gain over the structural control fell from about
`0.29%` under flopscope 0.8 to `0.19%` under 0.10. Data movement and planner overhead
now consume more of the arithmetic saved by folding neurons.

### 2. Float32 K2

This was the decisive experiment. K2 supplies classification statistics and the
analytic half of the final blend; float32 precision is sufficient for both. The
sampling path was already float32. No change to thresholds, samples, or merge policy
was needed.

### 3. v0.10 operation ledger

For the first Mini MLP, the packaged candidate used approximately:

| Component | FLOPs (B) |
|---|---:|
| float32 K2 and planning statistics | 2.205 |
| structural 32k propagation | 110.860 |
| total | 113.066 |

Across all 50 MLPs the total averaged `115.378B`; variation comes from the number of
neurons classified dead and the resulting sparse matrix sizes. The main remaining
cost is still propagating 32k samples, not the K2 pass.

### 4. Affine and two-anchor merges

| Merge rule | Raw final MSE | Adjusted score | FLOPs (B) | Effective (B) | Decision |
|---|---:|---:|---:|---:|---|
| centered affine, one anchor | 4.921640e-7 | 2.702357e-7 | 115.337 | 154.447 | reject |
| centered affine, two anchors | 4.921657e-7 | 3.439001e-7 | 115.340 | 188.948 | reject |

Regression coefficients and intercept folding did not improve raw error, and host
planning time was heavily penalized by effective-compute safeguards. Two anchors
were especially expensive. Any future regression merge must use a much cheaper,
more vectorized planner.

### 5. Coupled residual correction

The estimator tested
`HOPE_32768 + mean(full_m - HOPE_m)` using coupled antithetic Sobol points.

| Residual samples | Raw final MSE | Adjusted score | FLOPs (B) | Effective (B) | Decision |
|---:|---:|---:|---:|---:|---|
| 2,048 | 4.921891e-7 | 2.744048e-7 | 131.164 | 152.260 | reject |
| 4,096 | 4.912327e-7 | 3.071688e-7 | 146.819 | 170.696 | reject |
| 8,192 | 4.910691e-7 | 3.720123e-7 | 178.130 | 206.771 | reject |

The 4k and 8k corrections improve raw MSE by about `0.19%` and `0.22%`, confirming
that merge bias is measurable. They are nevertheless much too costly under the
compute-adjusted objective because they repeat a nearly full network pass.

### 6. Aggressive layers 13-24 schedules

| Schedule | Raw final MSE | Adjusted score | FLOPs (B) | Effective (B) | Decision |
|---|---:|---:|---:|---:|---|
| 48 merges + 4k residual | 4.968877e-7 | 3.472429e-7 | 145.847 | 192.738 | reject |
| 80 merges + 4k residual | 4.970227e-7 | 3.403592e-7 | 144.732 | 188.207 | reject |
| 80 merges + 8k residual | 4.912512e-7 | 4.066350e-7 | 175.838 | 225.161 | reject |
| 112 merges + 8k residual | 4.912042e-7 | 3.955711e-7 | 174.622 | 221.333 | reject |

Starting merges at layer 13 introduces too much bias with a 4k correction. An 8k
correction recovers the raw error, but adds roughly `63B` operations; the larger
merge schedules recover only a few billion. Deeper HOPE expansion is not viable in
this form.

## Evaluator Caveat

`whestbench 0.14.0`'s in-process local runner constructs MLP weights without an
explicit dtype, so list-backed dataset weights become float64 locally. The subprocess
worker explicitly casts weights to float32, matching the grader. Consequently the
in-process runner reports about twice the expected dtype-priced FLOPs and can show
false budget failures for this estimator.

Use `--runner subprocess` for grader-matched measurements. A 50-MLP subprocess run
completed with zero failures, approximately `115.2B` FLOPs and `133.6B` effective
compute per MLP, raw MSE `4.92e-7`, and adjusted score `2.40e-7`.

## Decision And Next Edge

Promote float32 HOPE-28 unchanged. Do not add residual correction or extend merging
to layer 13 yet. The next promising experiments are narrower:

1. Replace the top-pair planner with a fully vectorized, data-movement-aware rule.
2. Select merges by estimated downstream error per saved FLOP, not correlation alone.
3. Spend a tiny correction only on selected high-risk merged neurons or layers,
   avoiding a second full-network pass.
4. Sweep modest merge counts in layers 17-24 under float32 and v0.10, where the
   operation ledger says savings can still exceed planner cost.
