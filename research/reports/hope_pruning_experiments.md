# HOPE-style pruning experiments

Date: 2026-08-03
Branch: `codex/fourier-walsh-4k-experiments`
Dataset: phase-1 mini, first 50 MLPs, depth 32, width 256

## Decision

Do not replace `structural_tail16_dead_m050_sobol_32768` with the HOPE-style
candidate. Extra pruning plus a linear mean correction is a robust counted-FLOP
win, but it does not improve the challenge's effective-compute-adjusted score.
The implementation remains available as an experimental submission variant.

## 1. Capacity diagnostic

We measured 976 leave-one-neuron interventions in layers 27-30 using 256 fixed
antithetic Sobol samples. Each predictor was compared with the MSE change in the
sampled final mean after zeroing one surviving neuron.

| Predictor | Spearman rank | Log Pearson |
|---|---:|---:|
| Gaussian tail probability | 0.8806 | 0.7957 |
| Alpha | 0.8806 | not meaningful across signs |
| Downstream path energy | 0.8565 | 0.7665 |
| HOPE one-step capacity | 0.8555 | 0.7664 |
| Second moment | 0.8545 | 0.7657 |

Conclusion: HOPE/downstream capacity is well calibrated on a log scale, but the
existing Gaussian tail statistic is the better within-layer pruning ranker.

## 2. Structured pruning

All variants used 32,768 fixed antithetic Sobol points. Extra pruning was applied
to each of layers 17-30 after the existing m050 dead classification.

| Candidate | Raw final MSE | Effective adjusted | Counted FLOPs |
|---|---:|---:|---:|
| Dead-only control | 4.9216e-7 | 2.3862e-7 | 115.54B |
| Tail, extra 8/layer | 5.1084e-7 | 2.4312e-7 | 112.57B |
| HOPE, extra 8/layer | 5.1213e-7 | 2.4288e-7 | 112.57B |
| Downstream, extra 8/layer | 5.1250e-7 | 2.4337e-7 | 112.57B |
| HOPE, extra 16/layer | 1.1306e-6 | 5.1645e-7 | 109.63B |

Tail ranking caused the least raw damage. Eight extra removals saved about 2.97B
FLOPs but still lost 1.8-2.0% in effective adjusted score. Sixteen removals caused
a large nonlinear error and was rejected.

## 3. Mean-preserving corrections

For tail-ranked extra-8 pruning, omitted Gaussian activation means were tested in
two forms:

- Broadcast correction: inject a constant omitted-mean contribution before each
  downstream ReLU.
- Linear response: propagate one width-256 correction vector through downstream
  weights and diagonal Gaussian ReLU gains, then adjust the final sampled mean.

At correction scale 0.50:

| Candidate | Raw final MSE | Effective adjusted | FLOP-only adjusted |
|---|---:|---:|---:|
| Dead-only control | 4.9216e-7 | 2.3862e-7 | 2.0802e-7 |
| Broadcast, all omitted | 4.9722e-7 | 2.4184e-7 | 2.0478e-7 |
| Linear response, all omitted | 4.9716e-7 | 2.3947e-7 | 2.0459e-7 |

Because the linear correction is added only at the output, its scale can be swept
exactly from cached predictions. Parity-held-out optima were 0.61 and 0.62. At
scale 0.61, the projected scores were:

- raw MSE: 4.9667e-7, 0.92% worse than control;
- effective adjusted: 2.3925e-7, 0.27% worse;
- FLOP-only adjusted: 2.0441e-7, 1.74% better.

Paired 20,000-draw bootstrap intervals for the percentage change versus control:

- raw: +0.03% to +1.93%;
- effective adjusted: -0.71% to +1.34%;
- FLOP-only adjusted: -2.69% to -0.75%.

The counted-FLOP gain is robust. The official effective-compute change is
unresolved and has a worse point estimate because the extra small remote-array
operations incur charged residual wall time.

## 4. Mergeability scout

We evaluated 10,547,259 meaningful neuron pairs across layers 17-30. Pairs below
1% of median activation energy or 1% firing rate were excluded.

- Maximum activation cosine per layer/MLP: median 0.9960.
- Maximum centered activation cosine: median 0.9069.
- Maximum functional cosine (activation times outgoing-weight alignment): median
  0.2085, maximum 0.3073.
- Functional pairs above 0.90: zero.

The high uncentered activation similarity mostly reflects shared positive means.
Direct functional alignment is weak, but a later merge-and-fold follow-up showed
that outgoing alignment is unnecessary when the victim row is folded into the
anchor's outgoing weights. See `research/reports/hope_merge_experiments.md`.

## Artifacts

- `results/hope_capacity_diagnostics_limit50.json`
- `results/hope_pruning_rank_limit50.json`
- `results/hope_pruning_rank_limit50.npz`
- `results/hope_pruning_correction_tail8_limit50_tail_extra08_mean_extra_s050.json`
- `results/hope_pruning_correction_tail8_limit50_tail_extra08_mean_all_s050.json`
- `results/hope_pruning_correction_tail8_limit50_tail_extra08_linear_all_s050.json`
- `submissions/structural_tail16_tail8_linear_all061_sobol_32768/`

## Next implication

Additional pruning should not be chosen by a per-neuron score alone. The next
credible direction is a joint, budget-neutral exchange: prune only neurons whose
estimated damage is below the measured QMC error, then reinvest exactly the saved
FLOPs into more samples or a higher-order correction. The linear response remains
useful as a cheap damage estimator even though it is not a winning standalone
submission.
