# arc-whestbench-p2-higher-moments-2026

Higher-order joint activation moments for the **Phase 2** (`v2-phase2`) benchmark MLPs
of the ARC WhiteBox Estimation Challenge 2026 — the 1024-wide, 16-layer analog of
[`arc-whestbench-higher-moments-2026`](https://huggingface.co/datasets/keenanpepper/arc-whestbench-higher-moments-2026).

Source MLPs: the 100 MLPs of the `mini` split of
[`aicrowd/arc-whestbench-public-2026`](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026)
at revision `v2-phase2` (width 1024, depth 16, He-init, no biases, fp32 weights,
input `x ~ N(0, I)`, forward `a_l = relu(a_{l-1} @ W_l)`).

Moments are estimated from **N = 1e8** Monte-Carlo samples per MLP (fresh samples,
independent of the dataset's own N = 1e9 ground-truth bake), forward pass in fp32
(matching the grader dtype), all moment accumulation in fp64 on GPU.

## Files

`moments/mlp_{idx:05d}.npz` — one file per MLP (`idx` = row index in the mini split), ~322 MB each:

| key | shape | dtype | contents |
|---|---|---|---|
| `pre_mean`, `pre_m2`, `pre_m3`, `pre_m4` | (16, 1024) | f64 | pre-activation marginal raw moments E[z^k] |
| `mean`, `post_m2`, `post_m3`, `post_m4` | (16, 1024) | f64 | post-activation marginal raw moments E[a^k] |
| `pre_M11` | (16, 1024, 1024) | f32 | E[z_i z_j] |
| `pre_M21` | (16, 1024, 1024) | f32 | E[z_i^2 z_j] (first index squared) |
| `pre_M31` | (16, 1024, 1024) | f32 | E[z_i^3 z_j] (first index cubed) |
| `pre_M22` | (16, 1024, 1024) | f32 | E[z_i^2 z_j^2] |
| `M11` | (16, 1024, 1024) | f32 | post-activation E[a_i a_j] |
| `gt_mean` | (16, 1024) | f32 | the source dataset's N=1e9 `all_layer_means` (copied for convenience) |
| `global_index`, `mlp_name`, `n_samples`, `sample_seed` | scalars | | provenance |

Layer convention: index `l` = weight matrix `l` (0-based); `pre_*[l]` are moments of
`z_l = a_{l-1} @ W_l`, `post`/`mean`/`M11[l]` are moments of `a_l = relu(z_l)`.
Layer-0 pre-activations are exactly Gaussian with covariance `W_0^T W_0`.

`cache/mlp{idx:05d}.npz` — derived closure-training cache (~151 MB each), the Phase-2
analog of the phase-1 `ap_a4_bake` cache. P = 1024·1023/2 = 523,776 upper-triangular pairs
(`np.triu_indices(1024, 1)` order):

| key | shape | contents |
|---|---|---|
| `static` | (15, P, 3) f32 | normalized cumulant pair features for layers 1..15: x1 = sym-normalized κ21, x4 = sym-normalized κ31, x5 = normalized κ22 |
| `rn` | (15, P) f32 | Mehler-8 post-act second-moment residual, (M11_post − Mehler8(μ,σ,ρ)) / (σ_i σ_j) |
| `rho` | (15, P) f32 | true pre-act correlation |
| `a` | (15, 1024) f32 | μ/σ per neuron (layers 1..15) |
| `g3`, `g4` | (16, 1024) f32 | pre-act marginal standardized cumulants γ3, γ4 (all layers) |
| `mean` | (16, 1024) f32 | post-act mean (this bake, N=1e8) |
| `gt_mean` | (16, 1024) f32 | N=1e9 ground truth |

## Provenance / verification

- Sampling seed per MLP: `20260824 ^ (idx * 2654435761 % 2**63)` (torch CUDA Philox).
- Baked 2026-08-24 on H200, torch fp64 accumulators, chunk 131072
  (`whest` repo, `research/phase2/ap_p2_bake_moments.py` / `ap_p2_cache.py`).
- Parity: baked `mean` agrees with the source dataset's N=1e9 `all_layer_means`
  at the expected MC noise level (mean |err| ≈ 1.7e-5, rel ≈ 2.9e-5 at layer 15);
  `diag(pre_M11) == pre_m2` and `diag(M11) == post_m2` to fp32 rounding (≈6e-8 rel).
- Pair-moment entries carry ~1e-4-relative MC noise at N=1e8; marginal moments less.

## License

MIT, same as the source dataset.
