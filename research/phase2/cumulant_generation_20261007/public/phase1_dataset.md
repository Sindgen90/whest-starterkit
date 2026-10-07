---
license: mit
task_categories:
  - other
tags:
  - mechanistic-interpretability
  - moment-propagation
  - whitebox-estimation
  - neural-network-statistics
  - monte-carlo
pretty_name: WhestBench Higher Cross-Moments (pre & post activation)
size_categories:
  - 1K<n<10K
---

# WhestBench Higher Cross-Moments — pre & post activation ground truth

Monte-Carlo ground-truth **higher moments** of the activations of the 1000 random ReLU MLPs
in the [ARC WhiteBox Estimation Challenge 2026](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026)
`full` split. The official challenge dataset
([`aicrowd/arc-whestbench-public-2026`](https://huggingface.co/datasets/aicrowd/arc-whestbench-public-2026))
ships only the **per-neuron mean** post-activation. This dataset adds the full set of
**O(W²) moment-tensor components up to 4th order**, for **both the pre-activation and the
post-activation** at every layer — the joint statistics you need to study moment/cumulant
propagation beyond the mean.

- **1000 MLPs**, one `.npz` file each: `full/mlp_00000.npz` … `full/mlp_00999.npz`
- **N = 100,000,000** i.i.d. `N(0, I)` input samples per MLP
- Architecture (fixed): **width 256, depth 32**, bias-free, ReLU, He-init `W ~ N(0, 2/256)`
- ~63 GB total, fp32

---

## TL;DR — what each file contains

Indices in this README: `ℓ` = layer (0…31), `i,j` = neuron (0…255). `W = 256`, `D = 32`.

Every file holds the moments of two quantities at each of the 32 layers:

- **post-activation** `h⁽ˡ⁾ = relu(z⁽ˡ⁾)` → stored under the **bare** keys
- **pre-activation** `z⁽ˡ⁾ = h⁽ˡ⁻¹⁾ @ W⁽ˡ⁾` (before ReLU) → stored under the **`pre_`** keys
  (`h⁽⁻¹⁾` is the `N(0,I)` input, so `z⁽⁰⁾ = input @ W⁽⁰⁾`)

| key | shape | dtype | meaning (let `v` = `h⁽ˡ⁾` for bare keys, `z⁽ˡ⁾` for `pre_` keys) |
|---|---|---|---|
| `mean`     | (32, 256)       | f32 | `E[vᵢ]` — per-neuron mean. The post-ReLU per-neuron mean is the quantity the challenge asks estimators to predict; `mean[31]` is the final/output layer |
| `M11`      | (32, 256, 256)  | f32 | `E[vᵢ vⱼ]` — 2nd moment / Gram matrix (symmetric). The **(1,1)** component |
| `M21`      | (32, 256, 256)  | f32 | `E[vᵢ² vⱼ]` — the **(2,1)** component. **Not symmetric:** the *first* index `i` is the squared one |
| `M22`      | (32, 256, 256)  | f32 | `E[vᵢ² vⱼ²]` — the **(2,2)** component (symmetric) |
| `M31`      | (32, 256, 256)  | f32 | `E[vᵢ³ vⱼ]` — the **(3,1)** component. **Not symmetric:** the *first* index `i` is the cubed one |
| `m2`       | (32, 256)       | f32 | `E[vᵢ²]` — = `diag(M11)` |
| `m3`       | (32, 256)       | f32 | `E[vᵢ³]` — = `diag(M21)` |
| `m4`       | (32, 256)       | f32 | `E[vᵢ⁴]` — = `diag(M22)` |
| `pre_mean`, `pre_M11`, `pre_M21`, `pre_M22`, `pre_M31`, `pre_m2`, `pre_m3`, `pre_m4` | same as above | f32 | identical definitions, but for the **pre-activation** `z⁽ˡ⁾` |
| `official_alm` | (32, 256)   | f32 | the official `all_layer_means` (post-ReLU mean at N=1e9) copied for a built-in cross-check; `mean ≈ official_alm` to MC noise |
| `global_index` | scalar      | i64 | MLP index 0…999 in the `full` split |
| `name`     | scalar str      |     | e.g. `"full_00042"` |
| `n_samples`| scalar          | i64 | `100000000` |
| `width`, `depth` | scalar    | i64 | `256`, `32` |

> **All stored values are moments (expectations), already divided by N** — not raw sums.
> **Only ≤ 2 distinct neuron indices are stored.** The fully off-diagonal 3rd/4th tensors
> (`E[vᵢvⱼvₖ]`, …) are O(W³)/O(W⁴) (~TB) and are intentionally **not** included. See
> *"What's deliberately omitted"* below — for forward propagation they are recoverable as
> a W-contraction without ever materializing the big tensor.

---

## Index & layer conventions (read this)

- **Layer `ℓ`** runs `0…31`. `mean[ℓ]` is the post-ReLU mean *after* applying weight matrix
  `W⁽ˡ⁾`. The challenge scores the **last layer**, `mean[31]`.
- **Pre vs post at the same `ℓ`:** `pre_*[ℓ]` are moments of `z⁽ˡ⁾ = h⁽ˡ⁻¹⁾ @ W⁽ˡ⁾`; the bare
  `*[ℓ]` are moments of `h⁽ˡ⁾ = relu(z⁽ˡ⁾)`. They are computed from the **same samples**.
- **Asymmetry of `M21`/`M31`:** entry `[ℓ, i, j]` puts the *higher power on the first index*:
  `M21[ℓ,i,j] = E[vᵢ² vⱼ]`, `M31[ℓ,i,j] = E[vᵢ³ vⱼ]`. So `E[vᵢ vⱼ²] = M21[ℓ].T[i,j]`.
  `M11` and `M22` are symmetric.
- **Sign:** post-activations are `≥ 0` (ReLU output), so **all bare moments are non-negative**.
  Pre-activations are **signed**, so `pre_mean`, `pre_m3`, `pre_M21`, `pre_M31` (the odd-order
  pre quantities) can be negative — that is expected, not a bug.
- **Alignment is verified, not assumed:** the baked `mean[ℓ]` matches the official
  `all_layer_means[ℓ]` to ~3e-4 at every layer (the N=1e8 noise floor), which confirms the
  layer indexing here is identical to the official challenge dataset's.

---

## Connected moments / cumulants

The stored values are **raw** (non-central) moments. Most uses want the connected
(cumulant) forms. For neurons `i,j` at a layer, with `μ = mean`, `Mᵃᵇ` the raw moments
above, `M12 = M21.T`:

```
cov_ij   = M11 - μ_i μ_j                                   # = Cov(v_i, v_j)
var_i    = diag(cov)

# (2,1)  κ(v_i, v_i, v_j)  = E[(v_i-μ_i)² (v_j-μ_j)]
c21_ij   = M21 - 2 μ_i M11 - m2_i μ_j + 2 μ_i² μ_j

# (2,2)  κ(v_i, v_i, v_j, v_j)
cen22    = M22 - 2 μ_j M21 - 2 μ_i M12 + μ_j² m2_i + μ_i² m2_j + 4 μ_i μ_j M11 - 3 μ_i² μ_j²
c22_ij   = cen22 - var_i var_j - 2 cov_ij²

# (3,1)  κ(v_i, v_i, v_i, v_j)
cen31    = M31 - μ_j m3_i - 3 μ_i M21 + 3 μ_i μ_j m2_i + 3 μ_i² M11 - 3 μ_i³ μ_j
c31_ij   = cen31 - 3 var_i cov_ij
```

A reference implementation (`connected(npz, prefix="")` / `prefix="pre_"`) lives in the
reproduction script (see below). For a **Gaussian** pre-activation, `c21/c22/c31` vanish —
so the pre-activation cumulants are exactly the **leading non-Gaussianity** of `z⁽ˡ⁾`.

---

## Getting the MLP weights

Weights are **not** duplicated here; use the official challenge dataset, same index:

`global_index` is the position in the in-order concatenation of the 28 `full` shards
(shard sizes are not all equal, so stream rather than assume a fixed rows-per-shard):

```python
import numpy as np, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

def get_weights(global_index):
    gi = 0
    for s in range(28):
        p = hf_hub_download("aicrowd/arc-whestbench-public-2026",
                            f"data/full-{s:05d}-of-00028.parquet",
                            revision="v1-phase1", repo_type="dataset")
        col = pq.read_table(p, columns=["weights"]).column("weights")
        if gi + len(col) > global_index:
            return np.asarray(col[global_index - gi].as_py(), dtype=np.float64)  # (32,256,256)
        gi += len(col)
```

Forward convention (matches the moments): `z⁽ˡ⁾ = h⁽ˡ⁻¹⁾ @ W[ℓ]`, `h⁽ˡ⁾ = relu(z⁽ˡ⁾)`,
`h⁽⁻¹⁾ ~ N(0, I₂₅₆)`. **Sanity check:** the baked `mean` equals the official
`all_layer_means` to ~3e-4 (the N=1e8 MC noise on a unit-scale quantity).

---

## Loading

```python
import numpy as np
d = np.load("full/mlp_00042.npz")
print(d.files)
post_cov_last = d["M11"][-1] - np.outer(d["mean"][-1], d["mean"][-1])  # post covariance, final layer
pre_skew_diag = d["pre_m3"] - 3*d["pre_mean"]*d["pre_m2"] + 2*d["pre_mean"]**3  # pre-activation skew per layer
```

---

## Monte-Carlo precision

- **N = 1e8** samples/MLP. Statistical error scales as `1/√N`. Measured **relative**
  standard error of the components (median over significant entries):

  | quantity | rel. SE @ 1e8 |
  |---|---|
  | `mean` (1st)        | ~5e-5 |
  | `M11`/cov (1,1)     | ~1e-3 |
  | (2,2)               | ~5e-4 |
  | (3,1)               | ~1.6e-3 |
  | (2,1)               | ~2e-3 |

  The odd connected moments (2,1)/(3,1) are the noisiest because their *values* are small.
- **Arithmetic:** activations from **fp32** matmuls (TF32 **off** — its ~1e-3 error would
  exceed the noise floor); moment sums accumulated in **fp64**, stored fp32. fp64 sits far
  below the sampling floor, so the dataset is sampling-limited, not round-off-limited.
- Per-MLP RNG seed = `global_index` (torch CUDA `N(0,I)`). Moments are converged MC
  estimates; bit-exact reproduction additionally requires the same chunking/torch build.

---

## What's stored vs omitted

Only components with **≤ 2 distinct neuron indices** are kept; the fully off-diagonal
3rd/4th tensors (`E[vᵢvⱼvₖ]` with `i,j,k` distinct, etc.) are O(W³)/O(W⁴) (~TB) and are
omitted. What that does and doesn't cost, for forward moment propagation:

- **Mean channel — captured.** The next-layer mean `E[relu(z'ₐ)]` depends only on the
  *marginal* law of `z'ₐ = Σᵢ Wᵢₐ hᵢ`. By multilinearity of cumulants the marginal cumulant
  `κₚ(z'ₐ) = Σ Wᵢₐ…W·ₐ · κₚ[hᵢ,…]` is stored directly as `pre_*` at layer `ℓ+1`. So the
  contribution of layer `ℓ`'s joint cumulants — including the fully off-diagonal "(1,1,1)"
  block — *to the next marginal skew, hence to the mean* is captured, and can be isolated as
  `κ3(z') − (3) − (2,1)` (the (3)/(2,1) blocks being contractions of the stored diagonal &
  (2,1) data with `W`). This identity is verified to machine precision.
- **Pairwise (≤2-index) targets — captured.** `E[relu(z'ₐ)relu(z'_b)]` needs only
  ≤2-distinct-index pre-cumulants (`κ(z'ₐ,z'ₐ,z'_b)` = the stored pre `c21`, etc.), which are
  present.
- **Not recoverable.** The fully off-diagonal joint cumulants of the *next* pre-activation,
  `κ3(z'ₐ,z'_b,z'_c)` with `a,b,c` distinct, are themselves O(W³) and are **not** stored — the
  n³ restriction simply recurs one layer up. So this dataset lets you **measure** how much the
  n³ block feeds the marginal/mean channel (the n²-vs-n³ tractability question); it does **not**
  assert that n³ structure is unnecessary in general.

---

## Reproduction

Built with **`bake_moments.py`**, included in this repository (custom tooling — **not** part
of the upstream [whest-starterkit](https://github.com/AIcrowd/whest-starterkit); it depends
only on `numpy`, `torch`, `pyarrow`, `huggingface_hub`). Recipe: fp32 GEMMs → fp64
accumulators, TF32 **off**, atomic per-MLP work claiming, `--moments both`. The same file also
contains the `connected(npz, prefix=...)` helper (raw → cumulants) and `third_order_blocks(npz, weights)`
(the (3)/(2,1)/(1,1,1) decomposition described above). Example:

```bash
python bake_moments.py --source hf --split full --out ./out --n-samples 100_000_000 --moments both
```

## Citation

If you use this dataset, please cite the ARC WhiteBox Estimation Challenge 2026 and link
this repository. License: MIT (matching the challenge starter kit).
