# Dead/On/Kink Structural Tail Scout

Date: 2026-08-02

Branch: `codex/fourier-walsh-4k-experiments`

## Question

Can antithetic Sobol sampling through layer 30, followed by conservative
dead/always-on/kink classification in the last two layers, approach dense
64k-sample accuracy while fitting the Phase 1 compute budget?

## Method

The offline scout is implemented in:

```text
experiments/run_structural_tail_scout.py
```

It performs the following operations:

1. Corrected-K2 propagation supplies the Gaussian preactivation z-score
   `alpha = mu / sigma` for the last two layers.
2. A neuron is classified as always-on when `alpha >= threshold`, dead when
   `alpha <= -threshold`, and uncertain otherwise.
3. Antithetic Sobol samples are propagated densely through layer 30.
4. Only uncertain layer-31 columns are sampled through ReLU.
5. Always-on layer-31 paths into uncertain final neurons are collapsed into a
   single linear map. Dead paths are omitted.
6. Always-on final means are evaluated linearly, dead final means are zero,
   and only uncertain final neurons are sampled through ReLU.

Matrix slicing is charged by `flopscope` at the reduced shape, so the
structural savings are real under the challenge FLOP model.

The full runs use 25 even-`mlp_id` MLPs for the corrected-K2 combination
coefficient and report results on 25 odd-`mlp_id` MLPs.

## Classification audit

On the first 50 Mini MLPs with a hypothetical 65,536-sample candidate:

| Threshold | Mean uncertain L31 | Mean uncertain L32 | Estimated mean budget fraction | Estimated max budget fraction |
|---:|---:|---:|---:|---:|
| `2.5` | `86.28` | `83.14` | `0.976` | `0.988` |
| `3.0` | `103.74` | `102.02` | `0.982` | `0.997` |
| `3.5` | `122.92` | `121.76` | `0.989` | `1.007` |
| `4.0` | `142.40` | `141.48` | `0.996` | `1.018` |

Threshold `3` is the useful accuracy/compute boundary. Threshold `2.5`
permits visible classification bias; thresholds `3.5` and `4` are too close
to or above the hard budget at 64k.

## 64k accuracy diagnostic

Using per-MLP digitally shifted antithetic Sobol samples, before residual
wall-time penalties:

| Tail | Raw final MSE | FLOP-only adjusted score | Mean FLOPs | Budget failures by FLOPs |
|---|---:|---:|---:|---:|
| Dense | `2.9906e-7` | `3.0361e-7` | `276.14B` | all; dense exceeds budget |
| Structural, threshold `2.5` | `3.1347e-7` | `3.0764e-7` | `267.09B` | `0%` |
| Structural, threshold `3.0` | `2.9936e-7` | `2.9547e-7` | `268.69B` | `0%` |

Threshold `3` preserves the dense estimate almost exactly. Mean wrong-sign
fractions for classified neurons are approximately `4e-5`; the rare errors
have negligible effect compared with sampling error.

This does not reproduce the externally reported `2.18e-7` raw MSE. The best
comparable raw result here is approximately `2.99e-7`.

### Effective-compute correction

The 64k candidate is not submission-safe in its current form. Array lifecycle
and point-construction work leave material residual wall time, charged at
`1e11` FLOPs/second. The 64k rule therefore crosses the combined compute
budget even though its counted FLOPs fit.

Precomputing the fixed input rule in `setup()` removes point-set construction
from `predict()`, but forward-pass array allocation still contributes residual
time. This makes a lower sample count the robust scoring choice.

## Valid sample-count comparison

These runs use one fixed antithetic Sobol rule precomputed off-budget, as a
shape-dependent lookup table in the intended `setup()` model.

| Samples | Combo lambda | Combo raw final MSE | Combo adjusted score | Mean effective compute | Max effective compute | Structural-vs-dense final MSE |
|---:|---:|---:|---:|---:|---:|---:|
| `32,768` | `0.976871` | `5.6623e-7` | **`3.1997e-7`** | `153.83B` | `158.43B` | `3.23e-10` |
| `49,152` | `0.992405` | `3.8447e-7` | `3.2545e-7` | `230.72B` | `235.27B` | `3.08e-10` |

Both candidates are valid on all 25 held-out MLPs. The 49k rule is more
accurate, but the reduction does not repay its larger compute multiplier.
The 32k rule is the best adjusted-score point tested.

For comparison:

```text
current 24k lattice target adjusted score: 4.0263e-7
structural 32k Sobol adjusted score:        3.1997e-7
relative improvement:                      about 20.5%
```

## Decision

Continue this method. The structural classification and two-layer collapse
are validated, and the 32k fixed antithetic Sobol rule is the strongest valid
local candidate found in this branch.

## Submission estimator

Implemented:

```text
submissions/structural_tail_sobol_32768/estimator.py
```

The estimator is a single grader-safe file. It imports only the standard
library, `flopscope`, `flopscope.numpy`, and `whestbench`. Joe--Kuo direction
parameters for dimensions 2--256 are embedded, so no external direction file
or NumPy dependency is required. `setup()` generates and retains the fixed
32k Gaussian point set once.

Validation and guard tests:

```text
ruff:                            passed
whest validate:                  passed
tests/test_submission_imports.py passed
```

Canonical local runner command:

```powershell
whest run `
  --estimator submissions\structural_tail_sobol_32768\estimator.py `
  --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 `
  --split mini --runner local --n-mlps 50
```

Canonical first-50 Mini result:

| Metric | Result |
|---|---:|
| Adjusted final-layer score | **`3.15e-7`** |
| Raw final-layer MSE | `4.91e-7` |
| All-layers MSE | `1.26e-6` |
| Mean score multiplier | `0.64010970` |
| Mean compute utilization | `0.64010970` |
| Failed MLPs | `0 / 50` |

The run took `135.65s` for 50 MLPs on the local machine. A direct one-MLP
budget audit measured setup at `1.20s`, `134.70B` analytical FLOPs, and
`151.89B` effective compute for that row.

### Untouched Mini confirmation

The estimator was subsequently run through all 100 Mini MLPs. The canonical
runner completed in `248.19s` with no failures:

| Metric | All 100 canonical run |
|---|---:|
| Raw final-layer MSE | `6.29e-7` |
| All-layers MSE | `1.31e-6` |
| Mean score multiplier | `0.62940399` |
| Mean compute utilization | `0.62940399` |
| Failed MLPs | `0 / 100` |

The console renderer output retained by the experiment tool truncated the
canonical aggregate adjusted-score row. A compact deterministic replay was
therefore used to separate the two halves and retain exact aggregates:

| Subset | Raw final MSE | All-layers MSE | Adjusted score | Mean multiplier | Failures |
|---|---:|---:|---:|---:|---:|
| MLPs 0--49 | `4.9133e-7` | `1.2553e-6` | `2.7466e-7` | `0.55952` | `0` |
| MLPs 50--99 | `7.6650e-7` | `1.3614e-6` | `4.2776e-7` | `0.55776` | `0` |
| All 100 | `6.2891e-7` | `1.3084e-6` | `3.5121e-7` | `0.55864` | `0` |

The raw-MSE split is deterministic and shows that the untouched second half is
materially harder for this fixed Sobol rule. Adjusted scores vary with local
residual timing: the canonical runner charged a `0.6294` mean multiplier,
whereas the compact replay charged `0.5586`. The candidate is robustly within
budget, but comparisons at this compute level must report the runner and
runtime context.

Packaged artifact:

```text
submissions/structural_tail_sobol_32768.tar.gz
```

The archive contains only `estimator.py` and `manifest.json`, as intended. The
installed `whest` version does not expose the older `validate-package`
subcommand; package validation is therefore covered by source validation,
the import guard, archive inspection, and the canonical runner test.

### Remote smoke compatibility fix

The original single-file archive passed the local NumPy-backed validator and
subprocess runner but failed the grader smoke. The first defensive revision
moved fixed-point-set generation out of `setup()`: it ships the exact float32
point set as a pickle-free `samples.npz` and loads it with the documented
`fnp.load(str(path))` API.

The grader traceback from that revision identified the confirmed root cause:
remote `flopscope` arrays are immutable, while the local backend had allowed
indexed assignments such as `mean1[active1] = values`. Revision 2 replaces all
four structural-tail assignments with functional sparse projections through a
shipped identity matrix. It also uses `fnp.add` for the uncertain-path sum.
Only the obsolete offline Sobol builder retains indexed writes, and those are
to plain Python lists rather than `fnp` arrays.

Upload this folder-mode revision 2 package, not either earlier archive:

```text
submissions/structural_tail_sobol_32768_remote_safe_v2.tar.gz
SHA256: CADAFDF98DCF3D6D996A486D3A6643FEA09C3849D66D81E8ECA9592FE5601525
```

The archive contains `estimator.py`, `samples.npz`, and `manifest.json`, is
29.7 MiB, passes `whest validate`, and completes a real-width first-MLP run
through the subprocess worker. Against the original implementation, its
first-MLP predictions differ by only `7.62e-16` final-layer MSE.

Potential follow-up before a Full run:

1. Test a small number of stronger setup-time Sobol scrambles at 32k without
   tuning on held-out MLPs.
2. Compare fixed versus a small bank of setup-time scrambles on a calibration
   subset, then freeze the choice before another untouched evaluation.

## Chunking and pilot reclassification

The follow-up scout is implemented in:

```text
experiments/run_chunked_pilot_structural_scout.py
```

Here, thresholds are standardized Gaussian preactivation distances,
`alpha = mu / sigma`, rather than activation values. At threshold `2.5`, the
one-sided Gaussian wrong-sign probability is approximately `0.62%`; at
threshold `3.0`, it is approximately `0.135%`. Thus `2.5` classifies more
neurons as dead or always-on and saves tail work, while `3.0` is more
conservative.

The pilot variant starts at threshold `2.5` and demotes a classified neuron
back to uncertain if any of 256 antithetic pilot preactivations crosses zero.
The efficient implementation propagates the full 32k prefix once, selects the
pilot rows from the resulting layer-30 matrix, and reuses that same matrix for
the structural tail. No prefix samples are evaluated twice.

### Chunk-size audit

On a six-MLP timing scout, chunked threshold-3 propagation preserved the
monolithic estimate to approximately `1.9e-12` paired final-layer MSE. It did
not reduce effective compute:

| Prefix batch | Mean effective compute | Mean residual time |
|---:|---:|---:|
| Monolithic | `153.85B` | `0.188s` |
| `16,384` | `154.90B` | `0.198s` |
| `8,192` | `156.15B` | `0.211s` |
| `4,096` | `158.42B` | `0.234s` |

Chunking reduces peak temporary memory but adds matrix-dispatch and reduction
overhead. The 32k point set is small enough that monolithic propagation is the
better scoring implementation on this machine.

### Balanced 50-MLP pilot result

The comparison calibrates the K2/sample coefficient on 25 even-ID MLPs and
evaluates 25 odd-ID MLPs. Candidate execution order alternates independently
within both parity groups to neutralize warm-cache bias.

| Candidate | Combo raw MSE | FLOP-only adjusted | Full adjusted | Mean FLOPs | Mean effective compute |
|---|---:|---:|---:|---:|---:|
| Threshold `3.0`, no pilot | **`5.6623e-7`** | **`2.8053e-7`** | **`3.2094e-7`** | `134.795B` | `154.337B` |
| Threshold `2.5`, pilot 256 | `5.6831e-7` | `2.8076e-7` | `3.2183e-7` | **`134.400B`** | **`154.025B`** |

The pilot demoted an average of `7.24` layer-31 and `7.72` layer-32 classified
neurons. It reduced counted FLOPs by `0.395B`, but its `0.37%` raw-MSE loss was
larger than the compute benefit. It loses both adjusted comparisons, so it is
not promoted to a submission estimator.

An earlier fixed-order run made the pilot appear `0.31%` better on full
adjusted score because it always ran second. The balanced rerun reversed that
conclusion. Residual-time comparisons this close must use balanced ordering or
separate-process measurements.

### Accounting takeaway

The useful techniques from the compute-channel discussion are conventional:
precompute the fixed point set in `setup()`, avoid repeated budget queries and
temporary construction in `predict()`, and reduce actual matrix shapes through
structural classification. Moving arithmetic to uninstrumented backing arrays
would only shift work into the residual-time channel and is not used here.

Decision: retain the current monolithic threshold-3 estimator.
