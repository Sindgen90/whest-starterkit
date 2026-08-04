# Fourier/Walsh Spectral Diagnostics

Date: 2026-08-02

Branch: `codex/fourier-walsh-4k-experiments`

## Question

Can Fourier/Walsh analysis identify stable spectral structure in the random-MLP family, enough to improve the 4096-sample RQMC rule before building a new estimator?

## Implementation

Added:

```text
experiments/run_spectral_pointset_diagnostics.py
```

The script computes:

- ordinary FFT energy for rank-1 lattice ordered outputs;
- Walsh-Hadamard energy for Sobol/digital-net ordered outputs;
- stability of spectral profiles across MLPs and randomized shifts;
- dyadic energy bands;
- a cheap one-sparse alias-risk proxy for candidate 4096 lattice multipliers.

The diagnostic projects final-layer outputs onto:

- the all-ones direction;
- five selected neurons;
- eight fixed Rademacher directions.

## 50-MLP scout results

### 4096 rank-1 lattice, two shifts

Output:

```text
results/spectral_pointset_diagnostics_limit50_shifts2_lattice4096.json
```

Summary:

```text
profiles:                100
effective bin fraction:  0.962
top-8 energy:            0.0040
top-32 energy:           0.0157
max-bin energy:          0.000506
mean pairwise corr:      0.198
```

Interpretation: the Fourier spectrum is very broad. There is weak but nonzero family-level stability, not a concentrated set of dangerous bins.

### 24576 rank-1 lattice, one shift

Output:

```text
results/spectral_pointset_diagnostics_limit50_shifts1_lattice24576.json
```

Summary:

```text
profiles:                50
effective bin fraction:  0.863
top-8 energy:            0.0024
top-32 energy:           0.0090
max-bin energy:          0.000334
mean pairwise corr:      0.414
```

Candidate one-sparse alias penalties for 4096-node multipliers were nearly tied:

```text
50933:      0.0072035
117917:     0.0072074
17797:      0.0072182
1048573:    0.0072211
2891336453: 0.0072211
```

Interpretation: the larger lattice shows more stable spectral shape, but the energy is still broad and this simple alias proxy barely distinguishes candidate generators.

### 4096 Sobol, two digital shifts

Output:

```text
results/spectral_pointset_diagnostics_limit50_shifts2_sobol4096.json
```

Summary:

```text
profiles:                100
effective bin fraction:  0.912
top-8 energy:            0.0069
top-32 energy:           0.0261
max-bin energy:          0.000928
mean pairwise corr:      0.233
```

Interpretation: Walsh energy is slightly more concentrated than lattice Fourier energy at 4096, but still broad.

## Direct multiplier test

The 24576-spectrum proxy chose multiplier `50933` as the lowest-risk candidate, narrowly ahead of the incumbent `17797`. I tested both through the existing corrected-K2 + lattice convex-combo harness on the first 50 Mini MLPs.

Commands:

```powershell
python experiments\run_grid_combo.py --grid lattice --limit-mlps 50 --n-samples 4096 --seed-offset 8675309 --multiplier 17797
python experiments\run_grid_combo.py --grid lattice --limit-mlps 50 --n-samples 4096 --seed-offset 8675309 --multiplier 50933
```

Results:

| Multiplier | Lambda | Grid final MSE | Combo final MSE | Adjusted score | Combo all-layer MSE |
|---:|---:|---:|---:|---:|---:|
| `17797` | `0.784548` | `5.6167e-06` | `4.5267e-06` | `4.5267e-07` | `1.2424e-05` |
| `50933` | `0.799122` | `9.3178e-06` | `7.4221e-06` | `7.4221e-07` | `1.4842e-05` |

The spectral-scout candidate was substantially worse than the incumbent.

## Decision

Do not build an estimator from the current Fourier/Walsh idea yet.

The family spectra are not flat, but they are broad enough that a small set of recurring dangerous modes is not visible. The simple Fourier alias objective did not transfer to direct 50-MLP scoring.

Next spectral work, if any, should be more local:

- use residual spectra of `target - corrected_k2`, not raw network outputs;
- score candidate rules by actual paired residual reduction on calibration rows;
- treat Sobol/Walsh as a separate path, since Sobol still has better all-layer behavior in earlier experiments.

## Residual-aware scout

Added:

```text
experiments/run_residual_spectral_scout.py
```

Important correction: subtracting corrected K2 from every sampled final output
does not change nonzero Fourier/Walsh coefficients; it only changes the zero
bin. The useful residual question is therefore whether spectral features of a
4096-sample rule predict the paired final-layer residual:

```text
target_final - corrected_k2_final
```

The script evaluates each candidate lattice multiplier by:

- computing corrected K2 once per MLP;
- computing the 4096 lattice prediction and ordered final sample outputs;
- fitting `lambda` on even `mlp_id` rows for `k2 + lambda * (grid - k2)`;
- evaluating held-out odd `mlp_id` rows;
- correlating nonzero Fourier concentration/band features with projected error
  quantities.

Command:

```powershell
python experiments\run_residual_spectral_scout.py --limit-mlps 50
```

Output:

```text
results/residual_spectral_scout_limit50_n4096_c5.json
```

### Held-out result

| Multiplier | Mod 4096 | Lambda | Eval grid final MSE | Eval combo final MSE |
|---:|---:|---:|---:|---:|
| `17797` | `1413` | `0.784548` | `5.6167e-06` | `4.5267e-06` |
| `2891336453` | `2821` | `0.814692` | `5.5888e-06` | `4.6011e-06` |
| `117917` | `3229` | `0.855669` | `5.9176e-06` | `4.7713e-06` |
| `1048573` | `4093` | `0.809894` | `8.2541e-06` | `6.4761e-06` |
| `50933` | `1781` | `0.799122` | `9.3178e-06` | `7.4221e-06` |

Calibration ranking did not transfer: multiplier `117917` had the best even-row
calibration combo MSE (`4.8975e-06`) but lost to the incumbent on odd-row
evaluation.

### Residual spectral signal

The strongest correlations are stable in form but not directly actionable:
Fourier concentration features correlate with larger residual/error magnitude.
For the incumbent `17797`, held-out examples include:

| Feature | Response | Pearson |
|---|---|---:|
| `top_8_energy` | `abs_combo_error` | `0.423` |
| `top_32_energy` | `abs_k2_error` | `0.416` |
| `top_32_energy` | `abs_combo_error` | `0.416` |
| `top_8_energy` | `abs_k2_error` | `0.415` |
| `top_8_energy` | `abs_grid_error` | `0.412` |

Interpretation: spectra can identify harder rows/projections, but this first
residual-aware scout does not identify a better 4096 lattice multiplier than
the incumbent.

## Updated decision

Do not build a new estimator from Fourier/Walsh point-set selection yet.

The residual analysis gives a useful diagnostic for adaptive compute or error
prediction, but the first held-out candidate search still keeps the incumbent
4096 lattice (`17797`, shift `8675309`) as the best of the tested rules. The
next high-upside work is likely back on analytic propagation or on a residual
controller that uses spectral concentration only as a hardness feature.

## 24k lattice comparison target

Added a concrete 24,576-sample estimator variant:

```text
submissions/combo_k2_rqmc_24576_shift8675309/estimator.py
```

It uses the same corrected K2 baseline, lattice multiplier `17797`, and shift
offset `8675309`, with `N_SAMPLES = 24576`. The combo coefficient was calibrated
on even `mlp_id` rows from the first 50 Mini MLPs:

```text
lambda = 0.9177969764881879
```

Command:

```powershell
python experiments\run_residual_spectral_scout.py --limit-mlps 50 --n-samples 24576 --candidate-multipliers 17797
```

Output:

```text
results/residual_spectral_scout_limit50_n24576_c1.json
```

Held-out odd-row comparison against the 4096 incumbent:

| Rule | Samples | Lambda | Eval grid final MSE | Eval combo final MSE | Eval adjusted score | Mean combo FLOP fraction |
|---|---:|---:|---:|---:|---:|---:|
| 4k lattice | `4096` | `0.784548` | `5.6167e-06` | `4.5267e-06` | `4.5267e-07` | `0.0696` |
| 24k lattice | `24576` | `0.917797` | `1.0805e-06` | `1.0376e-06` | `4.0263e-07` | `0.3881` |

The 24k lattice is much more accurate and, on this 50-MLP held-out slice, still
slightly better on adjusted score despite crossing above the 10% score floor.
It is a good comparison target, but it is not obviously the final direction:
the compute tax is large, so any analytic improvement that reaches the same MSE
at lower FLOPs should beat it.

Validation:

```powershell
whest validate --estimator submissions\combo_k2_rqmc_24576_shift8675309\estimator.py
```

Result: passed.
