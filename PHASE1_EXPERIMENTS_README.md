# Phase 1 experiment handoff

Last updated: 2026-07-16  
Working branch: `phase1-corrected-k2`  
Push target: `sindgen/phase1-corrected-k2`  
Repo remote: `https://github.com/Sindgen90/whest-starterkit.git`

This document records the Phase 1 work done so far for the ARC White-Box Estimation Challenge. It is meant as a laptop handoff file: after pulling this branch, it should be clear what the current best submission is, how to rebuild it, what was tried, and what should not be repeated.

## Pulling this work on another machine

From a fresh clone or an existing clone with the `sindgen` remote:

```powershell
git fetch sindgen
git switch phase1-corrected-k2
git pull sindgen phase1-corrected-k2
```

If the branch does not exist locally yet:

```powershell
git fetch sindgen
git switch -c phase1-corrected-k2 --track sindgen/phase1-corrected-k2
```

The packaged `.tar.gz` submission files are ignored by Git via `*.tar.gz`, so they are local build artifacts. Rebuild them with `whest package` after pulling.

## Current best submission

Current best local Phase 1 candidate:

```text
submissions/combo_k2_stein_rqmc_4096_ridge232/estimator.py
```

Local packaged artifact, if present:

```text
submissions/combo_k2_stein_rqmc_4096_ridge232.tar.gz
```

Rebuild it:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
whest validate --estimator submissions\combo_k2_stein_rqmc_4096_ridge232\estimator.py --format plain
whest package --estimator submissions\combo_k2_stein_rqmc_4096_ridge232\estimator.py --output submissions\combo_k2_stein_rqmc_4096_ridge232.tar.gz --yes --format plain
whest validate-package submissions\combo_k2_stein_rqmc_4096_ridge232.tar.gz --format plain
```

Safe backup submission:

```text
submissions/combo_k2_rqmc_4096_shift8675309/estimator.py
```

Rebuild safe backup:

```powershell
whest package --estimator submissions\combo_k2_rqmc_4096_shift8675309\estimator.py --output submissions\combo_k2_rqmc_4096_shift8675309.tar.gz --yes --format plain
whest validate-package submissions\combo_k2_rqmc_4096_shift8675309.tar.gz --format plain
```

Important submission constraint discovered from the real grader: submission code cannot assume `numpy` is installed. A guard test enforces this:

```powershell
python -m pytest tests\test_submission_imports.py
```

## Scoring model

The primary metric is adjusted final-layer MSE:

```text
adjusted_final_layer_score = final_layer_mse * max(0.1, effective_compute / flop_budget)
```

Most current candidates use less than 10% of the FLOP budget, so they are on the `0.1` multiplier floor. In that regime, improving raw final-layer MSE is almost directly improving leaderboard score.

## Environment notes

The local conda env used during these runs was:

```text
arcwhitebox
```

The commands below assume that env is activated. On Windows, the `whest` CLI sometimes crashes on Unicode output unless UTF-8 is forced:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
```

## Results summary

The following table uses local Mini split protocols. “Full mini holdout” means lambda was fit on even `mlp_id` rows and evaluated on odd `mlp_id` rows, 50 held-out MLPs. Result JSONs live in `results/`, but that directory is ignored by Git.

| Candidate | Scope | Adjusted score | Raw final MSE | All-layer MSE | FLOP fraction | Decision |
|---|---:|---:|---:|---:|---:|---|
| corrected K2 | full validation | `2.655e-6` | `2.655e-5` | `2.282e-5` | `0.59%` | baseline |
| K2 + RQMC, `N=1293` | full mini holdout | `9.844e-7` | `9.844e-6` | `1.821e-5` | `2.60%` | superseded |
| K2 + RQMC, `N=4096` | full mini holdout | `4.439e-7` | `4.439e-6` | `1.169e-5` | `6.96%` | strong safe |
| shifted K2 + RQMC, `N=4096` | full mini holdout | `4.358e-7` | `4.358e-6` | `1.224e-5` | `6.96%` | safe backup |
| K2 + Stein CV RQMC | full mini holdout | `4.313e-7` | `4.313e-6` | `1.230e-5` | `7.05%` | current best primary |
| Sobol digital shift | full mini holdout | `4.517e-7` | `4.517e-6` | `8.659e-6` | `6.96%` | best all-layer, not primary |
| LHS Gaussian | 20-MLP scout | `6.880e-7` | `6.880e-6` | `1.278e-5` | `6.96%` | stopped |
| multi-lattice, 2 shifts | 20-MLP scout | `5.486e-7` | `5.486e-6` | `1.170e-5` | `6.96%` | stopped |
| multi-lattice, 4 shifts | 20-MLP scout | `5.991e-7` | `5.991e-6` | `1.065e-5` | `6.96%` | stopped |
| multi-lattice, 8 shifts | 20-MLP scout | `6.205e-7` | `6.205e-6` | `1.435e-5` | `6.96%` | stopped |
| lattice multiplier `2891336453` | full mini holdout | `5.677e-7` | `5.677e-6` | `1.201e-5` | `6.96%` | scout did not hold up |

## Approach notes

### 1. Corrected K2 Gaussian propagation

File:

```text
submissions/corrected_k2/estimator.py
```

This propagates a Gaussian approximation through ReLU layers using mean and covariance, then applies a global multiplicative correction:

```text
K2_COEFFICIENT = 0.9918858550287327
```

The correction was fit as a scalar least-squares factor to reduce final-layer MSE. It is stable and cheap, and it became the analytic component in later convex combinations.

Full validation result:

```text
final MSE:    2.655401484e-05
adjusted:     2.655401484e-06
all-layer MSE 2.282063967e-05
```

### 2. Ordinary MC and rank-1 lattice RQMC

Relevant files:

```text
experiments/run_sampling_baselines.py
submissions/rqmc_lattice_002/estimator.py
```

Ordinary MC and randomized rank-1 lattice RQMC were benchmarked against corrected K2. RQMC was immediately better than ordinary MC at similar budget. The standalone RQMC candidate was also used as the sampled component in a convex combination:

```text
prediction = (1 - lambda) * corrected_k2 + lambda * rqmc
```

The first real submission failure was caused by `import numpy as np` in submission code. The grader smoke-test environment rejected it with:

```text
ModuleNotFoundError: No module named 'numpy'
```

That has been fixed in all current submission estimators.

### 3. K2/RQMC convex combination

Relevant files:

```text
experiments/run_convex_combo.py
submissions/combo_k2_rqmc_4096/estimator.py
submissions/combo_k2_rqmc_4096_shift8675309/estimator.py
```

The best sample count was `N=4096`. Larger `N=5632` was worse:

| Samples | Lambda | Adjusted score | Raw final MSE | Decision |
|---:|---:|---:|---:|---|
| `1293` | `0.509384` | `9.844e-7` | `9.844e-6` | superseded |
| `2048` | `0.615073` | `9.788e-7` | `9.788e-6` | no material gain |
| `3072` | `0.748026` | `7.729e-7` | `7.729e-6` | better, not best |
| `4096` | `0.751711` | `4.439e-7` | `4.439e-6` | best sample-count run |
| `5632` | `0.694153` | `7.954e-7` | `7.954e-6` | rejected |

Then shift and multiplier sweeps were run. Changing the random shift offset from `1_000_003` to `8_675_309` gave a small improvement:

```text
lambda:      0.803765718
N:           4096
offset:      8_675_309
multiplier:  17_797
adjusted:    4.357823517e-07
```

### 4. Stein control variates

Relevant files:

```text
experiments/run_stein_cv_combo.py
submissions/combo_k2_stein_rqmc_4096_ridge232/estimator.py
```

This is the current best local primary-score candidate.

Construction:

- Start from the shifted `N=4096` RQMC samples.
- Use zero-mean Gaussian Stein features based on first-layer directions:

```text
w·x
(w·x)^2 - ||w||^2
```

- Use 32 first-layer directions, so 64 Stein features.
- Fit the control variate with cross-fitting:
  - fit coefficients on first half, apply to second half;
  - fit on second half, apply to first half;
  - average the two estimates.
- Center features and outputs inside the regression.
- Ridge penalty: `1.0`.
- Apply the CV only to the final layer.
- Recalibrate only the scalar K2/sample lambda.

Frozen constants:

```text
N_SAMPLES = 4096
RQMC_SEED_OFFSET = 8_675_309
RQMC_MULTIPLIER = 17_797
STEIN_FEATURE_COUNT = 32
STEIN_RIDGE = 1.0
LAMBDA = 0.8065181275829477
```

Full mini holdout:

```text
adjusted score: 4.313417763e-07
raw final MSE:  4.313417763e-06
all-layer MSE:  1.230064992e-05
```

This is only about 1% better than shifted RQMC on primary score, so treat it as an experimental submission rather than an overwhelming win.

### 5. Scrambled Sobol / digital-shift Sobol

Relevant file:

```text
experiments/run_grid_combo.py
```

The Sobol experiment used Joe--Kuo direction numbers from:

```text
https://web.maths.unsw.edu.au/~fkuo/sobol/new-joe-kuo-6.21201
```

Local file used during experiments:

```text
C:\tmp\new-joe-kuo-6.21201
```

This was a random digital shift, not full Owen scrambling. It did not beat the Stein candidate on final-layer primary score, but it gave the best all-layer MSE:

```text
adjusted score: 4.517283922e-07
raw final MSE:  4.517283922e-06
all-layer MSE:  8.659478435e-06
```

If the contest ever rewards all-layer MSE more directly, Sobol is worth revisiting.

### 6. LHS, multi-shift lattice, and generator search

Relevant file:

```text
experiments/run_grid_combo.py
```

Tested after Sobol:

- Latin hypercube Gaussian grid;
- multiple independently shifted rank-1 lattices with total `N=4096`;
- additional Korobov-style lattice multipliers.

None beat the current Stein CV candidate. The best generator scout was multiplier `2891336453`, but it failed on full holdout:

```text
20-MLP scout adjusted: 3.862e-07
full holdout adjusted: 5.677e-07
```

Decision: do not package these as submissions.

### 7. Factorized K3 prototype

Relevant files:

```text
experiments/factorized_k3.py
experiments/run_factorized_k3_prefixes.py
tests/test_factorized_k3.py
```

Implemented/reproduced:

- factor representation `T = F(A, B, C)`;
- linear propagation;
- factor concatenation;
- diagonal corrections;
- repeated-index corrections;
- source metadata;
- small-n algebra tests.

Tests passed:

```powershell
python -m pytest tests\test_factorized_k3.py
```

K3 had shallow promise but failed with depth. On 10 Mini MLPs:

| Third scale | Prefix 2 ratio | Prefix 4 ratio | Prefix 8 ratio | Decision |
|---:|---:|---:|---:|---|
| `-0.001` | `0.983` | `0.979` | `1.043` | slight shallow gain, depth worsens |
| `-0.003` | `0.951` | `0.943` | `1.833` | unstable by depth 8 |
| `-0.005` | `0.921` | `0.917` | `3.563` | unstable |
| `-0.010` | `0.852` | `0.893` | `12.011` | catastrophic by depth 8 |

Decision: do not continue K3 until factor cohort instrumentation and finite-memory pruning are implemented. This matches the original hypothesis that generic third-cumulant factors lose useful influence with depth.

## Experiment commands worth keeping

Run K2/RQMC combo:

```powershell
python experiments\run_convex_combo.py --n-samples 4096 --rqmc-seed-offset 8675309
```

Run Stein CV combo:

```powershell
python experiments\run_stein_cv_combo.py --feature-family ridge2 --feature-count 32 --ridge 1 --apply-layers final
```

Run grid combo examples:

```powershell
python experiments\run_grid_combo.py --grid sobol --n-samples 4096 --seed-offset 8675309
python experiments\run_grid_combo.py --grid lhs --limit-mlps 20 --n-samples 4096 --seed-offset 8675309
python experiments\run_grid_combo.py --grid multi_lattice --limit-mlps 20 --n-samples 4096 --seed-offset 8675309 --multiplier 17797 --n-shifts 4
python experiments\run_grid_combo.py --grid lattice --limit-mlps 20 --n-samples 4096 --seed-offset 8675309 --multiplier 2891336453
```

## What not to redo right now

- Do not increase `N` blindly past `4096`; `5632` was worse.
- Do not use NumPy in submission estimators.
- Do not submit the LHS, multi-lattice, or `2891336453` lattice candidates.
- Do not treat the Stein CV as a huge confirmed win; it is a small local primary-score gain.
- Do not invest in meta-learned control variates during Phase 1 unless the Phase 1 rule against sophisticated meta-learning is explicitly relaxed.

## Recommended next step

Submit or keep the Stein CV candidate as the experimental submission, but keep the shifted RQMC candidate as the safe backup.

Next research step, if continuing before the freeze:

1. Run the current Stein CV candidate through the real leaderboard.
2. If it underperforms the shifted RQMC candidate, fall back to `combo_k2_rqmc_4096_shift8675309`.
3. Only then consider a small Stein CV ablation:
   - feature count `16`, `32`, `48`;
   - ridge `0.3`, `1.0`, `3.0`;
   - final-layer only.

Keep this paired and limited; do not tune repeatedly on the same full Mini split.
