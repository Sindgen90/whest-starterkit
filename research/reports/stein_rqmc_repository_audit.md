# Stein/RQMC repository audit

Date: 2026-07-23  
Branch at audit time: `phase1-corrected-k2`  
Baseline estimator inspected: `submissions/rqmc_lattice_002/estimator.py`

## Existing ordinary RQMC estimator

The repository already contains a submission-safe ordinary randomized quasi-Monte Carlo estimator:

```text
submissions/rqmc_lattice_002/estimator.py
```

Important constants:

```text
N_SAMPLES = 1293
RQMC_SEED_OFFSET = 1_000_003
RQMC_MULTIPLIER = 17_797
```

The estimator imports only:

```python
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP
```

It does not import ordinary NumPy.

## RQMC point generation

Function:

```text
normal_lattice_samples(n_samples, width, seed)
```

Mechanism:

1. Create a rank-1 lattice generator using powers of `RQMC_MULTIPLIER` modulo `n_samples`.
2. Use index coordinates `i + 0.5`.
3. Add one random shift vector sampled from `fnp.random.default_rng(seed + RQMC_SEED_OFFSET)`.
4. Apply modulo one.
5. Clip uniforms to `[1e-7, 1 - 1e-7]`.
6. Map uniforms to standard normal samples with `flops.stats.norm.ppf`.
7. Cast the samples to `fnp.float32`.

The input sample matrix shape is:

```text
(N_SAMPLES, mlp.width)
```

For the Phase 1 public Mini split, `mlp.width = 256` and `mlp.depth = 32`.

## Forward pass and activation collection

The estimator performs a batched forward pass:

```python
x = normal_lattice_samples(N_SAMPLES, mlp.width, mlp.seed)
rows = []
for w in mlp.weights:
    x = fnp.maximum(fnp.matmul(x, w), 0.0)
    rows.append(fnp.mean(x, axis=0))
return fnp.stack(rows, axis=0)
```

The returned prediction shape is:

```text
(mlp.depth, mlp.width)
```

For the challenge public Mini MLPs, this is:

```text
(32, 256)
```

## Seeding and randomization

The ordinary RQMC estimator uses:

```text
mlp.seed + RQMC_SEED_OFFSET
```

for the random shift. New cross-fitted estimators must not split this same randomized lattice into train/eval folds. They must instead create independent RQMC randomizations with distinct seed offsets.

## FLOP/budget accounting

The estimator performs all numerical work inside `predict`, where the official runner wraps prediction in a `flops.BudgetContext`. Therefore lattice generation, inverse CDF, forward passes, and reductions are charged.

The estimator ignores the passed `budget` argument but remains well below the official budget.

Official budget constant used in research scripts:

```text
272_000_000_000 FLOPs per MLP
```

Score floor:

```text
0.1
```

## Data split / holdout

Existing scripts mostly use the public Mini split for development and previous even/odd calibrate/evaluate splits for lambda calibration. For the new Stein/RQMC work, a frozen whole-MLP split manifest was created:

```text
research/configs/stein_holdout_manifest.json
```

Policy:

```text
development: mlp_id 0-59
validation:  mlp_id 60-79
holdout:     mlp_id 80-99
```

Do not edit that manifest after reading holdout results.

## Packaging and validation workflow

Official commands:

```powershell
whest validate --estimator path\to\estimator.py --format plain
whest package --estimator path\to\estimator.py --output path\to\submission.tar.gz --yes --format plain
whest validate-package path\to\submission.tar.gz --format plain
```

Subprocess smoke:

```powershell
whest run --estimator path\to\estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner subprocess --n-mlps 5 --max-threads 1 --format plain
```

On Windows, set:

```powershell
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
```

## Baseline preservation commands executed

Submission import guard:

```powershell
python -m pytest tests\test_submission_imports.py
```

Result:

```text
1 passed
```

Estimator validation:

```powershell
whest validate --estimator submissions\rqmc_lattice_002\estimator.py --format plain
```

Result:

```text
success; Estimator class resolved; setup completed; prediction shape (2, 4); finite values
```

Small development subset subprocess run:

```powershell
whest run --estimator submissions\rqmc_lattice_002\estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v1-phase1 --split mini --runner subprocess --n-mlps 5 --max-threads 1 --format plain
```

Result:

```text
n_mlps: 5
failures: 0
raw final-layer MSE: 2.91e-05
adjusted final-layer score: 2.91e-06
all-layer MSE: 5.29e-05
mean compute utilization: 0.02169212
```

## Files/APIs to reuse

Reuse from current code:

```text
submissions/rqmc_lattice_002/estimator.py
  - rank-1 lattice construction
  - random shift seeding pattern
  - Gaussian inverse-CDF mapping
  - batched ReLU MLP forward loop

experiments/run_baselines.py
  - DATASET, REVISION, BUDGET, SCORE_FLOOR
  - as_mlp(row)
  - aggregate(...)
  - load_estimator(...)

tests/test_submission_imports.py
  - unsupported-import guard for submission-facing estimator.py files
```

New code should avoid changing the existing ordinary RQMC estimator except for explicit bug fixes. Submission variants should live under:

```text
submission_variants/
```

and research infrastructure under:

```text
research/
```
