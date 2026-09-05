# Structural QMC Suite

Date: 2026-08-02

Branch: `codex/fourier-walsh-4k-experiments`

## Protocol

All stages use the first 50 Mini MLPs, fit correction parameters on even
`mlp_id` rows, and report held-out results on odd rows. Candidate execution
order is balanced within both parity groups. Point-set construction happens
outside scoring contexts. Every candidate uses corrected-K2 classification,
threshold-3 structural tail propagation, and immutable functional array
updates compatible with the grader backend.

Harness and reusable prediction caches:

```text
experiments/run_structural_qmc_suite.py
results/structural_qmc_<stage>_limit50.{json,npz}
```

## Stage 1: Sample count

All candidates use the incumbent Joe--Kuo Sobol directions, XOR digital shift,
antithetic pairing, and point-set seed 0.

| Samples | Lambda | Combo raw MSE | Full adjusted | FLOP-only adjusted | FLOPs | Effective compute |
|---:|---:|---:|---:|---:|---:|---:|
| `16,384` | `0.971356` | `1.0771e-6` | `3.1147e-7` | **`2.7004e-7`** | `68.21B` | `78.83B` |
| `20,480` | `0.959552` | `8.8883e-7` | `3.2026e-7` | `2.7736e-7` | `84.85B` | `98.02B` |
| `24,576` | `0.968709` | `7.4606e-7` | `3.2098e-7` | `2.7816e-7` | `101.50B` | `117.17B` |
| `28,672` | `0.971012` | `6.2219e-7` | **`3.1144e-7`** | `2.7012e-7` | `118.15B` | `136.33B` |
| `32,768` | `0.976871` | **`5.6623e-7`** | `3.2229e-7` | `2.8053e-7` | `134.79B` | `154.93B` |

The full adjusted scores of 16k and 28k differ by only `0.011%`, which is
below residual-time reliability. The FLOP-only comparison slightly favors
16k, and 16k uses `42%` less effective compute. It is the conservative winner
of this stage. Both beat 32k by about `3.4%` on full adjusted score.

The poor 20k and 24k checkpoints show that point-count scaling is not smooth;
the dyadic 16k net is especially efficient. Larger grids remain useful when
raw accuracy matters, but 49k and 32k are not the adjusted-score optimum.

## Stage 2: Multiple digital shifts

All candidates use 32,768 total samples. Multi-component candidates concatenate
independently XOR-shifted antithetic Sobol nets before the same structural
propagation.

| Rule | Lambda | Combo raw MSE | Full adjusted | FLOP-only adjusted | Effective compute |
|---|---:|---:|---:|---:|---:|
| `1 x 32,768` | `0.976871` | **`5.6623e-7`** | **`3.2175e-7`** | **`2.8053e-7`** | `154.91B` |
| `2 x 16,384` | `0.970235` | `7.1526e-7` | `4.0764e-7` | `3.5398e-7` | `155.41B` |
| `4 x 8,192` | `0.975582` | `7.3263e-7` | `4.1740e-7` | `3.6286e-7` | `154.92B` |

Splitting is decisively harmful. One 32k net has about `21%` lower raw MSE
than two 16k nets at identical counted FLOPs. Preserving the larger net's
global balance matters more than averaging independent digital shifts.

## Stage 3: Scrambling

The six fixed 32k rules compare two XOR shifts, two nonsingular binary linear
matrix scrambles, and two deterministic nested Owen-style scrambles. All use
the same Joe--Kuo base directions and identical prediction FLOPs.

### Odd-ID evaluation

| Rule | Lambda | Combo raw MSE | Full adjusted | FLOP-only adjusted |
|---|---:|---:|---:|---:|
| Owen seed 0 | `0.967157` | **`5.1652e-7`** | **`2.9289e-7`** | **`2.5577e-7`** |
| XOR seed 0 | `0.976871` | `5.6623e-7` | `3.2078e-7` | `2.8053e-7` |
| LMS seed 0 | `0.985459` | `6.4629e-7` | `3.6672e-7` | `3.1940e-7` |
| Owen seed 1 | `0.992469` | `6.8492e-7` | `3.8729e-7` | `3.3861e-7` |
| XOR seed 1 | `0.953072` | `7.1078e-7` | `4.0425e-7` | `3.5183e-7` |
| LMS seed 1 | `0.986895` | `7.5837e-7` | `4.2931e-7` | `3.7415e-7` |

Owen seed 0 appears `8.8%` better than incumbent XOR on the odd half, but the
reverse-parity audit does not support promotion:

| Rule | Even-ID cross-fit raw MSE | Even-ID cross-fit adjusted |
|---|---:|---:|
| XOR seed 0 | **`4.1685e-7`** | **`2.3713e-7`** |
| Owen seed 0 | `6.4360e-7` | `3.6663e-7` |

Scramble sensitivity is large and the apparent held-out winner does not
transfer to the other half. Averaging the two cross-fit halves favors XOR seed
0 overall. No new scramble is frozen from this six-rule search.

## Stage 4: Adaptive sample controller

The controller starts from the 16k rule and conditionally selects 28k or 32k.
Rules use calibration-median thresholds on K2 boundary/uncertainty statistics,
K2 final norm, or the observed K2-versus-16k correction. A three-feature vote
combines relative correction, final boundary mass, and final uncertainty. No
rule sees target values at inference.

| Calibration parity | Evaluation half | Best controller | Adjusted score | Best static score |
|---:|---|---|---:|---:|
| even | odd IDs | 16k/28k by final uncertain fraction | `3.0637e-7` | `3.1144e-7` (28k) |
| odd | even IDs | 16k/32k by K2 final norm | `2.4173e-7` | **`2.3804e-7`** (32k) |

The odd-half rule improves its local static baseline by `1.6%`, but the same
uncertainty rule scores `3.2236e-7` under reverse parity and does not transfer.
The reverse-half controller also loses to its best static candidate.

An infeasible target-aware oracle choosing among 16k, 28k, and 32k reaches
`2.5140e-7` on odd IDs and `2.0916e-7` on even IDs. Adaptive selection has
meaningful theoretical upside, but the current K2 and disagreement diagnostics
do not recover it robustly. No controller is promoted.

## Stage 5: Conditional shrinkage

Each cached count and scramble candidate was tested with:

- the existing global K2/QMC lambda;
- four bins in absolute final-layer K2 z-score;
- five bins in signed final-layer z-score;
- four calibration-quantile bins in relative QMC correction magnitude;
- two MLP-level bins in normalized K2/QMC disagreement.

For incumbent XOR seed 0, every conditional method is slightly worse than the
global lambda in at least one parity. On odd IDs, for example, global shrinkage
scores `3.2078e-7`; signed-alpha bins score `3.2080e-7`, relative-correction
bins `3.2085e-7`, and MLP disagreement bins `3.2127e-7`.

Owen seed 0's relative-correction bins improve both parity directions by about
`0.1%` (`2.9263e-7` versus `2.9289e-7` on odd IDs, and `3.6628e-7` versus
`3.6663e-7` on even IDs), but the Owen source remains worse than incumbent XOR
after cross-fitting. No conditional shrinkage method is promoted.

## Final cross-fitted comparison

The final ranking averages the two cross-fitted directions: fit on even IDs and
evaluate odd IDs, then fit on odd IDs and evaluate even IDs. Equal half sizes
make the arithmetic mean an out-of-fold result over all first 50 MLPs.

| Candidate | Cross-fit raw MSE | Cross-fit full adjusted | Cross-fit FLOP-only adjusted |
|---|---:|---:|---:|
| XOR `1 x 32k`, seed 0 | **`4.9154e-7`** | **`2.7895e-7`** | `2.4361e-7` |
| XOR `28k`, seed 0 | `5.6060e-7` | `2.8121e-7` | **`2.4351e-7`** |
| XOR `24k`, seed 0 | `6.9864e-7` | `3.0099e-7` | `2.6062e-7` |
| XOR `16k`, seed 0 | `1.1346e-6` | `3.2898e-7` | `2.8447e-7` |
| Owen `32k`, seed 0 | `5.8006e-7` | `3.2976e-7` | `2.8725e-7` |
| XOR `2 x 16k` | `6.4445e-7` | `3.6742e-7` | `3.1907e-7` |
| XOR `4 x 8k` | `6.4486e-7` | `3.6704e-7` | `3.1968e-7` |

The best controller averages `2.7965e-7`, slightly worse than static 32k. The
best shrinkage entry is `2.7895e-7` and is exactly incumbent XOR seed 0 with
the original global lambda.

The 28k rule wins FLOP-only by only `0.042%`, below practical accounting
reliability, while losing full adjusted score by about `0.8%` and having worse
raw accuracy. The 16k odd-half win does not survive reverse-parity cross-fit.

## Decision

Retain the current remote-safe 32k XOR seed-0 estimator. None of the requested
count, ensemble, scramble, controller, or conditional-shrinkage variants beats
it robustly enough to justify a new submission artifact.

The strongest unresolved opportunity is the oracle controller gap rather than
more samples: target-aware count selection would improve cross-fitted score
substantially, but current K2 boundary and disagreement features do not predict
the correct per-MLP choice reliably.
