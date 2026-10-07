# Upload candidate: runtime-optimized calibrated covariance baseline

Date: 1 October 2026.

**Upload artifact:** `../../submissions/phase2_best_20261001.tar.gz`.
Final archive SHA-256:
`0a3fa420f2ac981333d4be58299bc80f38b8a2d0a3ad1982ce8d82dcbfac1d18`.
It contains `estimator.py`, the official manifest, MIT license, and provenance.
The archive's estimator bytes match the tested source and all manifest hashes
were checked. The official validator accepted the final archive with no issues.

The skewness/kurtosis experiments did not produce a better eligible estimator.
The strongest practical baseline remains the published calibrated covariance
method. A fresh full-width subprocess check of its original source exceeded
the local residual cap: 0.452944 seconds versus 0.400 seconds, with no numerical
exception. This motivated a small algebraic/runtime derivative.

## Changes

The original quadrature contribution is
`(rho*weight) * (rho-rho*node) * density`.
Factor it as `rho^2 * weight*(1-node) * density`, accumulating the weighted
densities and applying the common factor outside the loop. Evaluate the
density using division instead of a reciprocal power, and skip the final
covariance update because no subsequent layer consumes it. Remove an unused
prediction RNG initialization.

The eight nodes, calibration coefficients, Gaussian closure, and returned
means at every layer are preserved. There is no fitting, new dataset-derived
asset, oracle moment, or reference target in the candidate. Different
floating-point operation ordering causes very small numerical differences.
The upstream MIT license and attribution are retained in `venture_runtime/`.

## Paired eight-network check

Networks 0-3 and 8-11, four backend threads, unchanged public targets:

| Method | Average raw MSE | Nominal adjusted MSE | Average residual seconds | Local cap failures |
|---|---:|---:|---:|---:|
| Original | 1.417285136e-6 | 1.417285136e-7 | 0.46603 | 7/8 |
| Runtime derivative | 1.417351265e-6 | 1.417351265e-7 | 0.37401 | **0/8** |

The approximately 0.0047% raw-error difference is immaterial rounding drift,
not an accuracy improvement. Largest prediction difference across all layers
and tested networks: 4.291535e-6. FLOPs fall from 56,786,552,287 to
55,281,615,328, both below the score multiplier floor. This change addresses
execution reliability; it does not promise a leaderboard accuracy gain.

The candidate's residual times ranged from 0.30049 to 0.39444 seconds. The
maximum remains close to the cap, so local success does not guarantee remote
timing. The earlier original baseline had passed these caps on 26 September;
today's failures are recorded rather than removed from the comparison.

Four baseline/runtime numerical tests passed, including an independent
arccosine covariance identity and nonstandard-depth compatibility. Official
`whest validate` also passed. The full-width subprocess check is recorded in
`results/runtime_subprocess_20261001.json`; its 100-sample smoke target is used
only for execution validation, not an accuracy estimate.

The official full-width subprocess recheck passed with zero failures:
4.94728 seconds predict wall time, 0.395442 seconds residual time, and
55,281,615,328 FLOPs. Residual-time headroom remains narrow. This is the
strongest practical upload candidate from the current tested methods, with
baseline-level accuracy and improved local execution reliability.

## Reproduction

```powershell
& .venv-phase2/Scripts/python.exe research/phase2/benchmark_runtime.py --arrows ../tmp/phase2_mini_first8.arrow ../tmp/phase2_mini_validation8.arrow --output research/phase2/results/runtime_baseline_20261001.json
& .venv-phase2/Scripts/python.exe -m pytest research/phase2/test_runtime_baseline.py research/phase2/test_public_baseline.py -q
$env:PYTHONIOENCODING='utf-8'
& .venv-phase2/Scripts/whest.exe validate --estimator research/phase2/venture_runtime/estimator.py --format json
& .venv-phase2/Scripts/whest.exe run --estimator research/phase2/venture_runtime/estimator.py --runner subprocess --n-mlps 1 --n-samples 100 --seed 20260926 --max-threads 4 --format json
& .venv-phase2/Scripts/whest.exe package --estimator research/phase2/venture_runtime/estimator.py --output submissions/phase2_best_20261001.tar.gz --yes --format json
& .venv-phase2/Scripts/python.exe research/phase2/finalize_runtime_package.py
& .venv-phase2/Scripts/whest.exe validate-package submissions/phase2_best_20261001.tar.gz --format json
```
