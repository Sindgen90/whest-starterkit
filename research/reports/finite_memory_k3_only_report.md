# Finite-memory K3-only branch report

Branch: `phase1-k3-only`

This branch revisits K3 without K4.  The goal was to test whether factorized third-cumulant propagation can help once generic factors are given a finite memory and once deep K3 corrections are prevented from destabilizing the final layers.

## Implementation

Updated `experiments/factorized_k3.py`:

- added `generic_horizon`;
- added factor aging;
- converted propagated layer-local diagonal/repeated corrections into generic factors;
- preserved current-layer diagonal and repeated-index corrections as protected;
- added `max_k3_layer` cutoff, allowing K3 to influence only early layers and then clearing factor state.

Updated `experiments/run_factorized_k3_prefixes.py`:

- added `--generic-horizon`;
- added `--max-k3-layer`;
- included both values in result filenames and configurations.

Added `research/scripts/run_k3_cubature_combo.py`:

- calibrates a finite-memory K3 + signed-Hadamard-cubature convex blend;
- records raw MSE, adjusted score, FLOPs, and win fraction;
- uses the existing frozen development/validation manifest split.

Updated `tests/test_factorized_k3.py`:

- verifies propagated protected factors become generic;
- verifies finite-memory filtering drops only stale generic factors and preserves protected factors.

## Commands executed

```powershell
git switch -c phase1-k3-only
```

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe -m pytest tests\test_factorized_k3.py
```

```powershell
foreach ($h in 1,2,4,8) { C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe experiments\run_factorized_k3_prefixes.py --limit-mlps 10 --prefixes 2 4 8 16 --third-scale -0.001 --generic-horizon $h }
```

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe experiments\run_factorized_k3_prefixes.py --limit-mlps 10 --prefixes 2 4 8 16 24 32 --third-scale=-0.00003 --generic-horizon 2
```

```powershell
foreach ($scale in -0.0001,-0.0003) { C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe experiments\run_factorized_k3_prefixes.py --limit-mlps 10 --prefixes 2 4 8 16 24 32 --third-scale $scale --generic-horizon 2 }
```

```powershell
foreach ($case in '0.0003,8','0.0003,16','0.001,8','0.001,16') { $parts = $case.Split(','); $scale = '-' + $parts[0]; $cutoff = $parts[1]; C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe experiments\run_factorized_k3_prefixes.py --limit-mlps 10 --prefixes 2 4 8 16 24 32 --third-scale=$scale --generic-horizon 2 --max-k3-layer $cutoff }
```

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe research\scripts\run_k3_cubature_combo.py --third-scale=-0.001 --generic-horizon 2 --max-k3-layer 8
```

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe research\scripts\run_k3_cubature_combo.py --third-scale=-0.001 --generic-horizon 2 --max-k3-layer 4
```

## K3-only horizon sweep

Development/mini first 10 MLPs, fixed `third_scale=-0.001`.

| H | prefix 2 ratio vs K2 | prefix 4 | prefix 8 | prefix 16 | win frac @8 | last factor count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.983 | 0.967 | 8.424 | 5.06e7 | 0% | 1024 |
| 2 | 0.983 | 0.983 | 0.999 | 19.48 | 60% | 1536 |
| 4 | 0.983 | 0.979 | 1.026 | 3505 | 30% | 2560 |
| 8 | 0.983 | 0.979 | 1.043 | 10406 | 20% | 4608 |

Decision: `H=2` was the only remotely stable horizon, but unbounded K3 was still catastrophic by depth 16.

## K3-only scale sweep

Development/mini first 10 MLPs, fixed `H=2`.

| third_scale | p2 ratio | p4 ratio | p8 ratio | p16 ratio | p24 ratio | p32 ratio | depth-32 win |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -3e-5 | 0.9995 | 0.9995 | 0.9997 | 1.014 | 71.7 | 2.20e5 | 0% |
| -1e-4 | 0.9983 | 0.9982 | 0.9990 | 1.177 | 815 | 1.23e6 | 0% |
| -3e-4 | 0.9950 | 0.9946 | 0.9976 | 2.649 | 7281 | 6.62e6 | 0% |
| -1e-3 | 0.9834 | 0.9828 | 0.9990 | 19.48 | — | — | — |

Decision: reducing scale does not fix deep instability.  K3 has a shallow signal, but stale/deep factors become actively harmful.

## K3 cutoff sweep

Development/mini first 10 MLPs, fixed `H=2`.

| K3 config | p8 ratio | p16 ratio | p32 ratio | p32 win | FLOPs / MLP |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scale=-3e-4, cutoff=8` | 0.9976 | 0.9971 | 0.9970 | 90% | 7.89e9 |
| `scale=-3e-4, cutoff=16` | 0.9976 | 2.649 | 1.306 | 10% | 1.60e10 |
| `scale=-1e-3, cutoff=8` | 0.9990 | 0.9930 | 0.9908 | 90% | 7.89e9 |
| `scale=-1e-3, cutoff=16` | 0.9990 | 19.48 | 4.70 | 0% | 1.60e10 |

Decision: cutoff at 8 makes K3 non-catastrophic and slightly improves K2 at full depth.  Cutoff at 16 is still too late.

## K3 + signed Hadamard cubature

The current best submission candidate before this branch is corrected-K2 + signed-Hadamard cubature:

| method | validation final MSE | adjusted score | FLOPs |
| --- | ---: | ---: | ---: |
| K2 + signed Hadamard cubature 6144 | 3.7976e-6 | 3.8380e-7 | 2.749e10 |

Finite-memory K3 was then tested as the analytic component in the same style of convex blend.

### `H=2, scale=-1e-3, cutoff=8`

| split | method | final MSE | adjusted score | FLOPs | win vs cubature |
| --- | --- | ---: | ---: | ---: | ---: |
| development selection | cubature | 3.6341e-6 | 3.6341e-7 | 2.587e10 | — |
| development selection | K3+cubature | 3.0590e-6 | 3.7968e-7 | 3.376e10 | 83.3% |
| validation | cubature | 4.1252e-6 | 4.1252e-7 | 2.587e10 | — |
| validation | K3+cubature | 3.7853e-6 | 4.6982e-7 | 3.376e10 | 60.0% |

### `H=2, scale=-1e-3, cutoff=4`

| split | method | final MSE | adjusted score | FLOPs | win vs cubature |
| --- | --- | ---: | ---: | ---: | ---: |
| development selection | cubature | 3.6341e-6 | 3.6341e-7 | 2.587e10 | — |
| development selection | K3+cubature | 3.0603e-6 | 3.3435e-7 | 2.972e10 | 80.0% |
| validation | cubature | 4.1252e-6 | 4.1252e-7 | 2.587e10 | — |
| validation | K3+cubature | 3.8036e-6 | 4.1556e-7 | 2.972e10 | 60.0% |

## Decision

Stop K3 development for Phase 1 unless a cheaper representation appears.

K3 contains useful information: the early cutoff variants consistently improve raw MSE versus cubature and corrected K2.  But the factorized implementation is too expensive relative to the competition score formula.  Once paired with cubature, even the cheaper cutoff-4 version loses adjusted score on validation.

Recommended submission remains:

`submissions/combo_k2_signed_hadamard_cubature_6144.tar.gz`

