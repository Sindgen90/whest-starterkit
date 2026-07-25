# Cumulant mode-rank diagnostic

This tests the Tucker-K3 prerequisite: after removing diagonal/repeated-index protected structure, do third-cumulant contractions live in a low-dimensional mode space?

## Command actually executed

```powershell
research\scripts\run_cumulant_mode_rank_diagnostic.py --partition development --limit 12 --samples 2048 --probes 96 --layers 4 8 16 24 32 --ranks 4 8 16 32 64
```

## Generic all-distinct contraction spectrum

| layer | rank-4 | rank-8 | rank-16 | rank-32 | rank-64 | effective rank | generic/full energy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.280 | 0.439 | 0.639 | 0.833 | 0.966 | 28.17 | 0.477 |
| 8 | 0.556 | 0.696 | 0.822 | 0.924 | 0.987 | 8.27 | 0.325 |
| 16 | 0.832 | 0.893 | 0.940 | 0.977 | 0.997 | 2.45 | 0.335 |
| 24 | 0.908 | 0.941 | 0.969 | 0.989 | 0.999 | 1.53 | 0.403 |
| 32 | 0.944 | 0.964 | 0.982 | 0.995 | 1.000 | 1.26 | 0.466 |

## Decision rule

Tucker-K3 is promising only if rank 16 or 32 captures most generic contraction energy consistently.  If rank-32 is weak or the generic residual is tiny/noisy, do not implement a full Tucker estimator in Phase 1.
