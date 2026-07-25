# Sigma/cubature benchmark

Fixed full-network propagation benchmark for the higher-order unscented/sigma-point idea.

## Command actually executed

```powershell
research\scripts\run_sigma_point_benchmark.py --partition validation --methods rqmc_513 signed_hadamard_cubature_4096 signed_hadamard_cubature_6144
```

## Result matrix

| method | mean final MSE | adjusted score | win vs RQMC-513 | gain vs RQMC-513 | FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| signed_hadamard_cubature_6144 | 4.1252e-06 | 4.1252e-07 | 1 | 34.9 | 2.5872e+10 |
| signed_hadamard_cubature_4096 | 1.02314e-05 | 1.02314e-06 | 1 | 14.1 | 1.7248e+10 |
| rqmc_513 | 0.000144149 | 1.44149e-05 | nan | nan | 2.17374e+09 |

## Decision

`signed_hadamard_cubature_6144` beats RQMC-513 on this partition and deserves a validation check before any larger implementation.
