# Sigma/cubature benchmark

Fixed full-network propagation benchmark for the higher-order unscented/sigma-point idea.

## Command actually executed

```powershell
research\scripts\run_sigma_point_benchmark.py --partition development --methods rqmc_513 axis_cubature_512 hadamard_cubature_512 signed_hadamard_cubature_1024 signed_hadamard_cubature_2048 signed_hadamard_cubature_4096 signed_hadamard_cubature_6144
```

## Result matrix

| method | mean final MSE | adjusted score | win vs RQMC-513 | gain vs RQMC-513 | FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| signed_hadamard_cubature_6144 | 3.88422e-06 | 3.88422e-07 | 1 | 24 | 2.5872e+10 |
| signed_hadamard_cubature_4096 | 5.81946e-06 | 5.81946e-07 | 1 | 16 | 1.7248e+10 |
| signed_hadamard_cubature_2048 | 1.0111e-05 | 1.0111e-06 | 1 | 9.23 | 8.62401e+09 |
| signed_hadamard_cubature_1024 | 2.52911e-05 | 2.52911e-06 | 0.917 | 3.69 | 4.312e+09 |
| axis_cubature_512 | 3.88179e-05 | 3.88179e-06 | 0.783 | 2.4 | 2.156e+09 |
| hadamard_cubature_512 | 3.94255e-05 | 3.94255e-06 | 0.767 | 2.37 | 2.156e+09 |
| rqmc_513 | 9.33008e-05 | 9.33008e-06 | nan | nan | 2.17374e+09 |

## Decision

`signed_hadamard_cubature_6144` beats RQMC-513 on this partition and deserves a validation check before any larger implementation.
