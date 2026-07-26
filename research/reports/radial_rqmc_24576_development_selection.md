# Radial Rao-Blackwellized RQMC 24,576

The shifted Gaussian lattice is normalized onto the unit sphere. Because
the supplied zero-bias ReLU MLP is positively homogeneous, the independent
Gaussian radius is integrated exactly by multiplying every layer mean by
`E[||X||]`. No other mathematical component is changed.

## Command actually executed

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe research\scripts\run_radial_rqmc_24576.py --partition development_selection
```

## Result matrix

| method | final MSE | adjusted score | FLOPs | worst-decile MSE | win vs RQMC |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary RQMC 24576 | 1.04985e-06 | 4.01161e-07 | 1.03935e+11 | 2.61967e-06 | -- |
| diagonal-K3 residual stack | 2.94144e-06 | 3.14783e-07 | 2.91086e+10 | 8.7337e-06 | -- |
| radial RQMC 24576 | 1.06192e-06 | 4.05848e-07 | 1.03954e+11 | 2.61766e-06 | 0.467 |

## Gate

- Adjusted score improves over ordinary RQMC: `False`
- Paired win fraction is at least 0.60: `False`
- Worst-decile MSE is not worse: `True`
- Overall gate: `FAIL`

## Development decision

Relative to ordinary RQMC, this changed mean final MSE by +1.15%, median by -3.52%, and worst-decile MSE by -0.08%.
The validation gate `failed`. Validation was not evaluated and no package was created.
