# Antithetic RQMC 24,576

One 12,288-point randomly shifted Gaussian lattice is centrally reflected.
The resulting 24,576 points are propagated with equal weights. No analytic
correction, fitted coefficient, or other mathematical component is included.

## Command actually executed

```powershell
C:\Users\nikso\anaconda3\envs\arcwhitebox\python.exe research\scripts\run_antithetic_rqmc.py --partition development_selection
```

## Result matrix

| method | final MSE | adjusted score | FLOPs | worst-decile MSE | win vs RQMC |
| --- | ---: | ---: | ---: | ---: | ---: |
| ordinary RQMC 24576 | 1.04985e-06 | 4.01161e-07 | 1.03935e+11 | 2.61967e-06 | -- |
| diagonal-K3 residual stack | 2.94144e-06 | 3.14783e-07 | 2.91086e+10 | 8.7337e-06 | -- |
| antithetic RQMC 12288 pairs | 1.33231e-06 | 5.07508e-07 | 1.03611e+11 | 5.13732e-06 | 0.567 |

## Gate

- Adjusted score improves over ordinary RQMC: `False`
- Paired win fraction is at least 0.60: `False`
- Worst-decile MSE is not worse: `False`
- Overall gate: `FAIL`

## Decision

Reject this configuration. It changed mean final MSE by +26.90%, median final
MSE by -13.21%, and worst-decile final MSE by +96.11% versus ordinary RQMC.
It won 0.567 of paired MLPs, but its large outliers made both mean MSE and
adjusted score worse.

The frozen-validation gate failed. Validation was not evaluated and no
submission package was created.
