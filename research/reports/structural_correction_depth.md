# Structural Corrections and Tail Depth

All ablations use the exact 32,768-point antithetic Sobol asset and corrected K2. They were evaluated in balanced order on the first 50 Mini MLPs with opposite-parity lambda fitting. No mechanisms are combined in this experiment.

## Existing Depth Ladder

| Candidate | Raw final MSE | FLOPs | Full adjusted | Structural vs dense |
|---|---:|---:|---:|---:|
| `current_last2_t3` | 4.9154e-07 | 1.3491e+11 | 2.7682e-07 | 2.6409e-10 |
| `C_last6_tapered` | 4.9172e-07 | 1.2735e+11 | 2.6385e-07 | 3.3039e-10 |
| `D_last8_conservative` | 4.9178e-07 | 1.2516e+11 | 2.6074e-07 | 3.2067e-10 |
| `E_last8_balanced` | 4.9169e-07 | 1.2475e+11 | 2.6002e-07 | 3.2370e-10 |
| `F_last8_aggressive` | 4.9185e-07 | 1.2435e+11 | 2.5865e-07 | 3.3344e-10 |

## New Isolated Ablations

| Candidate | Mechanism | Raw final MSE | FLOPs | Full adjusted | Flop-only adjusted | Structural vs dense |
|---|---|---:|---:|---:|---:|---:|
| `F_last8_control` | last-eight control | 4.9185e-07 | 1.2435e+11 | 2.5846e-07 | 2.2436e-07 | 3.3344e-10 |
| `G_last8_dead_tail` | dead Gaussian tail mean | 4.9171e-07 | 1.2439e+11 | 2.6946e-07 | 2.2437e-07 | 3.8270e-10 |
| `H_last8_amplification` | multistep amplification thresholds | 4.9168e-07 | 1.2434e+11 | 2.5863e-07 | 2.2429e-07 | 3.3421e-10 |
| `I_last8_tail_risk` | tail probability x outgoing norm | 4.9156e-07 | 1.2435e+11 | 2.5895e-07 | 2.2424e-07 | 2.9585e-10 |
| `J_last10_tapered` | last ten | 4.9190e-07 | 1.2208e+11 | 2.5504e-07 | 2.2020e-07 | 3.3369e-10 |
| `K_last12_tapered` | last twelve | 4.9190e-07 | 1.2060e+11 | 2.5269e-07 | 2.1738e-07 | 3.3380e-10 |

## Paired Against F

Intervals are paired 20,000-draw bootstrap intervals for candidate minus F.

| Candidate | Raw change | Raw delta 95% CI | Adjusted change | Adjusted delta 95% CI | FLOP change |
|---|---:|---:|---:|---:|---:|
| `G_last8_dead_tail` | -0.030% | [-5.7317e-10, 2.7003e-10] | +4.253% | [9.2214e-09, 1.3049e-08] | +0.034% |
| `H_last8_amplification` | -0.035% | [-3.7962e-10, -7.8943e-12] | +0.066% | [-9.6264e-10, 1.2053e-09] | -0.005% |
| `I_last8_tail_risk` | -0.060% | [-7.8364e-10, 4.0368e-11] | +0.189% | [-5.9376e-10, 1.4557e-09] | +0.001% |
| `J_last10_tapered` | +0.009% | [5.3969e-12, 1.0302e-10] | -1.323% | [-4.7801e-09, -2.2930e-09] | -1.823% |
| `K_last12_tapered` | +0.010% | [6.8880e-12, 1.0810e-10] | -2.234% | [-7.7345e-09, -4.1023e-09] | -3.015% |

## Parity Check

| Candidate | Parity-0 raw | Parity-0 adjusted | Parity-1 raw | Parity-1 adjusted |
|---|---:|---:|---:|---:|
| `F_last8_control` | 5.6613e-07 | 2.9768e-07 | 4.1758e-07 | 2.1925e-07 |
| `G_last8_dead_tail` | 5.6647e-07 | 3.1007e-07 | 4.1694e-07 | 2.2884e-07 |
| `H_last8_amplification` | 5.6599e-07 | 2.9760e-07 | 4.1737e-07 | 2.1967e-07 |
| `I_last8_tail_risk` | 5.6553e-07 | 2.9756e-07 | 4.1758e-07 | 2.2035e-07 |
| `J_last10_tapered` | 5.6619e-07 | 2.9293e-07 | 4.1760e-07 | 2.1715e-07 |
| `K_last12_tapered` | 5.6619e-07 | 2.9039e-07 | 4.1762e-07 | 2.1499e-07 |

## Findings

1. **Dead-tail correction does not hold up.** It slightly improves aggregate raw MSE, but the parity directions disagree, structural-versus-dense error rises, and carrying the constants increases residual compute enough to lose adjusted score.
2. **Multistep amplification is neutral in this form.** It makes a tiny raw improvement but does not improve full adjusted score.
3. **Tail-risk reallocation is the useful accuracy signal.** At effectively identical counted FLOPs it lowers raw MSE and reduces structural-versus-dense final MSE from `3.3344e-10` to `2.9585e-10`. Its current NumPy controller overhead makes full adjusted score slightly worse; this mechanism is worth retaining for a later optimized combination.
4. **Greater depth continues to pay.** J and K preserve raw accuracy while reducing FLOPs. K is the strongest standalone result and wins adjusted score in both parity directions.

K uses thresholds `4.75, 4.5, 4.25, 4.0, 3.75, 3.6, 3.5, 3.4, 3.3, 3.2, 3.0, 3.0` on layers 21-32. Its empirical wrong-sign fractions at layers 21-24 are approximately `2.6e-8`, `6e-9`, `1.5e-7`, and `7.9e-7`; the newly introduced per-layer structural MSE remains below `2.2e-13` through layer 24.

No correction or adaptive rule is combined with J/K here. That remains a separate next experiment, as requested.

## Standalone Depth Artifacts

Both depth-only implementations reproduce their experiment caches exactly on a
real depth-32/width-256 MLP, pass `whest validate`, and pass the immutable-array
regression test.

Last ten:

```text
submissions/structural_tail10_sobol_32768_remote_safe.tar.gz
SHA256: A67BA0EAC795054CB22FB41A0CB2024677F45F546A7CDAF1EEA024292F05BBC1
```

Last twelve:

```text
submissions/structural_tail12_sobol_32768_remote_safe.tar.gz
SHA256: 9E152AE922A44F0D0AD93393E0364E96F11BA2FA638CBBCDA21AAACFC37B84E5
```

Each archive contains `estimator.py`, `samples.npz`, and `manifest.json`.
