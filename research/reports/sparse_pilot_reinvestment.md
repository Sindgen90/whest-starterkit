# Sparse, Pilot, and QMC Reinvestment Experiments

## Objective

Test the public structural-sampling ideas that remained compatible with the
challenge framing:

1. classify and prune almost-dead neurons;
2. verify aggressive dead classifications with a small pilot;
3. exploit exact activation sparsity inside sampled matrix products;
4. reinvest saved compute in a larger antithetic Sobol point set.

All serious comparisons used the same corrected-K2 propagation, fixed scrambled
Sobol construction, balanced candidate order, and parity cross-fitted K2/QMC
blend. Two-MLP runs were correctness and accounting smoke tests; selection used
the 50-MLP mini split.

## Structural Winner

The winning candidate uses 32,768 antithetic Sobol samples through the last 16
layers. Layers 17-30 classify only dead neurons and ReLU every retained neuron.
The final two layers retain symmetric dead/on/kink classification and collapse
always-on paths linearly.

The selected dead thresholds for layers 17-32 are:

```text
5.25, 5.00, 4.75, 4.50, 4.25, 4.00, 3.75, 3.50,
3.25, 3.10, 3.00, 2.90, 2.80, 2.70, 3.00, 3.00
```

Here a middle-layer neuron is pruned when its corrected-K2 Gaussian preactivation
z-score satisfies `alpha <= -threshold`. This is weight-derived structural
pruning, not classification from the sampled activations.

### 50-MLP kernel comparison

| Candidate | Raw final MSE | Counted FLOPs | Effective compute | Adjusted score |
|---|---:|---:|---:|---:|
| Q symmetric after layer 20 | 4.919014e-7 | 117.870B | 138.530B | 2.490678e-7 |
| dead-only, Q thresholds | 4.917974e-7 | 117.885B | 137.092B | 2.465599e-7 |
| **dead-only, m050 thresholds** | **4.921611e-7** | **115.541B** | **134.411B** | **2.419299e-7** |
| m050 plus 512 pilot rows | 4.924463e-7 | 115.971B | 138.807B | 2.498359e-7 |
| m075 plus 1,024 pilot rows | 4.921878e-7 | 114.996B | 137.901B | 2.480430e-7 |

Against Q, m050 changed raw MSE by `+0.0528%` but improved adjusted score by
`2.8659%`. The paired bootstrap 95% interval for the adjusted-score delta was
`[-8.813e-9, -5.623e-9]`.

Pilot verification rescued some analytically dead neurons, especially in later
tail layers, but repeated remote-to-host decisions erased the FLOP saving. It
also failed to improve mean raw MSE on 50 MLPs.

## Exact Sparsity

Two exact sparse products were prototyped:

- gathered support buckets for rows with 0, 1, 2, 4, or 8 cold activations;
- a simpler zero-block product that skipped only rows whose cold prefix was
  entirely zero.

On the two-MLP smoke, gathered bucketing reduced counted FLOPs from about 117.1B
to 103.2B, but dynamic gathers and synchronization increased effective compute
to about 292.4B. Zero-block reduced counted FLOPs only to 116.1B and still raised
effective compute to about 258.9B. Both were rejected.

The useful lesson is execution-model specific: sampled sparsity is real, but
dynamic row sorting, indexing, and host synchronization cost more than the dense
arithmetic they avoid under the current flopscope runtime.

## Sobol Reinvestment

### Broad sweep, 50 MLPs

| Samples | Raw final MSE | Effective compute | Adjusted score |
|---:|---:|---:|---:|
| **32,768** | **4.921611e-7** | **133.847B** | **2.410309e-7** |
| 49,152 | 3.711616e-7 | 199.604B | 2.706930e-7 |
| 65,536 | 3.066374e-7 | 265.535B | 2.964827e-7 |

The larger grids worked statistically: 49k reduced raw MSE by `24.59%` and 65k
by `37.70%`, winning raw MSE on 82% and 84% of MLPs. Their compute growth was
larger than the accuracy gain, worsening adjusted score by `12.31%` and `23.01%`.
The 84,992-point diagnostic used about 295B counted FLOPs on two MLPs and was not
taken to the full run.

### Near-floor sweep, 50 MLPs

| Samples | Raw final MSE | Effective compute | Adjusted score |
|---:|---:|---:|---:|
| **32,768** | **4.921611e-7** | **133.966B** | **2.411768e-7** |
| 33,280 | 4.981724e-7 | 136.190B | 2.484708e-7 |
| 33,792 | 5.030972e-7 | 138.505B | 2.549923e-7 |
| 34,816 | 4.877423e-7 | 142.264B | 2.533689e-7 |

Small additions near the half-budget scoring floor were not robust. The 34,816
set improved raw MSE by `0.90%` but worsened adjusted score by `5.06%`; the two
smaller additions worsened both aggregate raw MSE and adjusted score.

## Submission Artifact

The selected estimator is in
`submissions/structural_tail16_dead_m050_sobol_32768/`. It uses blend coefficient
`0.9787479459257391` and returns corrected-K2 intermediate rows plus the sampled
structural final row.

The actual artifact passed validation and a 50-MLP local runner test with zero
failures:

- raw final-layer MSE: `4.92e-7`;
- all-layer MSE: `2.24e-5`;
- mean counted FLOPs: about `115.6B`;
- local mean effective compute: about `153.2B`;
- local adjusted score: `2.76e-7`.

Residual timing in the standalone CLI run was higher than in the balanced scout,
so the paired experiment is the better measure of the algorithmic delta. The
artifact run establishes correctness and end-to-end grader compatibility.

## Conclusion

Keep the 32,768-sample m050 dead-only estimator. Do not include pilot refinement,
dynamic sparse kernels, or a larger Sobol grid in the current submission. A
future sparsity attempt should use a static block structure derivable from
moments without sample-dependent host decisions.
