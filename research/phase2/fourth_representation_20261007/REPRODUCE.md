# Reproduction and raw evidence

Run from `C:\Users\nikso\Desktop\Books\AI papers\ARC_Challenge\ARC_Whitebox_Challenge\whest-starterkit`, using `.venv-phase2\Scripts\python.exe` (whestbench 0.16.1 / flopscope 0.12.1). All numerical benchmarks were serial with four threads. The scripts reject existing main result paths; reproduce in a new copy of the study directory or choose fresh `--output`/gate prefixes. Do not erase this evidence.

## Inputs and seeds

- Small fixed weights: official `sample_mlp(width,16,fnp.random.default_rng(seed),seed=seed)`, width16 seeds82071/82072, width32 seeds82081/82082.
- Small references: four independent 32768-input streams, NumPy SeedSequence `[20261007, network_seed, stream, 913]`, float32 Gaussian inputs/network arithmetic, float64 moment accumulation. Full unbiased third/fourth k-statistics at postlayers0/1/7/8/14/15. All-layer means/C retained. No reference tensors enter inference.
- Partitions: contiguous; fixed random seed20261007+network seed; correlation greedy pivot-neighbors from baseline's first propagated C only. Rule selected once by the prespecified common-state criterion. Clusters never use oracle cumulants. All partitions fixed across depth.
- Full development: `../tmp/phase2_mini_first8.arrow`, public IDs0/1, baked 1e9-draw means; all methods use identical weights/targets.
- Official execution: unchanged CLI, subprocess, seed20261071, n_mlps2, n_samples100, max_threads4, default caps. These targets are too noisy for competitive accuracy ranking.
- Exploratory full-C references: public ID0 only, four independent8192-input streams, SeedSequence `[20261007,0,stream,1927]`, batch1024, postcheckpoints0/7/14; sample covariance uses n−1. Explicitly separate from two-network final-mean ranking.
- Fresh seeds20261171/20261172 and random-partition confirmation were reserved by the protocol for promising candidates; not run if the frozen development criterion fails.

## Commands used

```powershell
.venv-phase2/Scripts/python.exe -m pytest -q research/phase2/fourth_representation_20261007/test_equations.py
.venv-phase2/Scripts/python.exe research/phase2/fourth_representation_20261007/build_estimators.py
.venv-phase2/Scripts/python.exe -W ignore research/phase2/fourth_representation_20261007/oracle_study.py
.venv-phase2/Scripts/python.exe -W ignore research/phase2/fourth_representation_20261007/recurrence_study.py
.venv-phase2/Scripts/python.exe -W ignore research/phase2/fourth_representation_20261007/recurrence_study.py --output recurrence_results_v2.json --reuse-third
.venv-phase2/Scripts/python.exe -W ignore research/phase2/fourth_representation_20261007/profile_kernels.py
.venv-phase2/Scripts/python.exe -W ignore research/phase2/fourth_representation_20261007/profile_kernels.py --output kernel_profiles_v2.json --transport-only
.venv-phase2/Scripts/python.exe -W ignore research/phase2/fourth_representation_20261007/full_width.py --names baseline_snapshot qr block8 block16 qblock8 qblock16 qrblock8
.venv-phase2/Scripts/python.exe research/phase2/fourth_representation_20261007/build_selective.py
.venv-phase2/Scripts/python.exe research/phase2/fourth_representation_20261007/build_compat.py
.venv-phase2/Scripts/python.exe -W ignore research/phase2/fourth_representation_20261007/full_width.py --names baseline_snapshot selective_block8 --output selective_full_width_results.json
.venv-phase2/Scripts/python.exe -W ignore research/phase2/fourth_representation_20261007/gate_all.py --names baseline_snapshot selective_block8 qr block8 block16 qblock8 qblock16 qrblock8
.venv-phase2/Scripts/python.exe -W ignore research/phase2/fourth_representation_20261007/oracle_study.py --output oracle_results_v2.json
.venv-phase2/Scripts/python.exe -W ignore research/phase2/fourth_representation_20261007/exact_controls.py
.venv-phase2/Scripts/python.exe -W ignore research/phase2/fourth_representation_20261007/local_nonlinear.py
.venv-phase2/Scripts/python.exe -W ignore research/phase2/fourth_representation_20261007/timing_pair.py
.venv-phase2/Scripts/python.exe -W ignore research/phase2/fourth_representation_20261007/clipping_diagnostic.py
.venv-phase2/Scripts/python.exe -W ignore research/phase2/fourth_representation_20261007/large_covariance.py
.venv-phase2/Scripts/python.exe research/phase2/fourth_representation_20261007/make_report.py
```

Build order matters for the historical versions: the initial generic kernels used four einsums for cross-block transport, followed by an algebraically equivalent batched-matrix implementation. Source history preserves both. Final `build_estimators.py` uses the current kernel/closure fragments and selected correlation rule; primary small arithmetic is unchanged except rounding in equivalent matrix products. Float32 WᵀCW is explicitly symmetrized. `terminal_check.json` compares specialized terminal contraction against full diagnostic output.

`oracle_results_v2.json` adds four-stream jackknife intervals to the same fixed oracle experiment, without resampling/reselecting partitions. `recurrence_results_v2.json` corrects covariance symmetrization and includes the complete separate repair ablation; first-pass failures remain in `recurrence_results.json`. The explicitly offline missing-third experiment remains in that first report and does not use the modified covariance-affine implementation. `source_history/` contains source versions, not submission candidates. `offline_observers/` contains state-observation copies used only for covariance diagnostics.

`exact_transport_controls.json` tests QR/E8/F8 with complete affine transport on the four small networks; no new full-width selection uses those controls. `local_nonlinear_results.json` compares generation of post-ReLU marginal/Q/R cumulants from identical oracle lower-order inputs, distinct from accumulated propagation. `compatibility.json`, `compat_full_width_results.json` and `selective_block8_compat_gate*` document a width-four validator fix with identical width-1024 predictions/FLOPs. `timing_pair_results.json` compares official local/subprocess runners at identical seeds/spec/source hashes. Normalized oracle risks use pooled reference scales and remain ratios of estimates; only the linear cross-stream numerators have the stated unbiasedness property.

`clipping_diagnostic.json` is a post-hoc offline diagnosis only: an explicit positive-support finite distribution tests translation/recentering consistency, then a complete-fourth oracle ablation removes only raw-M3/M4 clipping while holding corrected mean/M2 fixed. It does not change a submission, tune a coefficient, or provide recurrence/held-out evidence. Final equation tests include this finite-distribution counterexample (12 passing tests). `make_report.py` generates this diagnostic if missing; do not rerun its standalone command over an existing result.

## Official gate and packaging

`gate_all.py` invokes the preserved `research/phase2/benchmark_joint_official.py`. That helper runs:

```powershell
.venv-phase2/Scripts/whest.exe run --estimator <absolute-estimator.py> --runner subprocess --n-mlps 2 --n-samples 100 --seed 20261071 --max-threads 4 --format json
```

Full stdout/stderr, commands, source SHA-256 and failures remain under `*_gate*`. The additional memory observer reads OS process counters; it does not change the evaluator, metering, cap or timing subtraction. No candidate passes solely because an in-process screen is fast.

Only a passing full-width family may run `package_candidate.py <name>`; it asserts matching frozen gate/source hashes, then uses the unchanged official `validate`, `package --yes`, and `validate-package` commands. Archive members are rehashed against the source. Names are dated `phase2_EXPERIMENTAL_<name>_20261007.tar.gz`; none is promoted without held-out adjusted evidence. No upload command is used.

## Evidence map

`AUDIT.md`: derivations and omissions. `PROTOCOL.md`: predeclared criteria. `environment.json`: installed versions/default spec. `preservation.json`: 528 pre-study file hashes. `preservation_verification.json`: final comparison. `inventory.json`: final new-source/raw-result hashes. `experiment_matrix.csv`: complete status matrix. `COMPARISON.md`: measured conclusions. `reference_n*_seed*.npz`, `recurrence_n*_seed*.npz`, `full_width_results_mlp*.npz`, `large_cov_reference_id0.npz`, `large_cov_states_*`: offline data/predictions only.

Peak working sets from shared in-process screens are cumulative, not per-method allocation maxima. Fresh bounded profilers and official per-run process-tree observations provide separate measurements. Shape/copy costs and individual metered calls are retained in kernel profiles. Covariance reference sampling uncertainty and across-network variation must be interpreted separately.
