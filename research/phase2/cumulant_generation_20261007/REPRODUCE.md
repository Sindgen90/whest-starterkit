# Reproduction and artifact map

Run PowerShell from `C:\Users\nikso\Desktop\Books\AI papers\ARC_Challenge\ARC_Whitebox_Challenge\whest-starterkit`. Use the existing `.venv-phase2`; no package or runner changes were made. Versions are recorded in preservation_verification.json. Keep numerical jobs serial. Every result-producing script refuses existing output paths; use fresh names for reruns.

The repaired submission is `submissions/phase2_joint_centered_20261007.tar.gz`, source `centered/estimator.py`. The original `joint_full_runtime/estimator.py` and `submissions/phase2_joint_full_runtime_20261002.tar.gz` remain byte-identical. The complete licensed benchmark is separately packaged as `submissions/phase2_REPRODUCTION_504aldo_V29_20261007.tar.gz`; it is not represented as an original prize-eligible solution. Both archives' absolute paths, source/archive hashes and file inventories are in *_archive.json.

## Focused verification

```powershell
.venv-phase2/Scripts/python.exe -m pytest research/phase2/cumulant_generation_20261007/test_generation.py -q
.venv-phase2/Scripts/python.exe research/phase2/cumulant_generation_20261007/verify_preservation.py
```

The three tests check the positive identity limit through mean1000, Gaussian marginal cumulants against independently split numerical quadrature, and centered algebra against raw unclipped algebra over varied means/variances. Original sources are preserved in baseline_snapshot and source_history. build.py/build_centered.py document the asserted source edits; do not rebuild frozen sources after measurement. instrument.py/instrument_public.py produce separate diagnostic sources, never packaged. The public estimator sources are complete self-contained V25/V29, pinned at commit1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d; public/504aldo_provenance.json records URLs and byte hashes, with the MIT notice preserved.

## Fresh serial screens using preserved sources/data

```powershell
.venv-phase2/Scripts/python.exe research/phase2/cumulant_generation_20261007/screen.py --names baseline_snapshot unclip34 centered coherent public_v25 public_v29 --reuse --output rerun_development.json
.venv-phase2/Scripts/python.exe research/phase2/cumulant_generation_20261007/screen.py --names baseline_snapshot unclip34 centered coherent public_v25 public_v29 --reuse --data research/phase2/cumulant_generation_20261007/public/confirmation_shard11.arrow --ids 1 2 3 4 --output rerun_confirmation.json
```

Confirmation shard11 rows1..4 are mini IDs80..83; this is verified in each report. `--reuse` calls setup once and reuses one estimator instance per method across networks. The original public V29 development screen was cold-only (new instances, no setup) and remains labeled as such. Setup is outside per-prediction timing. In-process runtime is a screen, not an execution gate. Prediction NPZs contain only offline comparison results and are never shipped.

## Unchanged official gate

```powershell
.venv-phase2/Scripts/python.exe research/phase2/benchmark_joint_official.py --candidate cumulant_generation_20261007/centered --runners subprocess --seeds 20261071 --n-mlps 2 --prefix research/phase2/cumulant_generation_20261007/rerun_centered_gate
.venv-phase2/Scripts/python.exe research/phase2/benchmark_joint_official.py --candidate cumulant_generation_20261007/public_v29 --runners subprocess --seeds 20261171 --n-mlps 4 --prefix research/phase2/cumulant_generation_20261007/rerun_public_gate
```

Default SDK resource caps, four threads, width1024/depth16. Parse n_failed_mlps and failure_breakdown; CLI exit0 can still contain failed predictions. 100-sample smoke means are not accuracy targets. V25's two residual failures are retained. run_gates.py adds read-only Windows process-tree memory observation around this exact helper; it does not modify accounting or caps. Windows RSS observation does not implement Linux RLIMIT_AS.

The archives passed `whest validate`, `whest package`, and `whest validate-package`. Exact commands and reports are in *_packaging_manifest.json and associated stdout/stderr. A successful tiny contract validation alone is insufficient; packaging additionally requires the full-width subprocess gate and held-out adjusted improvement.

## Reference diagnostics

Public Phase2 references are 1e8-sample higher moments for100 networks, not1000 networks at1e9 samples. Official baked mean targets are1e9-sample. Only moment IDs0/1/80/81 and the official confirmation shard were downloaded. All pinned hashes match published LFS hashes, and copied mean targets match exactly: reference_alignment.json. No new Monte Carlo data were generated.

- local_reference.json: teacher-forced marginal nonlinear generation, all16 layers on dev0/1.
- propagated_diagnostics.json: incoming/outgoing marginal cumulant errors, clipping rates, finite values and necessary moment slack in actual original-method recurrence. Diagnostic full-final contractions differ in rounding/cost from submission final specialization; their cost is not submission cost.
- public_propagation_diagnostics_v2.json: complete V29 recurrence and offline one-step interventions. v1 remains preserved; only its covariance MSE is invalid because it compared a logged zero diagonal with full reference covariance. v2 correctly compares off-diagonal covariance.
- birth_diagnostic.json: the leading Gaussian three-distinct third-cumulant birth tested against existing small-network four-stream references. Negative cross-stream risk estimates indicate reference noise, not negative physical error.
- comparison.csv and paired_confirmation.json: all methods' raw/adjusted/cost/runtime measurements and four-network paired differences.

MATHEMATICS.md gives centered formulas, the common damping rule, represented public sources/G feedback, exact versus approximate identities, and the fixed-weight distinction from Hanin. COMPARISON.md ranks the findings and records the recommendation. No reference arrays, targets, fitted new coefficients, or diagnostics are included in either tarball. No upload or messages were sent to AIcrowd.
