# Full joint cumulant runtime optimization

2 October 2026. The preserved reference is
`joint_full_candidate/estimator.py`, SHA-256
`3539c3eaa325a5ffa92279fb3e39034ba5b36d73c97612a7ecb0680289bca8d9`.
The derivative is `joint_full_runtime/estimator.py`, SHA-256
`bd4c808d47068dcf2b0515472925598f56c43b2b603a2159b982d4d1f9dbce7e`.
No equations, quadrature nodes, history length, rich-layer count, clipping bounds
or calibration were tuned against targets. No prior source, result or archive
was replaced. The previous `submissions/phase2_best_20261001.tar.gz` is retained.

## Investigating the timing discrepancy first

The archived official full-candidate failure remains in
`results/joint_full_subprocess_20261001.json`: 0.614115 s residual versus the
0.4 s cap. Fresh **unchanged official CLI** calls on the same generated network,
seed 20260926, width 1024, depth 16, four threads and 100 smoke-test samples gave:

| Reference source | Predict wall seconds | Residual seconds | FLOPs | Failures |
|---|---:|---:|---:|---:|
| Official local runner | 6.963509 | 0.267779 | 202,869,442,542 | 0/1 |
| Official subprocess runner | 8.371986 | 0.350419 | 202,869,442,542 | 0/1 |

Thus the fresh check reproduced a timing gap, but **did not reproduce the
historical cap failure**. It does not invalidate that failure. The two official
fresh reports have identical per-layer MSEs and FLOPs. The old frozen public
checks and the generated smoke test also use different networks, so their times
cannot be interpreted as a controlled runner comparison.

Read-only inspection of the installed worker shows that import/setup, request
decoding and weight construction happen before its BudgetContext; response
serialization and the harness finiteness check happen after it. Its prediction
window includes float32 output materialization and deletion of the prediction
reference. Startup/JSON transport wall time is therefore not the reported
residual. The thread limit is inherited in the worker environment.

`profile_joint_overhead.py` tests fresh processes, then a repeated call, using
the same generated network. All modes have C-contiguous float32 weights with
strides `(4096,4)`. It observes GC callbacks without changing their policy:

| Diagnostic mode | First residual s | Repeated residual s | First / repeated GC s |
|---|---:|---:|---:|
| Official-style float32 materialization | 0.284480 | 0.331161 | 0.024720 / 0.008054 |
| Legacy float64 conversion | 0.250941 | 0.328408 | 0.022889 / 0.008128 |
| Retained decoded JSON weight lists | 0.331145 | 0.325973 | 0.007943 / 0.007872 |
| Decoded lists released before prediction | 0.328310 | 0.323931 | 0.008199 / 0.007889 |

Materialization itself takes 0.118-0.151 ms; float64 conversion takes
0.022-0.025 ms. Releasing the retained input lists changes residual by roughly
2-3 ms in this experiment. These observations do not support materialization,
weight layout, or retained JSON/GC work as the main explanation. First versus
repeated calls also vary. The exact cause of the earlier 0.614115 s failure is
not isolated, and no specific allocator/cache/host-load explanation is claimed.
The optimization reduces real repeated work and temporary buffers; it does not
alter metering, warnings, GC settings, thread count or runner implementation.
Diagnostic in-process timings are never an eligibility gate.

Raw reports are retained under `results/joint_overhead_baseline_20261001_*`
and `results/joint_overhead_diagnostic_20261002*`. The baseline wrapper's initial
summary parser expected the wrong JSON key; its complete official reports
remain intact and are the evidence for the table above. The corrected wrapper
reads `results`, retains failures and exits unsuccessfully if any official check
fails or cannot be parsed.

## Changes to the estimator

At the final layer there is no outgoing state. For each output column `w_a`,
compute `v_a = w_a.T @ C @ w_a` directly with the metered contraction
`einsum('ij,ia,ja->a', C, W, W)`. The third cumulant requires only
`sum((A*B)*B, axis=1)` after factor transport; the two dense products that build
the repeated third slice are unnecessary. Fourth transport needs only the
diagonal of `S.T @ Q @ S`, where `S=W*W`, followed by the existing diagonal
adjustment. Only the first ReLU moment and its third/fourth response corrections
are computed. Covariance matrices and raw moments 2-4 are not formed there.
Diagnostic collection intentionally uses the full path, which is independently
checked against the specialized terminal mean.

For earlier layers, keep all joint covariance, repeated third slices and
paired fourth cumulants. Apply the eight non-Gaussian response coefficients to
O(width) response vectors instead of repeatedly scaling O(width squared)
kernels. Drop unconsumed derivatives of raw moments 3 and 4 and the redundant
variance floor. Reuse `2*mu` in repeated/fourth updates. Metered `fnp.add` and
`fnp.multiply` with public `out=` buffers reuse dead correction matrices and
symmetrization/source buffers. Old contents are never reused after mutation;
input weights and returned means remain separate, as repeat/input tests check.

| Unchanged official subprocess, seed 20260926 | Calls | FLOPs | Residual s | Predict wall s |
|---|---:|---:|---:|---:|
| Preserved reference | 3,367 | 202,869,442,542 | 0.350419 | 8.371986 |
| Terminal/coefficient specialization (v1) | 3,257 | 196,325,437,438 | 0.312894 | 8.300011 |
| Buffer reuse and repeated-operation removal (v2) | 3,122 | 196,317,442,558 | 0.298661 | 8.382649 |

The measured residual falls about 14.8% in these early gates; overall prediction
wall time does not show a material reduction. Counted calls fall 7.3%, FLOPs
fall 3.23%, and the score multiplier stays at the 0.1 floor. The tested v1 bytes
are saved as `results/joint_full_runtime_v1_estimator.py`. Both early gate reports
are retained; v2 is the frozen source used for accuracy and final gates.

## Frozen accuracy preservation

`benchmark_joint_runtime.py` compares the preserved reference and derivative
on development networks 0-1 and the same eight frozen exposed evaluation
networks 4-7 and 12-15. There is no fitting. The declared tolerances were
`rtol=5e-6`, `atol=2e-6`, maximum absolute prediction difference `1e-5`, and
absolute relative final-MSE drift at most `0.001` for each network.

All ten networks pass. Largest difference across all layers and neurons:
**3.814697265625e-6**. Largest individual relative MSE drift: **0.016771%**.

| Networks | Reference raw final MSE | Derivative raw final MSE |
|---|---:|---:|
| Development 0-1 | 9.135972748e-7 | 9.136408012e-7 |
| Frozen evaluation 4-7, 12-15 | 9.238894887e-7 | 9.238858483e-7 |
| All ten | 9.218310459e-7 | 9.218368389e-7 |

These are rounding differences, not an accuracy gain. The original full joint
improvement is preserved. The ten-network aggregate changes about 0.00063%;
the frozen eight-network aggregate changes about -0.00039%. These networks were
previously exposed, so this establishes numerical equivalence locally and does
not establish hidden-test improvement. Evidence:
`results/joint_full_runtime_accuracy_20261002.json`.

## Final execution gate and preservation audit

The frozen v2 bytes were then run through the unchanged official subprocess
runner for seeds 20260926, 20260927 and 20260928, two generated networks per
worker. All **six** predictions pass with the default 0.4 s residual, 120 s wall
and 2^41 FLOP caps. This covers three fresh workers and their repeated calls.

| Seed | First prediction residual s | Repeated prediction residual s |
|---|---:|---:|
| 20260926 | 0.296688 | 0.291997 |
| 20260927 | 0.295512 | 0.292064 |
| 20260928 | 0.298400 | 0.295137 |

Mean residual is **0.294966 s**, maximum **0.298400 s**, leaving 0.101600 s
against the local cap. Maximum prediction wall time is 8.395522 s. All six
have 196,317,442,558 FLOPs and 3,122 counted calls. Reports and stderr are in
`results/joint_full_runtime_final_gate_20261002*`. The tests pass **18/18**
and official `whest validate` returns `ok: true`; their logs are saved as
`results/joint_full_runtime_tests_20261002.txt` and
`results/joint_full_runtime_validate_20261002.json`.

`summarize_joint_runtime.py` checks that gate/accuracy source hashes match the
current source and audits 190 unchanged files: original candidate sources,
prior result JSONs, all existing archives and installed whestbench/flopscope
Python sources. The unchanged snapshot is
`results/joint_overhead_preserved_20261002.json`; the final audit is
`results/joint_full_runtime_audit_20261002.json`. One newly created diagnostic
manifest was captured while it was still appending, so its completed four
children and reference-source hashes are checked explicitly and both manifest
hashes are reported. The snapshot is never rewritten to hide this difference.

The derivative has passed the local official execution gate and preserves
accuracy on the frozen comparison. The earlier failures are retained. This is
a local result, not a remote runtime guarantee; the exact cause of the historical
timing spike remains unresolved. The prior upload archive remains unchanged,
and no new archive or upload is made.

## Reproduction

Run from the starterkit root. Use new output names for repeats; each helper
refuses to replace existing evidence. The official helper invokes the installed
`whest.exe` with its default Phase 2 caps and saves stdout and stderr from every
call, serially. It does not replace or instrument the official runner.

```powershell
& .venv-phase2/Scripts/python.exe research/phase2/profile_joint_overhead.py --output research/phase2/results/joint_overhead_diagnostic_repeat.json
& .venv-phase2/Scripts/python.exe -m pytest research/phase2/test_joint_cumulant.py research/phase2/test_joint_full_runtime.py -q --disable-warnings
& .venv-phase2/Scripts/python.exe research/phase2/benchmark_joint_runtime.py --output research/phase2/results/joint_full_runtime_accuracy_repeat.json
& .venv-phase2/Scripts/python.exe research/phase2/benchmark_joint_official.py --candidate joint_full_runtime --seeds 20260926 20260927 20260928 --n-mlps 2 --prefix research/phase2/results/joint_full_runtime_gate_repeat
$env:PYTHONIOENCODING='utf-8'
& .venv-phase2/Scripts/whest.exe validate --estimator research/phase2/joint_full_runtime/estimator.py --format json
& .venv-phase2/Scripts/python.exe research/phase2/summarize_joint_runtime.py --accuracy research/phase2/results/joint_full_runtime_accuracy_20261002.json --gate research/phase2/results/joint_full_runtime_final_gate_20261002.json --preserved research/phase2/results/joint_overhead_preserved_20261002.json --completed-diagnostic research/phase2/results/joint_overhead_diagnostic_20261002.json --output research/phase2/results/joint_full_runtime_audit_repeat.json
```

The 100-sample official runs test execution only; their noisy MSE is never used
as accuracy evidence. No new upload archive is created by this work.

## Packaging follow-up requested by the user

After the research checks above, the user requested a submission archive.
The official folder packager created
`submissions/phase2_joint_full_runtime_20261002.tar.gz` (6,373 bytes), containing
`estimator.py`, `LICENSE`, `PROVENANCE.md` and the official `manifest.json`.
Official `whest validate-package` passed with no issues. Every manifest hash
was verified, and the packaged estimator bytes match the frozen source that
passed the six official subprocess gates. The previous archive is unchanged.

Archive SHA-256:
`89035799f6acfe5dd14948fbedefd37ea2bc35472ce2ce72518c4485676782b3`.
The package audit is `results/joint_full_runtime_package_audit_20261002.json`;
official package/validation reports are
`results/joint_full_runtime_package_20261002.json` and
`results/joint_full_runtime_validate_package_20261002.json`.
No upload was performed.
