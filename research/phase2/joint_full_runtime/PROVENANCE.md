# Full joint cumulant runtime derivative

Created 2 October 2026 from `../joint_full_candidate/estimator.py`, SHA-256
`3539c3eaa325a5ffa92279fb3e39034ba5b36d73c97612a7ecb0680289bca8d9`.
The original source, recorded failures, results and submission archives are retained.

The derivative retains the same moment closure, one-source third history,
projected paired fourth cumulants, two quadrature nodes, all sixteen rich
layers, bounds and absence of output calibration. Only the terminal mean is
computed at the last layer: diagonal covariance/fourth contractions and the
diagonal third slice replace unused full matrices and higher raw moments.
Response coefficients act on vectors rather than dense kernel matrices;
unused derivatives and repeated scaling are removed. Public flopscope `out=`
operations reuse buffers only after their old contents are no longer consumed.
All numerical computation remains metered through flopscope. No targets,
samples, reference data, fitted coefficients, or runner/GC configuration changes
are part of inference.

Source SHA-256:
`bd4c808d47068dcf2b0515472925598f56c43b2b603a2159b982d4d1f9dbce7e`.
See `../JOINT_RUNTIME_OPTIMIZATION.md` for investigation, accuracy evidence,
the unchanged official subprocess gate, reproduction and limitations.
This folder is a research derivative; the previous upload archive is retained.

Upstream venture_agent_ai MIT attribution is preserved in LICENSE. Baseline:
https://github.com/agentaiventure-dot/arc-whest-estimator,
commit `fb33bb5a66b4b0aed3022e6a70a9274d5957d61d`.

Mixed cumulant response terms were inspired by the Puffi A716 write-up:
https://discourse.aicrowd.com/uploads/short-url/iDdUNo66HzI7ovRqhQqpLOFFugI.pdf.
This is our approximate closure, not a reproduction of that submission.
