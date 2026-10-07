# Joint cumulant candidate, 1 October 2026

Standalone local extension of the Gaussian covariance baseline. Upstream
venture_agent_ai MIT attribution is preserved in LICENSE. Baseline repository:
https://github.com/agentaiventure-dot/arc-whest-estimator,
commit fb33bb5a66b4b0aed3022e6a70a9274d5957d61d.

Mixed cumulant response terms were inspired by the Puffi A716 write-up:
https://discourse.aicrowd.com/uploads/short-url/iDdUNo66HzI7ovRqhQqpLOFFugI.pdf
This is our approximate closure, not a reproduction of the published algorithm.

Retains full covariance, compressed third cumulants and paired fourth cumulants
for all sixteen layers. One-node quadrature, one third-cumulant source history.
No fitted calibration or learned coefficient is applied. No targets, sampled
moments, datasets or inference-time fitting are bundled.
See ../JOINT_PROPAGATION.md for validation and limitations.
