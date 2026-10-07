# Published Phase 2 baseline, submission 329951

Author: venture_agent_ai. License: MIT, preserved in LICENSE.

Source repository: https://github.com/agentaiventure-dot/arc-whest-estimator

Pinned commit: `fb33bb5a66b4b0aed3022e6a70a9274d5957d61d`.

Source file: `estimator/estimator.py`, downloaded 26 September 2026.

Local estimator SHA-256: `0a22f57ce5670d1acac0b9bc9d15651fd49e04f1cae7210e2068369e3685c37c`.

The estimator is byte-for-byte unchanged. Its 16 x 5 calibration coefficients
are the author's published values; no local fitting or leaderboard tuning was
performed. The source contains an exact bivariate-Gaussian covariance identity
evaluated using eight-node numerical quadrature, plus a calibrated output
correction. The network's joint distribution is still approximated as Gaussian.

Author's write-up:
https://discourse.aicrowd.com/t/phase-2-write-up-exact-bivariate-relu-covariance-plus-per-layer-calibration-3x-over-the-kit-baseline-with-negative-results-submission-329951/18214

See ../BASELINE_REPLICATION.md for local measurements and limitations. The
upstream training corpus has not been audited for overlap with our public
evaluation networks, so our results are reproduction checks, not a new
independent generalization claim.
