# joint_centered

Source SHA256: d7a52fb02d45386b9012af9785368d45ef2dba08975b79dfa27225f2b3313f49

Frozen protocol and complete results: research/phase2/cumulant_generation_20261007/. Incumbent and all earlier archives preserved. No AIcrowd upload. Official subprocess CLI defaults, four threads, two smoke networks at seed20261071 passed. Accuracy uses distinct reserved public mini IDs80..83, separate from development IDs0/1. Official smoke targets are not accuracy evidence.

Held-out aggregate nominal adjusted objective: 8.19074758943e-08; incumbent: 8.41846140635e-08. Only four confirmation networks: this is a local comparison, not a leaderboard score or a guarantee.

No sampled/reference arrays, evaluation targets, or offline diagnostics are included. All estimator inference preparation uses metered operations.

Original local change: remove independent clipping of raw M3/M4 corrections. Compute marginal moments about max(incoming_mean,0) and recenter consistently, preserving M1/M2 guards; avoid large positive-mean raw cancellation. Off-diagonal closure remains unchanged.
