# Experimental partial-frame Kerdock/Puffi reconstruction — not promoted

Built 26 September 2026 from Skye Nygaard's MIT Kerdock construction and
Puffi's published ridge-control equations A31-A37. It is NOT either author's
archived estimator and NOT a current Phase 2 leader reproduction.

Compared with the disclosed full design, this adaptation uses 4–16 of the
512 Kerdock bases at width 1024, a fixed offline Haar rotation, no coordinate
basis, dense propagation, no pruning or terminal bypass, and our pre-existing
approximate A716 reconstruction without Puffi's unavailable fitted constants.
Partial frames do not form the complete spherical 5-design. The mean Gaussian
radius is precomputed offline; positive homogeneity integrates the radius
exactly, while angular quadrature remains approximate.

The direction geometry, FWHT, and dual ridge solve pass independent numerical
checks. Eight-network experiments are in ../results/kerdock_development.json.
All tested configurations lose to the existing covariance/MC blend after
compute charging. Best tested partial-frame adjusted score: 6.27428e-7;
existing blend on the same networks/seed: 2.85377e-7. This rejects these
configurations, not the original full/pruned Puffi method.

Default configuration is retained as the tested k8_cv5 diagnostic, not a
recommended submission. No archive was submitted.

Asset generator: ../build_kerdock_assets.py. Its finite-field generalization
uses the construction from:
https://github.com/SkyeNygaard/ARC-Whitebox/blob/main/arc_whitebox/scripts/eval_kerdock_design.py

The full MIT notice is in LICENSE. Assets depend only on dimension and fixed
offline design seeds; no evaluation network or label enters their generation.

Puffi equations:
https://discourse.aicrowd.com/uploads/short-url/iDdUNo66HzI7ovRqhQqpLOFFugI.pdf
