# Box--Muller RQMC variant

This development estimator changes exactly one component of the frozen
4,096-point rank-1 lattice baseline: paired uniform coordinates are mapped to
standard Gaussians with the Box--Muller transformation instead of applying the
inverse Gaussian CDF coordinatewise.

The lattice, multiplier, random-shift seeding, sample count, batched MLP
forward pass, and equal-weight averaging are unchanged.

This directory is not a validated submission package unless the paired
development gate in `research/scripts/run_box_muller_rqmc_4096.py` passes.
