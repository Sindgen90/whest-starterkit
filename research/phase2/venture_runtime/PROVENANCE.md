# Runtime derivative of published Phase 2 submission 329951

Upstream author: venture_agent_ai. MIT license preserved in LICENSE.
Repository: https://github.com/agentaiventure-dot/arc-whest-estimator
Pinned commit: fb33bb5a66b4b0aed3022e6a70a9274d5957d61d.
Original source SHA-256: 0a22f57ce5670d1acac0b9bc9d15651fd49e04f1cae7210e2068369e3685c37c.

Local changes on 1 October 2026:

- Factor the common rho-squared term out of the eight-node covariance integral.
- Evaluate its Gaussian density using division rather than a reciprocal power.
- Skip the unused covariance update after returning the final layer's mean.
- Remove the unused prediction RNG initialization.

Quadrature nodes, frozen calibration coefficients, Gaussian closure, and output
contract remain the same. Floating-point operation order differs; numerical
agreement is checked rather than claimed to be bit-identical.
No sampled oracle moments, reference targets, fitting outputs, or research
diagnostic code are included in this submission.

See ../RUNTIME_BASELINE.md for numerical comparisons and resource validation.
