# Approach 1: analytic surrogate and independent residual

**Experimental, not recommended.** The frozen 33-feature dictionary fails the independent variance screen and actual paired accuracy screen. It passes unchanged official subprocess execution and packaging; its separate archive is intentionally provided despite its accuracy loss.

For standard Gaussian X, let a_r be normalized columns of the first supplied weight matrix, choosing the 16 largest column norms. With t_r=0, use

\[g(x)=b+\sum_{r=1}^{16}A_r(a_r^Tx)+\sum_{r=1}^{16}c_r(a_r^Tx)_+.\]

The coefficients are vectors for **every neuron of every layer**, fitted by one regularized least-squares solve on 512 fresh network evaluations. Only coefficients are fitted; directions are fixed by the supplied weights. For a general hinge, with s=||a|| and alpha=t/s,

\[E[(a^TX+t)_+]=s\phi(\alpha)+t\Phi(\alpha),\qquad E[AX]=0.\]

Feature centers use the computed norms, including rounding of normalized directions. The implementation uses 0.001 times the mean Gram diagonal as ridge. No target table, sampled reference moment, or approximate cumulant mean is an analytic center. The final estimator is

\[\widehat\mu=E[g(X)]+{1\over4096}\sum_{k=1}^{4096}[f(X_k)-g(X_k)].\]

Fit samples and correction samples are disjoint independent draws. Conditional on the fitted coefficients, this is unbiased in exact arithmetic, and its coordinate variance is Var(f-g)/4096. Ridge does not change that identity. First-layer means are returned by their exact Gaussian formula; all later rows use the residual estimator. All dictionary construction, Gaussian draws, feature construction, full-network evaluations, fit and corrections occur inside the metered predict call.

[Nakatsukasa](https://arxiv.org/pdf/1806.05492), section 3.2, equations (3.8)-(3.9), analyzes bias when fitting and integrating with the same samples; its final paragraph describes an independent pilot/residual construction. Our Gaussian ReLU dictionary and multi-output ridge fit are adaptations. We use the independent construction rather than assuming the same-sample fit is unbiased. The paper offers no guarantee that this dictionary captures a deep width-1024 fixed network.

The frozen protocol is PROTOCOL.md. Pilot fit seeds are 20261201+ID and independent test seeds 20261202+ID; 1024 test samples per network. Final estimator randomizations use 20261211/12/13, with RNG seed equal to that value plus 1009 times mlp.seed. Public network IDs 0,1 and their baked 1e9-draw targets are development evidence, not fresh holdouts. No randomization was selected as a winner.

| ID | Independent residual variance | Residual / plain-sampling variance | Predicted raw MSE at 4096 |
|---|---:|---:|---:|
| 0 | 0.08348395 | 1.05812 | 2.03818e-5 |
| 1 | 0.09105685 | 1.05418 | 2.22307e-5 |

Actual average across both networks and all three randomizations: raw MSE **2.090582088e-5**, nominal adjusted **2.090582088e-6**. Baseline on the same networks: raw **9.136408012e-7**, adjusted **9.136408012e-8**. Candidate individual raw MSEs range 1.719777462e-5 to 2.729547154e-5. Actual FLOPs are **160,173,107,356**, so both methods have the 0.1 multiplier. Roughly 0.00374 average residual variance would be needed to match this baseline with 4096 samples; measured residual variance is about 0.08727. A larger, better dictionary might change this outcome, but we did not search for one.

In-process mean wall/residual times are 0.70438/0.05451 seconds, versus baseline 6.97701/0.20484 seconds. These establish no eligibility. The unchanged official CLI at seed 20261081, two width-1024/depth-16 networks, four threads and default caps passes **2/2** candidate predictions, with walls 0.72932/0.70258 seconds and residuals 0.05783/0.05293 seconds. The gate uses 100-sample smoke targets strictly for execution. Official validate, package, validate-package also pass. The archive contains estimator.py, LICENSE, PROVENANCE.md and the official manifest; archive/source hashes are in surrogate_archive.json.

All predictions are finite. An initial development run rejected indexed assignment because flopscope arrays are immutable; it is preserved in surrogate_screen.json and surrogate_screen_failure.txt. The functional concatenate fix does not change the estimator mathematics; corrected results are surrogate_screen_v2.json. Pilot source SHA predates this API fix. No evaluator or metering changes were made. The pilot and three actual randomizations agree on the large loss, so neither RQMC nor fresh held-out reference generation meets its frozen trigger. Finite public-reference error remains distinct from estimator sampling variability; the public reference has vastly more draws than this estimator.

Raw evidence: surrogate_pilot.json and its variance NPZs; surrogate_screen_v2.json and its per-network prediction NPZs; surrogate_gate*.json; surrogate_validate*.json; surrogate_packaging_manifest.json; surrogate_archive.json. Standalone source: surrogate/estimator.py. Reproduce with commands in REPRODUCE.md.
