# Cumulant generation and complete public-method audit

All expectations here average Gaussian **inputs at fixed supplied weights**. Hanin's random-weight ensemble recursions and vanishing odd ensemble cumulants are inapplicable without a separate derivation. The useful hierarchy lesson is to track omissions of comparable formal order, including products of lower cumulants; inverse-width guarantees for another ensemble are not error bounds here. See the preserved fourth_representation_20261007/AUDIT.md for the full tensor audit and Hanin equation references.

## The identified defect

For incoming marginal mean m, variance v, third cumulant t, and fourth cumulant k, the incumbent uses a truncated Gaussian response expansion

\[
 M_p=E_G[\operatorname{ReLU}(Z)^p]+\delta_p,
 \qquad\delta_p=\frac{t}{6}E_G[(\operatorname{ReLU}^p)^{(3)}]
                  +\frac{k}{24}E_G[(\operatorname{ReLU}^p)^{(4)}].
\]

This expansion is an approximation. Cumulant recentering identities are exact. The implementation separately clips each delta_p to +/-0.25 v^(p/2), a heuristic. In the positive identity limit delta_3=t and delta_4=4mt+k. Independent bounds destroy cancellation of 4mt when deriving k. A valid two-point distribution with variance .75, t=.75, k=-.375 gives the same true output cumulants after a positive translation; the original guard instead produces -6.355 at mean10 and -12.850 at mean20.

`unclip34` changes only these two raw guards. M1/M2 guards, finite third history, paired fourth projection, response term selection, Gaussian pair quadrature, variance floors, and terminal mean specialization remain fixed.

## Stable marginal arithmetic

Choose c=max(m,0), and compute R_p=E[(H-c)^p], H=ReLU(Z). Write a=m/sqrt(v), s=sqrt(v), phi=normal density at a, Phi=normal CDF at a, and S=Phi(-a). When m>0, the Gaussian reference moments about c=m are

\[
 R_1^G=s\phi-mS,\quad R_2^G=v\Phi-ms\phi+m^2S,
\]
\[
 R_3^G=(m^2+2v)s\phi-m^3S,
 \quad R_4^G=3v^2\Phi-(m^3+3mv)s\phi+m^4S.
\]

These follow by integrating (Z-m)^p on Z>0 and the atom (-m)^p on Z<=0. For m<=0 choose c=0 and the ordinary truncated moments. Gaussian tail cancellation remains possible far into the negative tail; this reformulation specifically removes large positive-mean raw cancellation. Response derivatives of (H-c)^p are the binomial shift of the original response derivatives, with c held constant during differentiation. The code evaluates the combined expressions directly. Adjustments for the retained M1/M2 guards preserve exact equivalence to unclip34 in real arithmetic.

For r=R_1 the exact final identities are

\[
 \mu=c+r,\quad C_{ii}=R_2-r^2,
 \quad T_{iii}=R_3-3rR_2+2r^3,
\]
\[
 K_{iiii}=R_4-4rR_3-3R_2^2+12r^2R_2-6r^4.
\]

The centered candidate fixes marginal arithmetic only. Its off-diagonal Cov(H_i^2,H_j) and Cov(H_i^2,H_j^2) calculations and recentering retain the incumbent's response closure. Thus it does not enforce joint realizability, restore missing mixed-index tensors, or prove PSD covariance. Variance floors are recorded as stabilization, not exact identities.

## Coherent stabilization ablation

Compute the four unguarded centered corrections h_p and take one layer scalar

\[
 \alpha=\min\left(1,\min_{p,i}\frac{0.25v_i^{p/2}}{|h_{p,i}|}\right).
\]

Scale *all incoming* third and fourth quantities by alpha before marginal and bivariate nonlinear responses: t, k, P=T_iij, and Q=K_iijj. Quadratic third terms automatically receive alpha^2. Derive every outgoing marginal cumulant from the resulting common corrected moments; do not clip individual raw M_p afterward. Apply the same rule at the final layer. The threshold is copied from the incumbent, not fitted. In the positive identity limit, alpha is independent of translating m, and outgoing t,k are alpha*t,alpha*k. This is a heuristic with joint and moment-feasibility limits, not a theorem that damping improves accuracy. A layer-wide minimum can shrink sound corrections because of a single outlier.

## Public V25/V29: a different represented tensor

Pinned source: [504aldo/whest-p2-cumulant-k3](https://github.com/504aldo/whest-p2-cumulant-k3/tree/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d), MIT (see downloaded LICENSE for its exact copyright notice). The complete files include the source births, older-source compression, response tables, regeneration, feedback, and terminal specialization. Default mean-rider correction is disabled (NO_CORR=True); LAM/REF_R tables are the author's published fitted parameters and are retained for reproduction. No new targets are fitted or embedded.

With weights transposed to W=w.T, the public source tensors contain Sym(X_s x P_s x Y_s) and Sym(M_s x P_s x P_s), including entries with three distinct neuron indices. Source legs evolve by W diag(Phi), and source contributions are summed across all birth layers. Sources older than four transports share a rank384 basis; sources older than seven use a nested rank224 basis. This is projection of persistent sources, unlike simply keeping more columns of our repeatedly regenerated repeated-index reconstruction.

There is also a loss **at birth**, before history truncation. For three distinct coordinates of Gaussian incoming Z, expansion in off-diagonal covariance gives the leading connected term

\[
 T^H_{ijk}=w_{2,i}w_{1,j}w_{1,k}C_{ij}C_{ik}
 +w_{2,j}w_{1,i}w_{1,k}C_{ji}C_{jk}
 +w_{2,k}w_{1,i}w_{1,j}C_{ki}C_{kj}+O(C_{\rm off}^3),
\]

where w1=Phi and w2=phi/sigma. This follows from the three connected two-edge Gaussian contraction trees; it is a local Gaussian covariance expansion, not a width-scaling guarantee. At birth the public hub source Sym(3A,I,A diag(w2)), A=diag(w1)C_off, represents all three trees. Our injected source has the form Sym(A,I,I), which is zero when i,j,k are all distinct. Retaining such a source for additional layers cannot recover the distinct-index *birth* content that was never represented. This explains a concrete mathematical difference beyond the fourth-core feedback and older history. None of the stored two-index reference planes determines a full three-index tensor, so its full-width error is not directly measured by those planes alone.

The source identity M=P diag(s)+3A diag(e)+Z L^T lets the method combine transport and D21 contractions without transporting a full third dense leg. These reassociations are exact for the chosen source representation; rank16 residual/feedback range finders, age confinement, term pruning, and the underlying nonlinear truncation are approximations. V29 changes matrix multiplication arithmetic and allocation; it retains V25's closure, with floating-point rounding differences.

Its G is a contracted fourth-order *matrix core*, not Q_ij=K_iijj. The Q slice enters its diagonal through row sums; off-diagonal content includes hub contractions and symmetrized K31. It approximates G_off=lambda*C_off. For a regenerated post-state G=diag(g)+lambda*(C-diag(v)),

\[
 \operatorname{diag}(WGW^T)=(W\odot W)(g-\lambda v)+\lambda\operatorname{diag}(WCW^T).
\]

That diagonal formula is exact **conditional on the regenerated G model**. Taking the next off-diagonal as lambda*C_pre_off discards the off-diagonal spill of W diag(g-lambda v) W.T. The adaptive lambda is an empirical rule based on mean(dG)/mean(var), with published reference ratios and a clipped scale range. These are not Hanin ensemble identities.

The crucial feedback creates new third sources from X3=A diag(lambda*w1)+P diag(w1^2*dG) and rank-one Y3=(METRIC_C/4)*w2*1.T, plus a thin M source. The nonlinear term tables include third-squared and third-fourth products. Their marginal mean terms are Gaussian mean +t*f'''/6+k_eff*f''''/24+t^2*f^(6)/72+t*k_eff*f^(7)/144. Higher response derivatives here are products of retained cumulants, not newly represented fifth/seventh cumulants. Our old eight bivariate kernels and linear marginal expansion are not the same truncation.

The author's [derivation map](https://github.com/504aldo/whest-p2-cumulant-k3/blob/1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d/docs/derivation_code_map.md) ties the source form to the challenge companion paper S.4.3 equations15–20; its own equation-number caveat is preserved. Those references concern fixed-weight propagation; do not substitute them for Hanin's weight-ensemble recursions. The September22 [public correction](https://discourse.aicrowd.com/t/everything-we-tried-a-factorized-k-3-cumulant-propagation-estimator-at-0-25-x-b-where-its-flops-go-and-25-measured-dead-ends-team-504aldo-rank-10/18218/5) explicitly distinguishes G from the paired fourth slice. The reported ablations and September/October scores are author reports, not measurements made here or verified current rankings.

## Reference and eligibility corrections

The [1000-network higher-moment dataset](https://huggingface.co/datasets/keenanpepper/arc-whestbench-higher-moments-2026) is Phase1 (256x32), at 1e8 fresh samples. The separate [Phase2 dataset](https://huggingface.co/datasets/keenanpepper/arc-whestbench-p2-higher-moments-2026) is 100 mini networks (1024x16), also 1e8 fresh samples, with copied official 1e9-sample mean targets. It includes pre marginal powers, pre M11/M21/M31/M22 and post M11 plus marginal powers. It lacks a full fourth tensor and post M21/M22. Marginal oracle tests can isolate nonlinear generation; incoming contraction errors include all upstream approximation losses and cannot uniquely identify a missing tensor component.

The [October5 organizer clarification](https://discourse.aicrowd.com/t/eligibility-of-solutions-built-on-another-participants-mit-licensed-forum-code-4-6/18240) permits licensed public bases and their published constants with attribution, but requires substantial original changes for prize consideration. A byte-for-byte reproduction is a benchmark, not a claim of original prize-eligible work. No messages or submissions are sent to AIcrowd.
