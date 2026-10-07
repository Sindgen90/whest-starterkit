# Fourth-order representations: identities and closure

This study conditions on **fixed supplied weights**, averaging over Gaussian inputs. All indices below are neuron coordinates. No independent-source decomposition, averaging over weights, or vanishing-odd-cumulant assumption is made. The unchanged incumbent is copied under `baseline_snapshot/`. The third-source algorithm, full covariance, quadrature order, clipping and one-generation history remain fixed in the primary comparisons.

## Exact quantities and conversions

For centered activations x=h−μ, Cij=E[xi xj], Tijk=E[xi xj xk], and

Kijkl = E[xi xj xk xl] − Cij Ckl − Cik Cjl − Cil Cjk.

K is invariant under all 24 permutations; R_ij=K_iiij generally differs from R_ji. Q_ij=K_iijj is symmetric; Q_ii=R_ii=κ4_i. The all-equal subtraction is 3 Cii². In terms of raw marginal moments,

κ3_i=M3_i−3μ_i M2_i+2μ_i³,

κ4_i=M4_i−4μ_i M3_i−3M2_i²+12μ_i² M2_i−6μ_i⁴.

Writing Bpq_ij=Cov(h_i^p,h_j^q), the exact recentering identities are

Qij = B22_ij − 2μ_j B21_ij − 2μ_i B21_ji + 4μ_i μ_j Cij − 2 Cij²,

Rij = B31_ij − 3μ_i B21_ij + 3μ_i² Cij − 3 Cii Cij.

B22 already subtracts the product of raw second moments. Subtracting Cii Cjj again in the Q formula would be wrong. These identities are exact when their constituent moments are consistent; the approximate bivariate closure does not ensure that consistency.

## Global projections and affine transport

P_Q K retains all permutations of 2+2 and all-equal entries. P_QR additionally retains every 3+1 entry. All other entries are zero. Let z=Wᵀh, w_a=W[:,a], s_a=w_a², d_i=Qii, R°=R−diag(diag R). Every affine tensor contraction is exact for its specified represented input tensor:

κ4_a(P_Q K)=3 s_aᵀ Q s_a−2 Σ_i d_i w_ia⁴,

Q'_ab(P_Q K)=s_aᵀ Q s_b+2(w_a⊙w_b)ᵀ Q(w_a⊙w_b)−2Σ_i d_i w_ia² w_ib²,

R'_ab(P_Q K)=3(w_a⊙(Q s_a))ᵀw_b−2Σ_i d_i w_ia³ w_ib.

The first expression counts six permutations per unordered unequal pair; the −2 diagonal correction counts all-equal once. The incumbent keeps the first Gram term of the Q expression off diagonal; its missing term is 2(H_ab−D_ab). Its diagonal is set to the exact projected marginal. This **transport approximation** is separate from throwing away the source tensor's other index patterns. Complete paired-Q Q transport is affordable only in the small controls here.

For the additional 3+1 entries, define u_a=R° w_a. Their exact contributions are

Δκ4_a = 4 Σ_i w_ia³ u_ia = 4 Σ_(i≠j) W_ia³ W_ja R_ij,

ΔQ'_ab = 2 Σ_i [w_ia² w_ib u_ib + w_ib² w_ia u_ia],

ΔR'_ab = Σ_i [w_ia³ u_ib + 3 w_ia² w_ib u_ia].

Q'_aa and R'_aa are assigned the same κ4_a. Removing the R diagonal before these additions prevents all-equal double counting. R transport is nonsymmetric. `test_equations.py` compares these formulas with independently populated literal permutation tensors and four successive dense affine contractions, including negative fourth cumulants.

For disjoint coordinate blocks B, P_B retains the full tensor only when all four indices lie in one block. The hybrid models are

K_E = P_Q K + Σ_B [K_B−(P_Q K)_B],

K_F = P_QR K + Σ_B [K_B−(P_QR K)_B].

Every source block contributes to every output Q/R entry and every destination core. The subtraction is performed **before** contraction. Block-only D sets cross-block K to zero; it retains the **full global C**. A block core's storage is n b³; transport into all destination blocks has n² b² intermediates and O(n² b³) arithmetic. Computing global output Q from cores costs O(n³ b+n² b³), and reconstructing global Q into output cores costs O(n³ b+n² b³). Small storage does not imply a cheap estimator.

Partitions are fixed after construction from the first propagated covariance. No runtime partition changes are supported. A valid hypothetical change would project the old represented tensor onto each new coordinate core: retain old-core residual entries whose four coordinates still belong to the same old block, and add the restricted old global model. Entries crossing old blocks cannot be recovered. Simply relabeling/reshaping cores would be invalid.

## Defined self-contained nonlinear update

Marginal ReLU moments use the incumbent Gaussian responses plus linear κ3/κ4 corrections and the same clipping. B11/B21/B22 use the incumbent Price-integral Gaussian anchors and eight response products. B31 has an additional Price anchor; with ρ=Cij/(σiσj), t=uρ, a=μ/σ, D_t the standardized bivariate density at the two zero thresholds, and Φ the normal CDF,

B31_ij = 3 Cij M2_i Φ(a_j)

 + 6 Cij² ∫₀¹(1−u){ (σi/σj)(1−t²)D_t + (φ(a_j)/σj) σi(a_i−t a_j) Φ[(a_i−t a_j)/√(1−t²)] }du.

The two-node rule matches the incumbent quadrature; this is numerical quadrature of an exact Gaussian covariance identity. Diagonals use Gaussian raw moments directly. Nonguassian bivariate terms retain the incumbent P=T_iij and Q responses and add R°_ij Fp'''_i Fq'_j/6 plus R°_ji Fp'_i Fq'''_j/6. This is a truncated response expansion, not exact integration of a realizable joint density.

For generic within-block entries, let g_i=E_G ReLU'(z_i), a_i=E_G ReLU''(z_i), b_i=E_G ReLU'''(z_i), and use transported input C,T,K restricted to the destination core. The proposed update is

Kout_ijkl ≈ g_i g_j g_k g_l Kin_ijkl

 + Σ_(root i) b_i g_j g_k g_l Cij Cik Cil

 + Σ_(center pair i<j) a_i a_j g_k g_l Cij(Cik Cjl+Cil Cjk)

 + Σ_(root i, neighbor j≠i) a_i g_j g_k g_l Cij T_ikl.

The sums range over the four *argument positions*, with k,l the remaining positions; the middle sum has six terms (12 paths), the last has 12 terms. These are the connected Gaussian trees and leading T*C contributions, plus linear incoming K through gains. The result is permutation-symmetrized. Every 2+2, 3+1 and all-equal entry is then replaced by the bivariate/marginal identities above, keeping the closure's marginal moments, covariance and repeated planes aligned. References never supply missing inference entries.

The third tensor used here is exactly the incumbent's symmetrized CP source: T≈Σ_r sym(A_r⊗B_r⊗B_r)/3. It is transported exactly within that CP representation. The retained source is multiplied by gains, the same finite-history condition is applied, and the repeated/diagonal residual is reinjected identically. Changing the fourth state can change these moments; it does not change the third algorithm. The small exact-Q control changes the paired off-diagonal contraction but still omits R response terms, so it is not identical to oracle B's complete physical Q tensor.

Shared numerical guards floor marginal variances at 1e−12, clip standardized Gaussian pair correlations to ±0.999999, and bound each raw-moment correction by 0.25 σ^k. These are heuristics, not distributional identities or covariance PSD repairs. In particular, positive covariance diagonals and clipped pair correlations do not imply a PSD full covariance.

## Measured priorities and structural omissions

The strongest identified **local fourth-generation** error is the raw-moment guard: in the positive-support identity limit of ReLU, ΔM3=κ3 and ΔM4=4μκ3+κ4. Independent bounds 0.25σ³/0.25σ⁴ do not preserve that coupling under recentering. The valid distribution μ+(−0.5,1.5) with probabilities (0.75,0.25) has C=0.75, T=0.75, K=−0.375, unchanged by ReLU at μ=10 or20. The guard instead produces K≈−6.35457 or−12.8498 and violates K≥−2C². Float64 checks isolate this from rounding. On complete-fourth oracle inputs, holding mean/M2 fixed and disabling only raw-M3/M4 clipping offline reduces next marginal-K4 risk from33888.7 to8.21954. This is a post-hoc explanation, not a tested recurrent fix or a promotion.

For **source representation**, retaining R resolves much more actual-weight contraction loss than completing paired-Q transport alone; cores recover further mixed patterns. For **full-width deployment**, transport cost and nonlinear recurrence eliminate the observed small-width gains. Those measured priorities qualify the structural list below; no width-based asymptotic ordering is assumed.

1. Source 2+1+1 and four-distinct K can survive contractions at the same order as retained Q/R. Blocks recover only those lying in the same coordinate block. Mixed patterns crossing blocks remain absent. Their actual-weight contraction errors, rather than only tensor norm, determine their relevance.
2. The fast global Q transport still omits its cross-pairing term. R and marginal contractions, and core-residual contractions, are exact within their input representation; that does not make hybrid Q transport globally exact. Common-state A/B separates this omission.

   In E/F, an exactly transported destination core's Q restriction can therefore differ from the fast global pre-Q plane used in the bivariate update. The post-update anchors deliberately replace core Q/R with the closure's repeated planes. This maintains post-state agreement but discards that local transport information. Oracle E/F use complete affine contractions; the fast recurrent E/F implementations are a separate cost-driven approximation. Small exact-transport controls quantify this distinction.
3. The generic nonlinear core omits connected Gaussian graphs with at least four cross-covariance edges, K*C and further K-response terms, T*T interactions, and higher responses. Some interactions between retained lower cumulants have the same perturbative order as retained linear K. The incumbent already includes a selected Pij Pji response, not all T*T diagrams. No small parameter established for these fixed-width input distributions makes these omissions negligible.
4. Distinct third entries lost by source reinjection/one-generation history are needed in T*C core generation. An explicitly offline comparison changes only complete versus repeated-only oracle source T with identical fourth anchors; it is not a replacement third-state estimator or a new submission.
5. Nonlinear recentering combines independently approximated marginal and bivariate moments. Assigning matching diagonals and symmetry is necessary but insufficient for joint moment consistency. Clipping is a heuristic and has no cumulant-hierarchy guarantee.

   Fourth recentering also subtracts large raw-moment terms. Its identity is exact algebraically, but float32 cancellation can add numerical error when means dominate fluctuations. That is separate from representation and nonlinear-closure errors; this study does not claim that a precision change has been accuracy-tested or would fix the closure.

Even S PSD together with C PSD would not be sufficient: the covariance of combined linear and quadratic features also contains T in off-diagonal blocks. Its necessary PSD condition couples C,T,S (when C is invertible, a Schur complement requires S−Tᵀ C⁻¹T PSD). The requested isolated repair tests S only and cannot establish that stronger condition. It also cannot restore a covariance that has already become indefinite.

No fifth- or higher-order cumulant state is added. Their absence in the truncation does not imply their real moments vanish. The dense oracle G tests **fourth representation and affine transport only**; its one-step response expansion can still be inaccurate and is not an exact nonlinear estimator.

## Moment consistency repair

For unique unordered centered quadratic features, S_(ij),(kl)=Kijkl+Cik Cjl+Cil Cjk must be PSD. The test records eigenvalues, symmetry and κ4_i+2 Cii². It does not test PSD of K: negative cumulants are valid. The separate D,b8 repair adds αJ, J_ijkl=δijδkl+δikδjl+δilδjk, with α prescribed in PROTOCOL.md. On unique quadratic features J has minimum eigenvalue at least one, so the stated α makes S PSD up to rounding. It raises κ4_i by 3α and off-diagonal Qij by α; off-diagonal R stays fixed. At repair time μ,C,T are unchanged, raw fourth moments change, and future means may change. Repair does not establish full realizability or covariance PSD.

## Relation to the papers

Hanin's [hierarchy paper](https://arxiv.org/pdf/2204.01058) organizes connected correlations perturbatively. Equations (3.1), (6.3), (6.5) and (6.6) give cumulant definitions and identities; (6.1) is the law of total cumulance. Its conditional Gaussian construction (3.5)–(3.6) integrates new random weights. Odd-cumulant cancellation, width bounds (3.4), random-covariance expansion (3.9)–(3.10), and ensemble recursions (3.15)–(3.17) do not directly apply to this fixed-weight input average. Exact cumulant definitions and affine identities transfer; ensemble cancellations and width guarantees do not. The preserved 5 October audit contains the earlier equation-by-equation comparison.

[Rontogiannis, Kofidis and Giampouras](https://arxiv.org/abs/2002.09759), §I Eq. (1), study sums of rank-(L_r,L_r,1) three-tensor terms. Our disjoint four-coordinate cores are a different restricted model. Their model selection motivates examining structure and cost, but provides no activation-distribution or propagation guarantee here.

[Compressive Independent Component Analysis](https://doi.org/10.1093/imaiai/iaac016), §2.3.2 Eqs. (2.12)–(2.15), uses multilinear fourth-cumulant transport and an independent-source model with diagonal source cumulants and whitening. Its decomposition and restricted-isometry results concern that model. Independence, diagonalizability and positive source kurtoses are not established here; none is assumed in these estimators.
