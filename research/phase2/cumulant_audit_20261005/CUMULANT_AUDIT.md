# Fixed-weight cumulant propagation audit

5 October 2026. Baseline: `../joint_full_runtime/estimator.py`, SHA-256
`bd4c808d47068dcf2b0515472925598f56c43b2b603a2159b982d4d1f9dbce7e`.
The reported submitted adjusted error 8.62e-8 is supplied by the user; our
local comparisons use identified networks and are separate measurements.
Existing sources, reports, references, archives, evaluator and metering files
are frozen in `preserved.json`. Experimental sources are separate standalone
folders. No AIcrowd submission is performed.

## Applicable mathematics and the ensemble distinction

Hanin's equations (3.1), (6.3), (6.5), (6.6) provide cumulant definitions,
multilinearity and moment/partition identities applicable to our input
distribution. Equation (6.1) is the general law of total cumulance. However,
his conditional Gaussian construction (3.5)-(3.6) integrates independent new
weights at fixed inputs. Conditioning on all our supplied weights removes
that randomness; later activations remain nonlinear functions of random
inputs. Consequently his odd-cumulant cancellation and width bounds (3.4),
random-covariance expansion (3.9)-(3.10), and ensemble recursions (3.15)-(3.17)
cannot be inserted into our recurrence. Their useful lesson is to track
connected index patterns and products at each perturbative order, then check
their sizes and depth accumulation. Even L/n=16/1024 does not certify this
closure's error. [Hanin, pp. 7-12 and 27-28](https://arxiv.org/pdf/2204.01058).

For example, at the first ReLU a zero-mean unit Gaussian gives
κ3(ReLU(Z))=(π+2)/(2π√(2π))>0 for fixed weights/input averaging.
The next linear map transports this nonzero tensor; odd cumulants have no
ensemble symmetry that forces them to vanish.

All equations below are derived for the fixed supplied matrices. Write
z=Wᵀh, μ=E[h], C=Cov(h), T_ijk=cum(h_i,h_j,h_k), and
U_ijkl=cum(h_i,h_j,h_k,h_l). Mean/covariance and complete tensor transports
are exact:

\[
m=W^\top\mu,\quad V=W^\top C W,\quad
T'_{abc}=\sum_{ijk}W_{ia}W_{jb}W_{kc}T_{ijk},\quad
U'_{abcd}=\sum_{ijkl}W_{ia}W_{jb}W_{kc}W_{ld}U_{ijkl}.
\]

Exact transport of an approximate tensor does not recover components erased
by an earlier projection.

## What the baseline represents

Third order uses columns A_r,B_r:

\[
\widehat T_{ijk}=\tfrac13\sum_r(A_{ir}B_{jr}B_{kr}
 +B_{ir}A_{jr}B_{kr}+B_{ir}B_{jr}A_{kr}),
\]
\[
d_{3,i}=\sum_r A_{ir}B_{ir}^2,\qquad
P=\tfrac13[(B\odot B)A^\top+2(A\odot B)B^\top],\quad P_{ij}=\widehat T_{iij}.
\]

These slices and the transformations A←WᵀA, B←WᵀB are exact for these
factors. A newly injected source B=I, A=3Pᵀ with A_ii=d3_i matches every
repeated slice and marginal. It assigns zero to three-distinct indices.
With history=1, older independently transported sources are discarded before
reinjection. Their repeated slices are reencoded in the new source, but their
distinct-index information is lost. The nonlinear multiplication of both
factors by Gaussian gains Φ(m_i/σ_i) is a first-derivative approximation,
not exact nonlinear cumulant transport.

Fourth order stores Q_ij=U_iijj, including Q_ii=U_iiii. Its reconstructed
tensor retains all six permutations of i,i,j,j and the all-equal entries;
3+1, 2+1+1 and four-distinct patterns are absent. Set S=W⊙W,
G=SᵀQS, H_ab=(w_a⊙w_b)ᵀQ(w_a⊙w_b),
D=Sᵀdiag(diag Q)S. Exact transport **within this projection** is

\[
Q'_{ab}=G_{ab}+2H_{ab}-2D_{ab},\qquad
d'_{4,a}=3G_{aa}-2D_{aa}.
\]

The baseline marginal contraction is exact for this paired tensor. Its
off-diagonal transport keeps G and omits 2(H−D). Completing this contraction
still leaves the other physical index patterns missing. In particular the
true marginal transport additionally includes
4∑_{i≠j}U_iiij W_ia³W_ja, plus 2+1+1 and four-distinct contractions.

## Nonlinear expansion and recentering

Let F_k^(r)=E_G[∂^r ReLU(Z)^k] for the Gaussian reference with current m,V.
For raw marginal moments the baseline uses

\[
M_k\approx M_{k,G}+d_3F_k^{(3)}/6+d_4F_k^{(4)}/24,\quad k=1,\ldots,4.
\]

This is a truncated, Gaussian-smoothed cumulant expansion. Clipping each
correction to ±0.25σ^k, flooring variances to 1e-12, clipping correlations and
using two-node Gaussian quadrature are heuristics/numerical approximations.
They do not define or guarantee a positive joint density.

For a bivariate raw covariance plane with responses f,g, the non-Gaussian
part retains the following products of marginal Gaussian responses:

\[
\begin{aligned}
\Delta\operatorname{Cov}(f_i,g_j)={}&Q_{ij}F_f^{(2)}F_g^{(2)}/4
 +P_{ij}F_f^{(2)}F_g^{(1)}/2+P_{ji}F_f^{(1)}F_g^{(2)}/2\\
 &+P_{ij}V_{ij}F_f^{(3)}F_g^{(2)}/2
  +P_{ji}V_{ij}F_f^{(2)}F_g^{(3)}/2\\
 &+P_{ij}P_{ji}F_f^{(3)}F_g^{(3)}/4
  +d_{3,i}V_{ij}F_f^{(4)}F_g^{(1)}/6
  +d_{3,j}V_{ij}F_f^{(1)}F_g^{(4)}/6.
\end{aligned}
\]

The code applies this to Cov(h,h), Cov(h²,h), Cov(h²,h²). Gaussian anchors
are Price-integral identities approximated by quadrature. The response
products above additionally truncate dependence on V_ij; they are not a
complete bivariate cumulant expansion at general correlation.

Define A21_ij=Cov(h_i²,h_j), A22_ij=Cov(h_i²,h_j²), now using post-ReLU means
and covariance. The implemented recentering identities are correct:

\[
P_{ij}=A21_{ij}-2\mu_i C_{ij},
\]
\[
Q_{ij}=A22_{ij}-2\mu_j A21_{ij}-2\mu_i A21_{ji}
       +4\mu_i\mu_jC_{ij}-2C_{ij}^2.
\]

The C_ii C_jj Gaussian pairing was already removed by the covariance in
A22; subtracting it again would be an error. The remaining two pairings
require −2C_ij². On the diagonal, κ4=M4−4μM3−3M2²+12μ²M2−6μ⁴.
Independent finite-distribution tests verify these identities, including
nonzero means. The estimator explicitly replaces C, P, Q diagonals with
their corrected marginal values and symmetrizes C,Q. These operations enforce
those slices, not PSD or compatibility of all approximate moments with a
single joint law. No full PSD guarantee is supplied.

## Ranked omissions

1. **Mixed fourth patterns and new distinct third sources.** No retained
   plane reconstructs U_iiij. For two standard zero-mean Gaussians with small
   correlation c, elementary Price differentiation after ReLU gives
   U_iijj=c/(2π)+O(c²), whereas
   U_iiij=(3/8−3/(4π))c+O(c²). The discarded 3+1 pattern is already of the
   same order as retained paired fourth information in the first layer.
   Likewise, three-distinct post-ReLU third cumulants are generated even from
   Gaussian preactivations: their leading connected terms are the sum over
   three roots of F1_root^(2) F1_other1^(1) F1_other2^(1)
   V_root,other1 V_root,other2. Reinjection from P alone erases these terms.
   Sparse-root or covariance-mode factors are a more direct future target
   than adding cumulants of order five through eight.
2. **Finite third memory.** Distinct components generated by affine transport
   survive one generation and are then discarded. The tested compression
   retains 16 older CP columns by ||A_r||₂||B_r||₂², a triangle-inequality
   upper bound on each symmetric term's Frobenius norm. Selected terms are
   unchanged; this is column pruning, not an invariant rotation/SVD of the
   factors. The discarded bound does not guarantee readout accuracy. Their
   repeated contributions are subtracted before injecting the new residual
   source, preserving the current P,d3 slices while retaining some history.
3. **Incomplete paired fourth transport.** The exact missing 2(H−D) is
   identifiable. Earlier complete width-32 experiments improved means by
   1.51% but worsened variance by 0.52%; that cannot settle width 1024. The
   present signed-mode ablation approximates Q_off by λuuᵀ, removes its
   spurious input diagonal, and adds
   2λ[(Wᵀdiag(u)W)⊙(Wᵀdiag(u)W)−Sᵀdiag(u²)S]. Eight power iterations
   are an approximate mode finder. Restore the original paired output
   diagonal. Only early source transitions (zero-based layers 1,4) are used
   to keep additional FLOPs below the available headroom.
4. **Lower-order interactions and response consistency.** A formal expansion
   E_G[(1+D3+D4+D3²/2+…)f], D3=T:∇³/6, D4=U:∇⁴/24, contains products
   as well as cumulants. No convergence or Hanin remainder bound is claimed.
   The mixed operator P_ij∂i²∂j/2+P_ji∂i∂j²/2 generates both the retained
   P_ij P_ji term and omitted P_ij² F_f^(4)F_g^(2)/8 and its transpose.
   These squares are tested as a fixed analytical ablation in all three
   covariance planes. The marginal κ3²F^(6)/72 term is absent, but existing
   oracle data found negligible local mean benefit. A separate ablation adds
   the missing leading pure marginal κ4_i V_ij F_f^(5)F_g^(1)/24 and transpose
   consistently to all three planes. These response derivatives do not add
   fifth/sixth cumulant states. First-order U_iiij F_f^(3)F_g^(1)/6 terms and
   further cross products remain unrepresented.

With measured standardized skewness s3 and excess kurtosis s4, compare s3²
and |s4| directly rather than assigning the fixed-weight problem the paper's
ensemble scaling. A term's per-entry size also cannot justify deletion
without examining its contraction with the supplied downstream weights.

## Controlled experiments and local/accumulated error

`protocol.json` froze three single-change candidates before targets were
loaded. Screening uses identical exposed IDs 0-3. The initial ≥2% advancement
threshold was not met; `confirmation_amendment.json` records the decision to
confirm the inexpensive third-memory result anyway, before loading IDs 8-11.
No parameters were changed. These public confirmation networks were excluded
from this selection but exposed in older studies. `third_square_protocol.json`
records the one additional mathematical ablation; it receives no coefficient
search or combination with the other candidates. Fresh seeds 20261051,20261052
are evaluated only after one candidate is frozen. Every inference implementation
imports only flopscope, flopscope.numpy and the official estimator interface;
reference samples, targets and fits stay in separate offline scripts.

The current evaluator is whestbench 0.16.1 / flopscope 0.12.1. With lambda=0,
score=MSE·max(0.1,F/2⁴¹), and failures zero predictions and force multiplier 1.
The 0.1 floor ends at 219.902 billion FLOPs, leaving 23.585 billion (12.014%)
above the frozen baseline. Default gates are 0.4 s residual, 120 s prediction
wall, 5 s setup and 8192 MiB process memory. Four threads are used throughout.
In-process times are descriptive only. Official CLI reports and the official
CLI's unchanged SubprocessRunner/scoring function establish execution eligibility.

The offline diagnostic reuses four existing independent 65,536-draw streams
per network; no new data are generated for the initial local-error study.
On IDs 0-3, the full baseline has final MSE 9.42560e-7 and incoming relative
variance RMSE 0.9763%. Supplying true incoming means reduces final local
readout error to 1.71460e-7; repairing both means/variances leaves 6.08701e-8
with the propagated shape cumulants. Supplying all oracle m,v,κ3,κ4 gives
cross-stream risk −1.07259e-9: its remaining bias is unresolved at the sample
precision, not negative true error. The oracle Gaussian readout has risk
1.93631e-7. These controls are not an additive or causal error decomposition.
They isolate a substantial state/shape propagation gap from the marginal
ReLU formula. Compressed history reduces variance RMSE only to 0.9745%.
No sampled-state reset, target or correction is fed into an eligible estimator.

Measured result tables, official gates, recommendation and preservation
checks are in `RESULTS.md` and the adjacent JSON reports.
