import csv,hashlib,json,math
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
def read(name):return json.loads((R/name).read_text())
def main():
    dev={}
    for path in ('early_full_width.json','centered_full_width.json','public_v25_full_width.json','public_v29_full_width.json'):dev.update(read(path)['summary'])
    held=read('confirmation.json');names=list(held['summary']);local=read('local_reference.json');prop=read('propagated_diagnostics.json');public=read('public_propagation_diagnostics_v2.json');birth=read('birth_diagnostic.json')
    gates={};rows=[]
    for name in names:
        gate_name=name+'_gate.json'
        if not (R/gate_name).exists():
            gate=read('../fourth_representation_20261007/baseline_snapshot_gate.json');label='preserved prior unchanged-source gate'
        else:gate=read(gate_name);label='current gate'
        rr=[x for r in gate['records'] for x in r['summary']['per_mlp']];fails=sum(r['summary']['n_failed_mlps'] for r in gate['records']);gates[name]={'failed':fails,'max_residual':max(x['residual_wall_time_s'] for x in rr),'max_wall':max(x['wall_time_s'] for x in rr),'flops':[x['flops_used'] for x in rr],'label':label}
        hs=held['summary'][name];rows.append(dict(method=name,development_raw=dev[name]['raw_mse'],development_nominal_adjusted=dev[name]['nominal_adjusted'],confirmation_raw=hs['raw_mse'],confirmation_nominal_adjusted=hs['nominal_adjusted'],screen_mean_flops=hs['flops'],screen_mean_wall_s=hs['wall_s'],screen_mean_residual_s=hs['residual_s'],official_n_failed=fails,official_max_residual_s=gates[name]['max_residual'],official_max_wall_s=gates[name]['max_wall']))
    with (R/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    extra=read('public_v29_confirmation_gate.json');er=[x for r in extra['records'] for x in r['summary']['per_mlp']];ef=sum(x['summary']['n_failed_mlps'] for x in extra['records']);mem=read('public_v29_confirmation_gate_memory.json');peak=max(x['peak_working_set_bytes'] for x in mem['processes']);allowed=ef==0 and peak<8192*1024*1024
    hr=[x for x in held['records'] if x['method']=='public_v29'];assert [x['flops_used'] for x in er]==[x['flops'] for x in hr],'Official cold/steady FLOPs differ from screen; rescore held-out rows before promotion.'
    base=held['summary']['baseline_snapshot']['nominal_adjusted'];gain=100*(1-held['summary']['centered']['nominal_adjusted']/base);factor=base/held['summary']['public_v29']['nominal_adjusted']
    paired=[]
    for name in ('unclip34','centered','coherent','public_v29'):
        bb=[x['nominal_adjusted'] for x in held['records'] if x['method']=='baseline_snapshot'];nn=[x['nominal_adjusted'] for x in held['records'] if x['method']==name];delta=np.array(nn)-bb;mean=float(delta.mean());half=3.182446*float(delta.std(ddof=1))/2;paired.append({'method':name,'paired_adjusted_difference_candidate_minus_incumbent':mean,'lower95_t3':mean-half,'upper95_t3':mean+half,'n_networks':4})
    (R/'paired_confirmation.json').write_text(json.dumps(paired,indent=2))
    text=f'''# Focused cumulant generation and public-method reproduction, 2026-10-07

The original estimator is improved by the centered marginal repair: **{gain:.2f}% lower held-out adjusted objective** at essentially unchanged FLOPs. V29 is the much stronger complete reproduction: **{factor:.2f}x lower nominal held-out adjusted objective**. V25 fails the current official subprocess residual cap despite good raw error. The original incumbent and all prior results/archives remain unchanged.

Development uses previously exposed public mini IDs0/1. The six sources/defaults were frozen before reading reserved mini IDs80..83. No sweep, new fitted coefficient, or reference sampling was used. Four held-out networks are a small local confirmation, not a leaderboard result. Paired differences and approximate t3 intervals are in paired_confirmation.json. Coherent damping was slightly worse in development and slightly better on confirmation; that small reversal does not justify choosing a new guard over the centered algebraic repair.

## Measured comparison

Adjusted objective = final-layer MSE * max(0.1, FLOPs/2^41). Resource failures invalidate eligibility; the table's nominal objectives are retained for comparison even for V25. The compute floor ends at219.902 billion FLOPs. Screens are in-process, four threads; official gates use the unchanged official subprocess CLI/default caps, separate smoke seeds and 100-sample targets for execution only.

| Method | Dev raw MSE | Dev nominal adjusted | Held-out raw MSE | Held-out nominal adjusted | Mean screen GFLOPs | Mean screen seconds | Official max residual seconds | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
'''
    for r in rows:text+=f"| {r['method']} | {r['development_raw']:.6g} | {r['development_nominal_adjusted']:.6g} | {r['confirmation_raw']:.6g} | {r['confirmation_nominal_adjusted']:.6g} | {r['screen_mean_flops']/1e9:.6f} | {r['screen_mean_wall_s']:.3f} | {r['official_max_residual_s']:.6f} | {'FAIL '+str(r['official_n_failed'])+'/2' if r['official_n_failed'] else 'PASS 2/2'} |\n"
    text+=f'''
V29 has a real cold/steady distinction: held-out FLOPs by call are {', '.join(f"{x['flops']/1e9:.6f}" for x in hr)} billion. Its additional unchanged official four-network run at seed20261171 has **{ef} failures**, max residual **{max(x['residual_wall_time_s'] for x in er):.6f}s**, max wall **{max(x['wall_time_s'] for x in er):.3f}s**, and largest observed individual process resident peak **{peak/1024**3:.3f}GiB**. The four official FLOP counts match the held-out screen exactly. Extended execution eligibility under the measured checks: **{allowed}**. Windows memory observation is read-only; it does not reproduce Linux RLIMIT_AS or establish safety across a 100-network worker. Full memory logs include both per-process lifetime peaks and sampled simultaneous process-tree RSS.

The first V29 development screen instantiated a new estimator for each network and omitted setup; its cold-only average48.18s and residual0.615s are preserved, but are not the suite lifecycle. Confirmation calls setup once and reuses the estimator; source/state semantics and cold/steady compute are now explicit. The cohorts also differ, so these observations do not isolate a causal timing overhead. The official subprocess output alone grants execution eligibility. The preserved earlier matched local/subprocess timing experiment found close agreement for our incumbent and tensor candidates; no claim is made that IPC explains their earlier custom-harness discrepancies.

## Nonlinear generation versus propagation

Phase2 references are 100 width1024/depth16 networks with1e8 fresh-input moment samples and copied official1e9 mean targets. The cited1000-network higher-moment collection is Phase1, width256/depth32, also1e8 moment samples. Downloaded only four Phase2 moment files (dev0/1, reserved80/81) and one official confirmation shard. Reference IDs, full moment-file LFS hashes, baked means (exact equality), and initial Gaussian variance are checked in reference_alignment.json. These files remain offline and outside all archives.

With *reference incoming marginal statistics*, average local fourth-cumulant MSE is **{local['summary']['baseline_snapshot']['mse_mean_variance_third_fourth'][3]:.8g} -> {local['summary']['centered']['mse_mean_variance_third_fourth'][3]:.8g}** after repair. Raw M4 clipping fires on **{100*local['summary']['baseline_snapshot']['average_raw_clip_fractions'][3]:.3f}%** of coordinates; M1/M2 never clip on this diagnostic. Third raw corrections clip only very rarely. Local mean/variance predictions are unchanged by removing M3/M4 guards. The centered identities are verified against independent Gaussian quadrature, unclipped raw algebra, and the positive identity limit through mean1000 (three meaningful tests passed).

During actual recurrence on development, average post marginal fourth MSE drops from **{prop['summary']['baseline_snapshot']['mean_post_mse'][3]:.8g} to {prop['summary']['centered']['mean_post_mse'][3]:.8g}**. Incoming third-cumulant MSE remains **{prop['summary']['centered']['mean_pre_mse'][2]:.8g}**; variance MSE is essentially unchanged. Thus fixing the large local kurtosis defect does not repair the rest of the represented tensor or its propagation. All variants remain finite. The minimum necessary marginal moment slack K4+2*variance^2 improves from **{prop['summary']['baseline_snapshot']['min_marginal_moment_slack']:.8g} to {prop['summary']['centered']['min_marginal_moment_slack']:.8g}**; a small residual negative slack remains. Neither the centered repair nor coherent attenuation proves joint realizability or PSD covariance.

## Ranked omissions and next mathematical target

1. **Independent raw-moment stabilization was defective and is now isolated/fixed.** Exact positive-ReLU recentering requires deltaM4=4*m*kappa3+kappa4. Calculate centered corrections and recenter consistently; do not bound raw M3/M4 independently. The coherent layer attenuation scales all incoming third/fourth quantities and their nonlinear interactions consistently, but shows no convincing additional benefit here.
2. **Our third sources omit distinct-index content at birth and then discard old history.** For Gaussian input to a ReLU, the leading three-distinct third cumulant is the sum of three covariance-tree terms w2_i*w1_j*w1_k*Cij*Cik plus cyclic permutations. Our Sym(A,I,I) birth is zero there. The public Sym(3A,I,A*diag(w2)) birth includes it and keeps compressed older sources. Existing four-stream small-network references reduce distinct-index birth risk from **{birth['summary']['distinct_birth_risk_zero']:.8g} to {birth['summary']['distinct_birth_risk_gaussian_tree']:.8g}** with that leading tree. Its next-weight marginal-contraction risk is within reference noise (a negative cross-stream estimate is not negative physical error). This is small-network mechanism evidence, not a tested new width1024 candidate.
3. **The public contracted fourth core and lower-cumulant interactions were absent from our earlier comparisons.** Its G is not paired Q. G feeds new third sources; its mean table includes third-squared and third-fourth products. Complete V25/V29 reproduction is therefore essential; Q/R/block tests do not test the same mechanism. The source map, conditional exact affine identity, empirical lambda rule, projection losses, and licensing are in MATHEMATICS.md. Hanin's cumulant identities transfer to fixed weights; his random-weight cancellations and width guarantees do not.
4. **For the reproduced public method, accumulated mean/variance error is the strongest remaining diagnostic lead.** On development's final layer, its marginal response formula gives MSE **{public['summary']['final_mse_controls']['propagated_formula']:.8g}**. Replacing just incoming mean/variance with reference values gives **{public['summary']['final_mse_controls']['oracle_mean_variance']:.8g}**; replacing only incoming third gives **{public['summary']['final_mse_controls']['oracle_third']:.8g}**, only effective fourth **{public['summary']['final_mse_controls']['oracle_fourth']:.8g}**, and all four **{public['summary']['final_mse_controls']['oracle_all']:.8g}**. Fresh-reference mean noise is approximately0.7–1.0e-9 on these networks, so the all-oracle result is near that scale. These are one-step offline interventions, not deployable improvements or additive error attribution. Earlier fourth-core feedback can affect all subsequent means/covariances. Investigate covariance update consistency and the effect of compressed third births on that update before adding tensor formats or higher cumulant orders.

The public diagnostic formula matches logged means within {public['summary']['formula_parity_maxabs']:.3g} absolute. C is captured after its diagonal is zeroed in place; v2 compares covariance **off diagonal**. The v1 diagnostic JSON/script remain preserved but their covariance_mse includes a known zero-diagonal artifact; their other metrics are unaffected. No reference target enters any estimator.

## Recommendation and artifacts

Use `centered/estimator.py` as the repaired version of our original approximation. Its negligible difference from unclip34 is floating-point arithmetic; the new formulation preserves the translation identity. Keep the frozen original as the comparison anchor. Use the complete public V29 as the stronger research benchmark, subject to the execution result above; do not represent a reproduction as our original algorithmic contribution. The [October5 organizer clarification](https://discourse.aicrowd.com/t/eligibility-of-solutions-built-on-another-participants-mit-licensed-forum-code-4-6/18240) requires substantial original changes for prize consideration even with licensed attribution. Public reported September/October scores are dated participant reports, not verified current rankings.

The original repair is packaged after held-out improvement, official subprocess success, official contract validation and package validation. The archive contains estimator.py, LICENSE, PROVENANCE.md and manifest.json only. A public reproduction archive, if all extended checks pass, is separately labeled REPRODUCTION and retains the author's MIT notice and exact source. Archive manifests record absolute paths, hashes and contents. No AIcrowd upload or forum message is made.

Reproduce with REPRODUCE.md. Detailed measurements: comparison.csv, confirmation.json, local_reference.json, propagated_diagnostics.json, public_propagation_diagnostics_v2.json, birth_diagnostic.json, all *_gate*.json, and source hashes in confirmation_freeze.json. All654 pre-existing study/archive files and130 SDK source files are checked against prior hashes in preservation_verification.json.
'''
    (R/'COMPARISON.md').write_text(text,encoding='utf-8');print('Wrote report',gain,factor,'extended eligible',allowed)
if __name__=='__main__':main()
