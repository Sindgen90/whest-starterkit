"""Build readable tables and a complete matrix from preserved raw evidence."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text())
def fmt(x):return f'{x:.6g}' if isinstance(x,(int,float)) else str(x)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(fmt,r))+' |' for r in rows])
def main():
    if not (ROOT/'clipping_diagnostic.json').exists():
        from clipping_diagnostic import main as diagnose_clipping
        diagnose_clipping()
    oracle=read('oracle_results_v2.json');small=read('recurrence_results_v2.json');large=read('full_width_results.json');sel=read('selective_full_width_results.json');compat=read('compat_full_width_results.json')
    matrix=[]
    for r in oracle['records']:matrix.append(dict(stage='oracle',family=r['family'],block_size=r['block_size'],width=r['n'],seed=r['seed'],layer=r['source_layer'],partition=r['partition'],status='run',evidence='oracle_results_v2.json'))
    for r in small['records']:matrix.append(dict(stage='recurrence',family=r['method'],block_size='',width=r['n'],seed=r['seed'],layer='all',partition=small['partition'],status=r['status'],evidence='recurrence_results_v2.json'))
    exact=read('exact_transport_controls.json')
    for r in exact['records']:matrix.append(dict(stage='exact affine control',family=r['method'],block_size=8,width=r['n'],seed=r['seed'],layer='all',partition=small['partition'],status=r['status'],evidence='exact_transport_controls.json'))
    for net in read('recurrence_results.json')['offline_third']:
        for r in net['records']:matrix.append(dict(stage='offline third control',family='complete vs repeated T',block_size=r['block_size'],width=net['n'],seed=net['seed'],layer=r['source_layer'],partition=small['partition'],status='run',evidence='recurrence_results.json'))
    for filename in ('kernel_profiles.json','kernel_profiles_v2.json'):
        for r in read(filename)['records']:matrix.append(dict(stage='bounded kernels',family=r['kernel'],block_size='',width=1024,seed=20261007,layer='one call',partition='',status=r['status'],evidence=filename))
    local=read('local_nonlinear_results.json')
    for r in local['records']:matrix.append(dict(stage='local nonlinear',family=r['family'],block_size=8 if r['family'].endswith('8') else '',width=r['n'],seed=r['seed'],layer=r['source_layer'],partition=small['partition'],status='run',evidence='local_nonlinear_results.json'))
    for r in read('clipping_diagnostic.json')['records']:matrix.append(dict(stage='offline guard diagnosis',family='G clipped vs unclipped M3/M4',block_size='',width=r['n'],seed=r['seed'],layer=r['source_layer'],partition='',status='run',evidence='clipping_diagnostic.json'))
    full=[]
    for name in ('baseline_snapshot','qr','block8','block16','qblock8','qblock16','qrblock8','qrblock16','selective_block8','selective_block8_compat'):
        cohort=compat if name=='selective_block8_compat' else (sel if name=='selective_block8' else large)
        rows=[r for r in cohort['records'] if r['method']==name]
        run=[r for r in rows if r['status']=='run'];gate=ROOT/(name+'_gate.json')
        official=read(gate.name) if gate.exists() else None
        if official:
            summary=official['records'][0].get('summary',{});failed=summary.get('n_failed_mlps',2)
            causes=[{'budget_exhausted':'FLOPs','time_exhausted':'wall','residual_wall_time_exhausted':'residual','combined_budget_exhausted':'combined','error':'error'}.get(k,k) for k,v in summary.get('failure_breakdown',{}).items() if v]
            status='run/pass' if failed==0 else 'failed '+','.join(causes)
            per=summary.get('per_mlp',[]);wall=max((x['wall_time_s'] or 0 for x in per),default=0);res=max((x['residual_wall_time_s'] or 0 for x in per),default=0)
            detail=json.loads(Path(official['records'][0]['report']).read_text());real_score=detail['results']['adjusted_final_layer_score']
        elif name=='qrblock16':status='infeasible FLOPs';wall=res=real_score=None
        else:status='failed screen' if not run else 'no official eligibility';wall=res=real_score=None
        raw=float(np.mean([r['raw_mse'] for r in run])) if len(run)==2 else None
        adjusted=float(np.mean([r['nominal_adjusted'] for r in run])) if len(run)==2 else None
        flops=run[0]['flops'] if run else (read('qrblock16_infeasible.json')['lower_bound_flops'] if name=='qrblock16' else None)
        if flops is None and official:flops=max((p['flops_used'] for p in per),default=None)
        full.append([name,status,raw,adjusted,None if flops is None else flops/1e9,wall,res,real_score])
        matrix.append(dict(stage='full width',family=name,block_size='',width=1024,seed='public 0,1 / official 20261071',layer='all',partition='correlation' if 'block' in name else '',status=status,evidence=name+'_gate.json' if gate.exists() else ('qrblock16_infeasible.json' if name=='qrblock16' else 'full_width_results.json')))
    with (ROOT/'experiment_matrix.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['stage','family','block_size','width','seed','layer','partition','status','evidence']);writer.writeheader();writer.writerows(matrix)
    lines=['# Fourth-order representation study — 7 October 2026','','**Recommendation: retain the incumbent.** No candidate met the frozen adjusted-score advancement criterion. The user-reported leaderboard value 8.62e-8 is not used as a local target. Sources, prior results and archives were preserved; the preservation check is recorded separately. No AIcrowd upload.','','## Paired full-width development and unchanged official gates','',table(['Method','Execution status','Public raw MSE','Nominal adjusted','GFLOPs','Official max wall s','Official max residual s','Official smoke adjusted'],full),'',
    'Public means are baked 1e9-draw targets on identical IDs 0,1. Nominal adjusted = raw MSE × max(0.1,F/2^41), and is **not an eligible score for a failing method**. Official smoke scores use 100 samples on different seed-20261071 networks and are execution evidence only; failure uses the official fallback predictions/scoring. Do not compare smoke scores to baked-target scores. The full per-network and layerwise means remain in the raw JSON/NPZ files. Q+R/block16 has no full-width accuracy result: its GFLOPs entry is a mandatory-kernel lower bound. Q/block16 exhausted its budget before final output: its GFLOPs entry is partial consumed compute, and no final MSE is claimed.','','Current installed versions: whestbench 0.16.1, flopscope 0.12.1. Unchanged spec: F=2^41, wall 120 s, residual 0.4 s, memory 8192 MB, four threads, lambda=0. Setup timeout 5 s and predict-timeout field 30 s are recorded in environment.json; actual official outcomes, rather than assumptions about those fields, govern eligibility. The score floor ends at 219.902325555 GFLOPs, 12.014% above the 196.317442558-GFLOP incumbent.','','## Representation, transport and nonlinear errors','',
    'The common-oracle study has 372 cells: widths 16/32, two independent weight seeds each, depth 16, source checkpoints 0/7/14, four independent streams of 32768 inputs each. It compares A–G and every requested block-size/partition combination. Four-stream cross-products of linear errors remove k-statistic sampling noise in expectation. Delete-one-stream jackknife intervals use t3 and are approximate with only four streams. One-step nonlinear quantities retain plug-in bias; their intervals must not be read as rigorous confidence bounds on the exact nonlinear closure. Network-to-network variation is separate from input-sampling uncertainty.','']
    orows=[]
    for family in 'ABCDEFG':
        rr=[r for r in oracle['records'] if r['family']==family]
        orows.append([family]+[float(np.mean([r['cross_stream_risk'][key] for r in rr])) for key in ('d4','Q','R','actual_weight_probes','one_step_mean','one_step_cov')])
    lines += [table(['Oracle family','d4 risk','Q risk','R risk','Weight-probe risk','One-step mean risk','One-step C risk'],orows),'',
    'The D/E/F rows average all requested sizes and partition controls; this is a descriptive overview, not a weighted scientific ranking across units or networks. Detailed normalized per-network/layer comparisons and uncertainty are in oracle_results_v2.json. A uses the legacy Gram plane; its general four-linear probes refer to its underlying paired-Q representation, since the legacy Gram approximation alone does not define a complete transported tensor. B retains exactly the same source projection as A, so its marginal fourth contraction is identical. C changes representation. G removes fourth representation/transport loss but still has nonlinear response-expansion error.','',
    'The frozen clustering algorithm uses only baseline-propagated first-layer C: smallest available pivot, highest absolute normalized-correlation neighbors, stable index tie breaks, fixed through depth. The predeclared aggregate E/F contraction criterion chose **correlation**, with normalized risks contiguous=0.358502, random=0.377848, correlation=0.283180. No oracle cumulants choose individual clusters; no coefficient is fitted.','',
    'Completing paired-Q transport does not recover discarded 3+1, 2+1+1 or four-distinct information. Q+R markedly reduces actual-weight contraction errors, while full cores can reduce them further. These oracle gains do not guarantee recurrence gains. The complete-fourth oracle still has one-step nonlinear error; generic core generation and bivariate response truncation are independent limitations. See AUDIT.md for exact formulas, multiplicities, overlap subtraction, and the specified nonlinear closure.','','## Small recurrent ablations','']
    sr=[]
    for name,s in small['summary']['methods'].items():
        rr=[r for r in small['records'] if r['method']==name and r['status']=='run']
        sr.append([name,len(rr),s['final_risk'],100*s['improvement_fraction']])
    lines += [table(['Method','Completed networks','Final cross-stream mean risk','Improvement %'],sr),'',
    'All primary recurrent runs retain full covariance and the unchanged third-source algorithm. The third tensor can change through changed fourth-informed moments, but its factorization, finite history and reinjection are unchanged. Small-width aggregate gains are highly variable across four networks and are not full-width performance guarantees. The exact-Q control changes paired off-diagonal transport while omitting R responses; it is distinct from the common-oracle physical-Q test B.','',
    'The explicitly offline missing-third diagnostic uses oracle fourth anchors and compares complete versus repeated-only source T in generic core generation. Its aggregate contraction risks were 0.0522024 versus 0.0475430, respectively. More third information did not improve this truncated core update on average. This narrow result does not prove older CP history has no value; it prevents attributing the present failures solely to missing third entries. No oracle-filled state enters an estimator.','','## Moment consistency and numerical stability','',
    'Full per-layer small-network S minima, violation sums, symmetry, marginal slack and covariance eigenvalues are in recurrence_results_v2.json. The separate D,b8 repair makes the quadratic-feature matrices PSD at the repaired state but is catastrophically worse in recurrence: aggregate final mean risk 503441.969 versus 0.00959520 uncorrected. Its covariance became indefinite, reaching minimum eigenvalue −1.52062e8 on one network. Local fourth-moment consistency does not repair an inconsistent lower-order closure. Negative K itself is valid and was never projected PSD.','',
    'Initial small failures and legacy kernels are retained in recurrence_results.json/source_history. Some batched einsums triggered incorrect symmetry inference across batch axes and were replaced by equivalent metered matrix products. Later float32 affine covariance asymmetry was handled by explicit symmetrization of WᵀCW, without changing the SDK or disabling validation. The corrected small study completed all 52 cases. Twelve independent equation tests pass, covering Gaussian-tree leading order, CP-core contraction, the Gaussian B31 anchor and the raw-guard counterexample.','',
    'large_covariance_results.json is an explicitly exploratory public-ID-0 diagnostic with fresh 4×8192 reference inputs and checkpoints 0/7/14. It reports covariance cross-stream risks, approximate sampling intervals, eigenvalues and block consistency. Failed budget cases retain only completed checkpoints. It is separate from the paired two-network final-mean development screen.','','## Cost, memory and the single selective schedule','',
    'Initial block16 all-to-all transport took 19.167 s per call. The equivalent batched-matrix transport took 1.828 s, versus 0.505 s for block8. It increased the metered copy/reshape arithmetic and all such work is counted. Process peak working set in that fresh bounded profiler was 3.403 GB; small storage was not treated as proof of cheap propagation. Original and revised kernel profiles, individual operations/calls, copies, construction, nonlinear updates, third restriction, and optional consistency repair are retained.','',
    'Global QR/block16 mandatory affine kernels alone require 2,279,623,227,392 FLOPs over 14 state transitions, exceeding the 2,199,023,255,552 cap before covariance, third state, nonlinear updates, preparation or final contractions. Tagged-Q checks give the same relevant kernel costs. This implementation is infeasible; no dense 1024^4 allocation was made.','',
    'Process memory observations in *_gate_memory.json are read-only OS process-tree observations every 0.5 s, including helper/CLI/backend/worker. Each process supplies its lifetime peak; sampled tree sums can miss short-lived children and are not exact tensor-allocation accounting. In-process screen peak working sets are cumulative within a cohort process and are explicitly labeled. Shape bounds and fresh bounded profiles complement these observations. Neither timing type replaces official residual/wall outcomes.','',
    'Only one selective schedule was tested: D,b8 enriches source layers 0 and 7, then generates the next global Q from current full-covariance/marginal/bivariate moments. Discarded cross-block source cumulants are not copied back. The schedule and family were frozen in selective_selection.json before its full-width confirmation; D,b8 had the strongest small mean improvement. A vectorized partition implementation preserves the same greedy rule. No schedule or damping sweep was run. Terminal-only marginal contractions avoid unused Q/R/core work; a small independent comparison differed by at most 3.73e-9 in returned means.','',
    'The packaged selective_block8_compat source caps block width at n for the official width-four/depth-two contract check. It has identical arithmetic, outputs and metered FLOPs at width1024; compatibility.json and the separate paired full-width/gate results document this necessary contract fix. It is a packaging version of the same schedule, not another selected experiment. The original source and its gate remain preserved.','',
    'No primary candidate met the ≥2% paired adjusted improvement criterion on both development networks; fresh held-out weight-seed confirmation and a promising-method random-partition control were therefore not triggered. This is a deliberate frozen stopping rule, not evidence of held-out accuracy.','',
    '## Packaging and preservation','']
    archives=[]
    for p in ROOT.glob('*_archive.json'):
        a=read(p.name);archives.append([a['archive'],a['sha256'],a['source_sha256']])
    if archives:lines += [table(['Validated EXPERIMENTAL archive (not recommended)','Archive SHA-256','Estimator SHA-256'],archives),'']
    else:lines += ['No new archive passed all required official execution and package checks. Failed or diagnostic-only sources are not submission-ready.','']
    lines += ['The incumbent archive remains `submissions/phase2_joint_full_runtime_20261002.tar.gz`, SHA-256 `89035799f6acfe5dd14948fbedefd37ea2bc35472ce2ce72518c4485676782b3`. Incumbent source SHA-256 `bd4c808d47068dcf2b0515472925598f56c43b2b603a2159b982d4d1f9dbce7e`. The 528-file pre-study preservation ledger covers previous research, archives and official SDK/metering sources; preservation_verification.json records the final check.','',
    'Source hashes and raw-file hashes are in inventory.json; full statuses are in experiment_matrix.csv. Reproduction commands and seed rules are in REPRODUCE.md and PROTOCOL.md. No new best was promoted and nothing was uploaded.']
    extra=['','## Additional isolated controls','', 'Exact-transport controls retain the identical nonlinear update but complete the global paired-Q off-diagonal contraction. These are small-network diagnostics only; no full-width source was selected from them.','']
    erows=[]
    for name in ('qr','qblock8','qrblock8'):
        rr=[r for r in exact['records'] if r['method']==name and r['status']=='run']
        erows.append([name,len(rr),float(np.mean([r['mean_risks'][-1] for r in rr])) if rr else None])
    extra += [table(['Exact affine model','Completed networks','Final mean risk'],erows),'','Local bivariate fourth-generation test: common oracle mu,C,T and model-specific fourth input, compared against independent post-ReLU references. This separates generation of marginal/Q/R cumulants from accumulated state error. Risks are nonlinear plug-in diagnostics, with four-stream intervals in local_nonlinear_results.json.','']
    lr=[]
    for name in ('A','C','D8','E8','F8','G'):
        rr=[r for r in local['records'] if r['family']==name]
        lr.append([name]+[float(np.mean([r['risks'][k] for r in rr])) for k in ('post_d4','post_Q','post_R')])
    extra += [table(['Oracle input family','Generated marginal K4 risk','Generated Q risk','Generated R risk'],lr),'','Full-width exploratory covariance/stability checks on public ID0 only; partial checkpoints for budget failure.','']
    cr=[]
    for r in read('large_covariance_results.json')['records']:
        dd=r['diagnostics'];last=dd[-1] if dd else {}
        cr.append([r['method'],r['status'],last.get('layer'),last.get('cov_cross_stream_risk'),min((d['cov_min_eigenvalue'] for d in dd),default=None),min((d.get('S_min_eigenvalue',float('inf')) for d in dd),default=None)])
    extra += [table(['Method','Observer status','Last checkpoint','Last C risk','Min C eigenvalue','Min block S eigenvalue'],cr),'','## Same-seed local/subprocess timing','']
    tr=[]
    for r in read('timing_pair_results.json'):
        for mode in ('local','subprocess'):
            s=r[mode]['summary'];pp=s['per_mlp']
            tr.append([r['method'],mode,s['n_failed_mlps'],float(np.mean([p['flops_used'] for p in pp]))/1e9,max(p['wall_time_s'] for p in pp),max(p['residual_wall_time_s'] for p in pp)])
    extra += [table(['Source','Official runner','Failed/2','GFLOPs','Max wall s','Max residual s'],tr),'', 'These CLI runs use identical source hashes, seed20261071, deterministic contest generation, sample count, thread limit and caps. Only the runner differs. Matched official local/subprocess timings agree closely: the large gap in the earlier custom screen does not persist under these conditions. That screen used a different cohort/harness, so the evidence does not justify attributing its entire discrepancy to IPC. No framework or timing accounting was changed. Only subprocess outcomes gate packaging/promotion.','']
    clip=read('clipping_diagnostic.json')
    extra += ['## Recentring and raw-moment clipping — offline diagnosis','',
    'In the positive-support identity limit of ReLU, the exact raw corrections are ΔM3=κ3 and ΔM4=4μκ3+κ4. The incumbent clips them separately at 0.25σ³ and 0.25σ⁴. That fourth bound does not account for the mean-dependent 4μκ3 term. Independent raw-moment clipping therefore does not commute with centering and can generate an invalid fourth cumulant even when the input is a valid distribution entirely above zero. This is an identifiable closure error, separate from missing tensor entries.','',
    table(['Mean (same centered distribution)','True K4','Clipped K4','Offline unclipped M3/M4 K4','Clipped K4+2v²'],[[r['mean'],r['true_input_and_output_fourth'],r['clipped_fourth'],r['unclipped34_fourth'],r['clipped_marginal_slack']] for r in clip['translation_counterexample']]),'',
    'The finite distribution has probabilities (0.75,0.25) at μ+(−0.5,1.5), variance0.75, third cumulant0.75 and fourth cumulant−0.375. ReLU is exactly the identity at both support points. The Gaussian negative tail at these large positive means is negligible. This check uses float64 to separate the systematic guard effect from float32 cancellation.','',
    'On the same complete-fourth oracle inputs, holding corrected mean/M2 fixed and disabling only raw-M3/M4 clipping **offline** changes aggregate next marginal-K4 risk from '+fmt(clip['summary']['clipped'])+' to '+fmt(clip['summary']['unclipped34'])+'. Per-case clip fractions and approximate stream intervals are in clipping_diagnostic.json. This post-hoc explanation is not a recurrent candidate, is not used for selection, and does not establish that removing guards in a submission would be stable or improve the adjusted objective. Primary third/fourth algorithms remain frozen.','']
    lines += extra
    (ROOT/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
