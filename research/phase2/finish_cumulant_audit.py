"""Summarize immutable evidence, retain baseline, and verify preservation."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import numpy as np
from benchmark_cumulant_audit import ROOT,AUDIT


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = AUDIT/'audit.json'
    assert not output.exists()
    protocol = json.loads((AUDIT/'protocol.json').read_text())
    screen = json.loads((AUDIT/'screen.json').read_text())
    square = json.loads((AUDIT/'third_square_screen.json').read_text())
    confirmation = json.loads((AUDIT/'confirmation.json').read_text())
    fresh = json.loads((AUDIT/'fresh_validation.json').read_text())
    local = json.loads((AUDIT/'local_diagnostics.json').read_text())
    assert not fresh['summary']['promotion_criterion_passed'], 'Review passing candidate for packaging.'
    expected = protocol['candidate_sha256'].copy()
    expected['third_square'] = json.loads((AUDIT/'third_square_protocol.json').read_text())['source_sha256']
    for name,digest in expected.items():
        assert sha(AUDIT/name/'estimator.py')==digest
    assert sha(ROOT/'joint_full_runtime/estimator.py')==protocol['baseline_sha256']
    frozen = json.loads((AUDIT/'preserved.json').read_text())
    changed = [path for path,digest in frozen.items() if not Path(path).is_file() or sha(Path(path))!=digest]
    assert not changed, changed
    archive = ROOT.parents[1]/'submissions/phase2_joint_full_runtime_20261002.tar.gz'
    archive_digest = sha(archive)
    assert archive_digest=='89035799f6acfe5dd14948fbedefd37ea2bc35472ce2ce72518c4485676782b3'
    with tarfile.open(archive,'r:gz') as tar:
        entry = [m for m in tar.getmembers() if m.name.endswith('estimator.py')]
        assert len(entry)==1
        assert hashlib.sha256(tar.extractfile(entry[0]).read()).hexdigest()==protocol['baseline_sha256']
    snapshot = AUDIT/'baseline_snapshot'
    assert not snapshot.exists()
    snapshot.mkdir()
    shutil.copyfile(ROOT/'joint_full_runtime/estimator.py',snapshot/'estimator.py')
    shutil.copyfile(ROOT/'joint_full_runtime/LICENSE',snapshot/'LICENSE')
    assert sha(snapshot/'estimator.py')==protocol['baseline_sha256']
    for name in expected:
        folder = AUDIT/name
        shutil.copyfile(ROOT/'joint_full_runtime/LICENSE',folder/'LICENSE')
        (folder/'PROVENANCE.md').write_text(
            f'# Experimental ablation: {name}\n\n'
            f'Created 5 October 2026 from joint_full_runtime. Baseline SHA-256: {protocol["baseline_sha256"]}.\n\n'
            f'Candidate SHA-256: {expected[name]}. Separate standalone source; no reference data or fitted targets. '
            'NOT promoted, packaged or submitted. See ../CUMULANT_AUDIT.md, ../RESULTS.md and the frozen protocol JSONs. '
            'Screening timings alone do not establish execution eligibility.\n',encoding='utf-8')
    methods = {**screen['summary'],**square['summary']}
    baseline = methods['baseline']['raw_mse']
    for name,result in methods.items():
        result['raw_improvement_fraction']=1-result['raw_mse']/baseline
    official = {}
    for tag in ('baseline_gate','memory_gate'):
        report = json.loads((AUDIT/(tag+'.json')).read_text())
        official[tag] = [x for r in report['records'] for x in r['summary']['per_mlp']]
    result = {'decision':'Retain frozen joint_full_runtime and existing archive. No new submission packaged.',
              'reason':'Top-16 older third sources improve exposed and excluded public point estimates, but fresh-network gains have mixed signs and the paired 95% sampling interval includes zero. Other ablations do not warrant advancement.',
              'baseline_sha256':protocol['baseline_sha256'],'candidate_sha256':expected,
              'preservation':{'checked_files':len(frozen),'changed_files':changed,'runner_and_metering_unchanged':True,
                              'archive':str(archive.resolve()),'archive_sha256':archive_digest,'archive_estimator_matches_baseline':True},
              'screen':methods,'confirmation':confirmation['summary'],'fresh_validation':fresh['summary'],
              'official_initial':official,'official_fresh':fresh['official'],
              'tests':{'passed':33,'warnings':399,'command':'.venv-phase2/Scripts/python.exe -m pytest research/phase2/test_cumulant_audit.py research/phase2/test_joint_full_runtime.py research/phase2/test_joint_cumulant.py research/phase2/test_fourth_transport.py -q --disable-warnings'},
              'evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for p in AUDIT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    rows = ['| Ablation | Raw final MSE | Nominal adjusted | FLOPs (billions) | Change vs baseline | In-process wall / residual (s) |',
            '|---|---:|---:|---:|---:|---:|']
    for name,r in methods.items():
        rows.append(f'| {name} | {r["raw_mse"]:.9e} | {r["nominal_adjusted"]:.9e} | {r["flops"]/1e9:.6f} | {100*r["raw_improvement_fraction"]:+.4f}% | {r["wall_s"]:.3f} / {r["residual_s"]:.3f} |')
    s = fresh['summary']
    interval = s['final_paired_delta_interval']
    lines = [
        '# Measured results and recommendation',
        '',
        '**Retain the existing best estimator and archive.** No new candidate is promoted or packaged; no upload is made.',
        '',
        'Four standalone analytical ablations were evaluated without fitting or a parameter grid. '
        'All have finite predictions on four identical width-1024, depth-16 exposed public networks (IDs 0-3). '
        'Lower MSE/score is better; a positive percent change below denotes improvement. '
        'These are baked 1e9-draw targets. The in-process timings in this table are descriptive; '
        'they do not establish eligibility.',
        '',*rows,'',
        'The one-mode cross-pairing candidate is essentially unchanged in accuracy. The pure marginal '
        'fourth response has a small 0.17% aggregate improvement and mixed per-network effects. '
        'Adding mixed third squares worsens every screened network: formal completeness of one term '
        'does not make approximate cumulant inputs or the remaining closure accurate. No variants '
        'were combined or retuned. The mixed third-square and marginal-fourth changes update all three covariance '
        'planes rather than changing only returned means.',
        '',
        '## Compressed-history confirmation',
        '',
        'On IDs 8-11, excluded from this candidate selection but exposed in prior work, raw MSE is '
        f'{confirmation["summary"]["baseline"]["raw_mse"]:.9e} for baseline and '
        f'{confirmation["summary"]["third_memory16"]["raw_mse"]:.9e} for compressed history '
        f'({100*(1-confirmation["summary"]["third_memory16"]["raw_mse"]/confirmation["summary"]["baseline"]["raw_mse"]):.4f}% improvement). '
        'Three of four networks improve; one worsens. Both retain the 0.1 multiplier. The original '
        '2% screen threshold was not met; the documented confirmation amendment advances only this '
        'cheap candidate to check the small effect. This deviation is explicit and the source stayed frozen.',
        '',
        'Two fresh networks use the official He sampler at seeds 20261051 and 20261052. Each receives '
        'four independent reference streams of 65,536 draws (524,288 total input draws across both networks). '
        'Samples are offline and never enter inference. Cross-stream risk subtracts independent '
        'reference sampling variance in expectation. It estimates error conditional on these networks; '
        'the t intervals describe reference sampling uncertainty, not uncertainty over all possible weights.',
        '',
        '| Fresh seed | Baseline raw cross-stream risk | History raw cross-stream risk | Candidate minus baseline; sampling-only 95% interval |',
        '|---|---:|---:|---|']
    for rec in fresh['records']:
        delta = rec['paired_raw_delta']
        lines.append(f'| {rec["seed"]} | {rec["cross_stream_risk"]["baseline"][-1]:.9e} | {rec["cross_stream_risk"]["third_memory16"][-1]:.9e} | {delta["mean"][-1]:+.6e} [{delta["lower_95"][-1]:+.6e}, {delta["upper_95"][-1]:+.6e}] |')
    lines += ['',
        f'Aggregate raw risk is {s["final_raw_cross_stream_risk"]["baseline"]:.9e} versus '
        f'{s["final_raw_cross_stream_risk"]["third_memory16"]:.9e}, a '
        f'{100*s["raw_improvement_fraction"]:.4f}% point improvement. Estimated adjusted risks are '
        f'{s["final_adjusted_cross_stream_risk"]["baseline"]:.9e} versus '
        f'{s["final_adjusted_cross_stream_risk"]["third_memory16"]:.9e}. '
        f'The paired raw difference is {interval["mean"]:+.6e}, with 95% interval '
        f'[{interval["lower_95"]:+.6e}, {interval["upper_95"]:+.6e}]. '
        'It includes zero and one fresh network worsens. The frozen promotion criterion fails. '
        'We do not assert that the population objective improved or that this candidate is definitively worse.',
        '',
        '## Official execution gates and numerical checks',
        '',
        'The unchanged official CLI first checked baseline and compressed history on identical '
        'synthetic smoke networks at seed 20261005 (100-draw targets are for execution only). '
        'The official CLI’s unchanged SubprocessRunner and scoring function then checked both '
        'estimators on the same two fresh networks with their pooled offline targets. Default caps '
        'and four threads are unchanged. All eight official predictions passed. '
        'The source ledger confirms no evaluator or metering edits.',
        '',
        '| Estimator | Official FLOPs | Official residual range (s) | Maximum official prediction wall (s) | Failures |',
        '|---|---:|---:|---:|---:|']
    for name,tag in (('baseline','baseline_gate'),('third_memory16','memory_gate')):
        vals = official[tag]+fresh['official'][name]['per_mlp']
        lines.append(f'| {name} | {vals[0]["flops_used"]:,} | {min(v["residual_wall_time_s"] for v in vals):.6f}–{max(v["residual_wall_time_s"] for v in vals):.6f} | {max(v["wall_time_s"] for v in vals):.6f} | 0/4 |')
    lines += ['',
        'The official adjusted scores against pooled fresh MC targets are '
        f'{fresh["official"]["baseline"]["adjusted_pooled_target_score"]:.9e} and '
        f'{fresh["official"]["third_memory16"]["adjusted_pooled_target_score"]:.9e}. '
        'They include reference MC noise; use the paired/cross-stream estimates above for the accuracy '
        'comparison. Neither is a leaderboard result or a remeasurement of the user-reported 8.62e-8.',
        '',
        '33 mathematical/regression tests pass. These cover exact affine third-factor transport, '
        'residual reinjection, recentering and Gaussian pairing subtraction, signed fourth-mode '
        'transport, response coefficients, zero ablations, input immutability, repeatability, '
        'terminal/full-state equivalence, and the preserved estimator’s previous checks. '
        'All screened large-network predictions are finite. In the local baseline/history diagnostic, '
        'minimum incoming variance exceeds 0.0967 and no marginal mean correction hits its clip bound. '
        'Symmetrization/diagonal replacement do not prove PSD or existence of a joint law.',
        '',
        '## What was learned',
        '',
        'Local oracle K3/K4 readouts remain accurate at the reference sampling resolution, while '
        'propagated means, variance and shape information leave a much larger error. Older CP columns '
        'can be retained for only 2.932 billion additional FLOPs, avoiding the cost of a whole second '
        'dense generation, but this pruning rule does not establish a reliable improvement. '
        'The next mathematical target is newly generated distinct third information and mixed fourth '
        '3+1 information, whose first-layer size is comparable to retained paired information. '
        'Do not infer from the failed one-mode or mixed-square ablation that all such contributions '
        'are irrelevant. Do not add higher cumulant orders or tune coefficients on these references.',
        '',
        '## Preserved submission and reproduction',
        '',
        f'{len(frozen)} pre-existing evidence/evaluator/metering files were rehashed; none changed. '
        'The baseline source also has an identical snapshot in baseline_snapshot/. '
        f'The preserved submission is `{archive.resolve()}`, SHA-256 `{archive_digest}`. '
        'The estimator bytes inside the tar match the frozen baseline. Existing official archive '
        'validation remains intact; no new tar is created for rejected candidates.',
        '',
        'From whest-starterkit, use .venv-phase2/Scripts/python.exe and the scripts '
        '`benchmark_cumulant_audit.py`, `diagnose_cumulant_audit.py`, '
        '`validate_cumulant_fresh.py`, and `benchmark_joint_official.py`. '
        'All experiment outputs are append-only: choose new output names on reruns. '
        'Source hashes, fixed network indices/seeds, parameters, exact commands for official '
        'checks, raw reports, MC moments/predictions and the preservation ledger accompany this report. '
        'See CUMULANT_AUDIT.md for equations and adaptations of Hanin’s paper.']
    (AUDIT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (AUDIT/'tests.txt').write_text(result['tests']['command']+'\n33 passed, 399 warnings in 10.94s\n'
                                    'Result recorded from the completed tool execution; warnings suppressed in display, counts retained.\n',encoding='utf-8')
    print(json.dumps({'decision':result['decision'],'preserved_files':len(frozen),'archive':str(archive.resolve()),
                      'results':str((AUDIT/'RESULTS.md').resolve())},indent=2))


if __name__=='__main__':
    main()
