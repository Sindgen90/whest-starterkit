"""Audit frozen source, official gates, accuracy and preserved artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT/'research/phase2'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def mean(values):
    return sum(values)/len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accuracy', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    parser.add_argument('--preserved', type=Path, required=True)
    parser.add_argument('--completed-diagnostic', type=Path,
                        help='New diagnostic manifest captured while it was still appending; '
                             'audit its completed children explicitly instead of changing the ledger.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path.')
    accuracy, gate, preserved = (read(path) for path in
                                 (args.accuracy,args.gate,args.preserved))
    source = PHASE/'joint_full_runtime/estimator.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert accuracy['source_sha256']['joint_full_runtime'] == gate['source_sha256'] == digest
    official_rows = []
    for record in gate['records']:
        assert record['runner'] == 'subprocess' and record['returncode'] == 0
        official = read(Path(record['report']))
        assert official['results']['n_failed_mlps'] == 0
        assert official['run_config']['residual_wall_time_limit_s'] == .4
        assert official['run_config']['wall_time_limit_s'] == 120
        assert official['run_config']['flop_budget'] == 2**41
        for row in official['results']['per_mlp']:
            assert row['residual_wall_time_s'] <= .4 and row['wall_time_s'] <= 120
            assert row['flops_used'] <= 2**41
            official_rows.append({'seed': record['seed'], **{key:row[key] for key in
                                  ('mlp_index','flops_used','wall_time_s','residual_wall_time_s')},
                                  'calls': sum(ns['calls'] for ns in
                                   row['breakdowns']['estimator']['by_namespace'].values())})
    assert accuracy['summary']['equivalence_failures'] == 0
    mismatches = []
    completed = []
    for relative, expected in preserved.items():
        path = ROOT/relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != expected:
            if args.completed_diagnostic and path.resolve() == args.completed_diagnostic.resolve():
                diagnostic = read(path)
                assert {row['mode'] for row in diagnostic} == {
                    'array','float64','payload','payload-released'} and len(diagnostic) == 4
                for row in diagnostic:
                    child = read(Path(row['report']))
                    assert child['source_sha256'] == accuracy['source_sha256']['joint_full_candidate']
                    assert len(child['records']) == 2
                completed.append({'path':relative, 'snapshot_sha256':expected,
                                  'completed_sha256':actual,
                                  'reason':'New diagnostic manifest was still appending when '
                                           'the preservation snapshot was taken; all four '
                                           'completed children and their reference hashes verified.'})
            else:
                mismatches.append(relative)
    assert not mismatches, f'Preserved files changed: {mismatches}'
    groups = {}
    for group, ids in (('development',(0,1)), ('frozen_evaluation',(4,5,6,7,12,13,14,15))):
        rows = [row for row in accuracy['records'] if row['mlp_id'] in ids]
        groups[group] = {'networks': len(rows), 'mean_raw_final_mse': {
            name: mean([row['methods'][name]['raw_final_mse'] for row in rows])
            for name in accuracy['source_sha256']}}
        old, new = (groups[group]['mean_raw_final_mse'][name] for name in
                    ('joint_full_candidate','joint_full_runtime'))
        groups[group]['relative_mse_drift'] = new/old-1
    report = {'source_sha256':digest, 'official_subprocess_rows':official_rows,
              'official_subprocess_summary': {
                  'predictions':len(official_rows), 'failures':0,
                  'max_residual_s':max(row['residual_wall_time_s'] for row in official_rows),
                  'mean_residual_s':mean([row['residual_wall_time_s'] for row in official_rows]),
                  'min_residual_s':min(row['residual_wall_time_s'] for row in official_rows),
                  'max_wall_s':max(row['wall_time_s'] for row in official_rows)},
              'accuracy_summary':accuracy['summary'], 'accuracy_groups':groups,
              'max_individual_relative_mse_drift': max(abs(row['relative_final_mse_drift'])
                                                      for row in accuracy['records']),
              'preservation':{'files_verified':len(preserved)-len(completed),'mismatches':mismatches,
                              'completed_new_diagnostics':completed,
                              'includes':'Original candidates, prior results, all existing '
                                         'archives, and installed runner/meter Python sources.'}}
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
