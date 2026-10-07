"""Summarize the fixed fourth-transport pilot and verify preserved artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--preserved',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path.')
    report = json.loads(args.report.read_text(encoding='utf-8'))
    preserved = json.loads(args.preserved.read_text(encoding='utf-8'))
    diagnostic_hashes = {key.replace('\\','/'):value for key,value in report['diagnostic_code_sha256'].items()}
    assert len(report['records'])==4 and report['frozen_submission_unchanged']
    expected_new_replica = Path('research/phase2/fourth_transport_candidate/estimator.py')
    verified = []
    new_replica = None
    for relative,digest in preserved.items():
        path = ROOT/relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.resolve()==(ROOT/expected_new_replica).resolve():
            # This file was created during this experiment as an initial copy
            # before the snapshot, then modified into the research replica.
            # It is not a pre-existing source or frozen submission artifact.
            assert digest==report['frozen_submission'][str(ROOT/'research/phase2/joint_full_runtime/estimator.py')]
            assert actual==diagnostic_hashes['fourth_transport_candidate/estimator.py']
            new_replica = {'path':relative,'initial_copy_sha256':digest,'research_sha256':actual}
        else:
            assert actual==digest, f'Pre-existing artifact changed: {relative}'
            verified.append(relative)
    assert new_replica is not None
    by_network = []
    for record in report['records']:
        old,new = (record['methods'][mode] for mode in ('gram','complete'))
        risk_old,risk_new = (method['mean_cross_stream_risk_by_layer'][-1] for method in (old,new))
        var_old,var_new = (method['prevariance_relative_rmse_by_layer'][-1] for method in (old,new))
        by_network.append({'seed':record['seed'],'final_mean_risk_gram':risk_old,
                           'final_mean_risk_complete':risk_new,
                           'final_mean_relative_improvement':1-risk_new/risk_old,
                           'final_variance_rmse_gram':var_old,
                           'final_variance_rmse_complete':var_new,
                           'final_variance_relative_improvement':1-var_new/var_old})
    by_checkpoint = []
    for index in range(5):
        old = sum(record['local_oracle_paired_transport_only'][index]
                  ['off_diagonal_transport_cross_stream_risk']['gram'] for record in report['records'])/4
        new = sum(record['local_oracle_paired_transport_only'][index]
                  ['off_diagonal_transport_cross_stream_risk']['complete'] for record in report['records'])/4
        by_checkpoint.append({'source_layer':report['source_checkpoints_zero_based'][index]+1,
                              'gram_off_diagonal_transport_risk':old,
                              'complete_off_diagonal_transport_risk':new,
                              'relative_improvement':1-new/old})
    budget_floor = .1*2**41
    frozen_flops = 196317442558
    summary = {'frozen_source_and_archive_unchanged':True,
               'pre_existing_artifacts_verified':len(verified),
               'new_research_replica':new_replica,'by_network':by_network,
               'one_step_oracle_transport_by_checkpoint':by_checkpoint,
               'aggregate':report['summary'],
               'headroom_before_score_multiplier_increases':{
                   'frozen_flops':frozen_flops,'floor_flops':budget_floor,
                   'additional_flops':budget_floor-frozen_flops,
                   'relative_additional_flops':budget_floor/frozen_flops-1},
               'decision':'Do not pursue dominant modes from this pilot: the predeclared '
                          'criterion for improving BOTH final mean and variance was not met. '
                          'No submission candidate was changed or promoted.'}
    args.output.open('x',encoding='utf-8').write(json.dumps(summary,indent=2))
    print(json.dumps({key:value for key,value in summary.items() if key!='aggregate'},indent=2))


if __name__ == '__main__':
    main()
