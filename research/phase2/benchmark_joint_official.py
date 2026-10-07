"""Run the unchanged official CLI serially and retain every report, including failures.

The 100-sample targets are an execution smoke test, not accuracy evidence.
No runner, metering configuration, resource cap, or estimator source is modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', default='joint_full_candidate')
    parser.add_argument('--runners', nargs='+', default=['subprocess'],
                        choices=['local', 'subprocess'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[20260926])
    parser.add_argument('--n-mlps', type=int, default=1)
    parser.add_argument('--prefix', type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / args.candidate / 'estimator.py'
    executable = Path(sys.executable).with_name('whest.exe')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    report = {'protocol': 'Unchanged official CLI; default Phase 2 caps; four threads; '
                         '100 samples for execution only; serial runs; all failures retained.',
              'source': str(source), 'source_sha256': digest, 'records': []}
    manifest = args.prefix.with_suffix('.json')
    paths = [manifest]
    for runner in args.runners:
        for seed in args.seeds:
            paths.extend(Path(f'{args.prefix}_{runner}_{seed}{suffix}')
                         for suffix in ('.json', '.stderr.txt'))
    if any(path.exists() for path in paths):
        parser.error('Output already exists; choose a new prefix to preserve all evidence.')
    manifest.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    for runner in args.runners:
        for seed in args.seeds:
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise RuntimeError('Candidate changed during the frozen execution check.')
            command = [str(executable), 'run', '--estimator', str(source),
                       '--runner', runner, '--n-mlps', str(args.n_mlps),
                       '--n-samples', '100', '--seed', str(seed),
                       '--max-threads', '4', '--format', 'json']
            print(f'Running {runner}, seed {seed}, source {digest}', flush=True)
            result = subprocess.run(command, capture_output=True, encoding='utf-8', env=env)
            output = Path(f'{args.prefix}_{runner}_{seed}.json')
            output.write_text(result.stdout, encoding='utf-8')
            Path(f'{args.prefix}_{runner}_{seed}.stderr.txt').write_text(
                result.stderr, encoding='utf-8')
            record = {'runner': runner, 'seed': seed, 'command': command,
                      'returncode': result.returncode, 'report': str(output)}
            try:
                official = json.loads(result.stdout)
                results = official['results']
                summary = {key: results[key] for key in
                           ('n_failed_mlps', 'failure_breakdown')}
                summary['per_mlp'] = [{key: row[key] for key in
                                      ('mlp_index', 'flops_used', 'wall_time_s',
                                       'residual_wall_time_s')}
                                     for row in results['per_mlp']]
                # Keep the complete official output separately, including breakdowns.
                record['summary'] = summary
            except (ValueError, KeyError):
                record['parse_error'] = 'Missing/invalid official summary; inspect raw output.'
            report['records'].append(record)
            manifest.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(record), flush=True)
    if any(record.get('parse_error') or record['returncode'] != 0 or
           record.get('summary', {}).get('n_failed_mlps', 1) != 0
           for record in report['records']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
