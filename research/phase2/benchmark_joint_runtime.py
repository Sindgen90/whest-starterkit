"""Frozen accuracy/equivalence check; execution eligibility uses the official CLI.

Compare the preserved full source and its runtime derivative on the same exposed
public networks, with no fitting. Never overwrite reports or drop failed checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
from datasets import Dataset
from threadpoolctl import threadpool_limits
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
from whestbench.scoring import materialise_predictions
from benchmark_runtime import load

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', default='joint_full_runtime')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; choose a new path.')
    names = ('joint_full_candidate', args.candidate)
    modules = {name: load(name) for name in names}
    hashes = {name: hashlib.sha256((ROOT/name/'estimator.py').read_bytes()).hexdigest()
              for name in names}
    report = {'protocol': 'Frozen paired accuracy/equivalence check on exposed public '
                         'networks 0,1,4-7,12-15. No fitting. Four threads. '
                         'In-process timings are diagnostic only; official subprocess '
                         'checks are the execution gate.',
              'source_sha256': hashes,
              'tolerances': {'rtol': 5e-6, 'atol': 2e-6, 'max_abs': 1e-5,
                             'relative_final_mse_drift': .001},
              'records': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(4), warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for shard, indices in (('../tmp/phase2_mini_first8.arrow', (0,1,4,5,6,7)),
                               ('../tmp/phase2_mini_validation8.arrow', (4,5,6,7))):
            data = Dataset.from_file(shard).with_format('numpy')
            for index in indices:
                row = data[index]
                weights = fnp.asarray(row['weights'], dtype=fnp.float32)
                mlp = MLP(weights.shape[1], len(weights), list(weights), seed=20261001)
                truth = np.asarray(row['all_layer_means'], dtype=np.float64)
                predictions = {}
                mse = {}
                record = {'shard': shard, 'mlp_id': int(row['mlp_id']), 'methods': {}}
                for name, module in modules.items():
                    if hashlib.sha256((ROOT/name/'estimator.py').read_bytes()).hexdigest() != hashes[name]:
                        raise RuntimeError('Source changed during frozen accuracy check.')
                    estimator = module.Estimator()
                    started = time.perf_counter()
                    with flops.BudgetContext(flop_budget=2**41, quiet=True) as budget:
                        pred = estimator.predict(mlp, 2**41)
                        arr = materialise_predictions(pred, depth=mlp.depth, width=mlp.width)
                        del pred
                    wall = time.perf_counter() - started
                    predictions[name] = np.asarray(arr, dtype=np.float64)
                    assert arr.shape == truth.shape and np.isfinite(arr).all()
                    mse[name] = float(np.mean((predictions[name][-1]-truth[-1])**2))
                    summary = budget.summary_dict(by_namespace=True)
                    record['methods'][name] = {'raw_final_mse': mse[name],
                                               'per_layer_mse': np.mean((predictions[name]-truth)**2,axis=1).tolist(),
                                               'flops': budget.flops_used,
                                               'wall_s': wall,
                                               'budget': summary}
                a, b = (predictions[name] for name in names)
                record['max_abs_prediction_difference'] = float(np.max(np.abs(b-a)))
                record['relative_final_mse_drift'] = mse[names[1]]/mse[names[0]]-1
                try:
                    np.testing.assert_allclose(b, a, rtol=5e-6, atol=2e-6)
                    assert record['max_abs_prediction_difference'] <= 1e-5
                    assert abs(record['relative_final_mse_drift']) <= .001
                    record['equivalent'] = True
                except AssertionError as exc:
                    record['equivalent'] = False
                    record['error'] = str(exc)
                report['records'].append(record)
                args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
                print(json.dumps({key: value for key, value in record.items() if key != 'methods'}), flush=True)
    report['summary'] = {'equivalence_failures': sum(not r['equivalent'] for r in report['records']),
                         'max_abs_prediction_difference': max(r['max_abs_prediction_difference'] for r in report['records']),
                         'mean_raw_final_mse': {name: float(np.mean([r['methods'][name]['raw_final_mse']
                                                               for r in report['records']])) for name in names}}
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2), flush=True)
    if report['summary']['equivalence_failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
