"""Diagnostic only: separate cold/warm calls, output conversion and live JSON inputs.

Observe GC pauses without changing its policy. These measurements do not replace
the execution gate in benchmark_joint_official.py. Each mode uses a fresh process.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parent


def diagnostic(args):
    from whestbench.concurrency import apply_thread_limit
    apply_thread_limit(4)
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from whestbench import sample_mlp
    from whestbench.scoring import materialise_predictions
    from whestbench.subprocess_worker import _payload_to_mlp
    from threadpoolctl import threadpool_info

    path = ROOT / args.candidate / 'estimator.py'
    spec = importlib.util.spec_from_file_location('profile_candidate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mlp = sample_mlp(1024, 16, fnp.random.default_rng(fnp.random.SeedSequence(20260926).spawn(3)[0]))
    request = None
    if args.mode.startswith('payload'):
        # Retain precisely the nested-list representation the official worker
        # keeps in its request during predict. JSON parse and weight loading
        # occur before the budget, as they do in the unchanged worker.
        wire = json.dumps({'width': mlp.width, 'depth': mlp.depth,
                           'weights': [w.tolist() for w in mlp.weights], 'seed': mlp.seed})
        request = json.loads(wire)
        del wire, mlp
        mlp = _payload_to_mlp(request)
        if args.mode == 'payload-released':
            request = None
    est = module.Estimator()
    gc_events = []
    starts = {}
    active = False

    def observe_gc(phase, info):
        if not active:
            return
        generation = info['generation']
        if phase == 'start':
            starts[generation] = time.perf_counter()
        elif generation in starts:
            gc_events.append({'generation': generation,
                              'seconds': time.perf_counter() - starts.pop(generation),
                              'collected': info['collected']})

    report = {'protocol': 'Diagnostic, not an eligibility gate. No GC policy changes.',
              'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'mode': args.mode, 'threads': threadpool_info(),
              'weight_strides': list(mlp.weights[0].strides), 'records': []}
    gc.callbacks.append(observe_gc)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            for iteration in range(2):
                gc_events.clear()
                active = True
                with flops.BudgetContext(flop_budget=2**41, quiet=True) as budget:
                    pred = est.predict(mlp, 2**41)
                    convert_start = time.perf_counter()
                    if args.mode == 'float64':
                        arr = np.asarray(pred, dtype=np.float64)
                    else:
                        arr = materialise_predictions(pred, depth=mlp.depth, width=mlp.width)
                    del pred
                    conversion_s = time.perf_counter() - convert_start
                active = False
                summary = budget.summary_dict(by_namespace=True)
                report['records'].append({'iteration': iteration,
                                          'conversion_s': conversion_s,
                                          'gc_events': list(gc_events),
                                          'gc_seconds': sum(e['seconds'] for e in gc_events),
                                          'budget': summary,
                                          'prediction_sha256': hashlib.sha256(
                                              np.asarray(arr, dtype=np.float32).tobytes()).hexdigest()})
                print(json.dumps({'mode': args.mode, 'iteration': iteration,
                                  'residual': summary['residual_wall_time_s'],
                                  'gc_seconds': sum(e['seconds'] for e in gc_events),
                                  'conversion_s': conversion_s}), flush=True)
    finally:
        active = False
        gc.callbacks.remove(observe_gc)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', default='joint_full_candidate')
    parser.add_argument('--mode', choices=['array', 'float64', 'payload', 'payload-released'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; use a new path.')
    if args.mode:
        diagnostic(args)
        return
    outputs = [args.output.with_name(f'{args.output.stem}_{mode}.json') for mode in
               ('array', 'float64', 'payload', 'payload-released')]
    if any(path.exists() for path in outputs):
        parser.error('Diagnostic output exists; use a new path.')
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    records = []
    for mode, output in zip(('array', 'float64', 'payload', 'payload-released'), outputs):
        subprocess.run([sys.executable, __file__, '--candidate', args.candidate,
                        '--mode', mode, '--output', str(output)], check=True, env=env)
        records.append({'mode': mode, 'report': str(output)})
        args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
