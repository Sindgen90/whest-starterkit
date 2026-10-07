"""Reuse existing offline reference streams to isolate state and readout errors."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from datasets import Dataset
from threadpoolctl import threadpool_limits
from whestbench import MLP
from benchmark_cumulant_audit import load, ROOT
from diagnose_higher_moments import cumulants, corrections
from diagnose_distribution import gaussian_relu, risk_summary


def readout(m,v,k3,k4):
    sigma = np.sqrt(np.maximum(v,1e-12))
    delta = corrections(m,v,k3,k4)[...,:2].sum(-1)
    return gaussian_relu(m,v)+np.clip(delta,-.25*sigma,.25*sigma)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--names',nargs='+',default=['baseline'])
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    assert not a.output.exists()
    data = Dataset.from_file('../tmp/phase2_mini_first8.arrow').with_format('numpy')
    report = {'protocol':'No new draws or fitting. Four independent old streams of 65536 draws per network; true premeans from baked 1e9-draw targets; readout resets are diagnostics, not a causal additive error decomposition.',
              'records':[],'reference_sha256':{}}
    with threadpool_limits(4),warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        for index in (0,1,2,3):
            row = data[index]
            mid = int(row['mlp_id'])
            path = ROOT/f'results/higher_moments_20261001_mlp{mid}.npz'
            report['reference_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            oracle = np.load(path)
            truth,premean = oracle['truth'],oracle['premean']
            weights = fnp.asarray(row['weights'],dtype=fnp.float32)
            mlp = MLP(weights.shape[1],len(weights),list(weights),seed=0)
            oracle_states = [cumulants(raw,65536) for raw in oracle['powers']]
            pooled_v,pooled_k3,pooled_k4 = cumulants(oracle['powers'].mean(0),4*65536)
            for name in a.names:
                mod,src = load(name)
                est = mod.Estimator()
                est.collect_diagnostics = True
                # Full terminal state for diagnosis; counts/timing are excluded from promotion.
                with flops.BudgetContext(flop_budget=2**41,quiet=True):
                    predicted = np.asarray(est.predict(mlp,2**41),dtype=np.float64)
                m,v,k3,k4,mu,mass = np.asarray(est.trace,dtype=np.float64).transpose(1,0,2)
                interventions = {key:[] for key in ('oracle_gaussian','oracle_k34','repair_mean','repair_variance','repair_mean_variance','repair_shape')}
                for ov,ok3,ok4 in oracle_states:
                    # The first incoming vector is exactly Gaussian for these fixed weights.
                    ok3 = ok3.copy(); ok4 = ok4.copy()
                    ok3[0] = 0; ok4[0] = 0
                    interventions['oracle_gaussian'].append(gaussian_relu(premean,ov))
                    interventions['oracle_k34'].append(readout(premean,ov,ok3,ok4))
                    interventions['repair_mean'].append(readout(premean,v,k3,k4))
                    interventions['repair_variance'].append(readout(m,ov,k3,k4))
                    interventions['repair_mean_variance'].append(readout(premean,ov,k3,k4))
                    interventions['repair_shape'].append(readout(m,v,ok3,ok4))
                record = {'mlp_id':mid,'method':name,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
                          'propagated_mse':np.mean((predicted-truth)**2,axis=1).tolist(),
                          'premean_rmse':np.sqrt(np.mean((m-premean)**2,axis=1)).tolist(),
                          'relative_prevariance_rmse':np.sqrt(np.mean(((v-pooled_v)/pooled_v)**2,axis=1)).tolist(),
                          'standardized_k3_rmse':np.sqrt(np.mean((k3-pooled_k3)**2/pooled_v**3,axis=1)).tolist(),
                          'standardized_k4_rmse':np.sqrt(np.mean((k4-pooled_k4)**2/pooled_v**4,axis=1)).tolist(),
                          'minimum_prevariance':float(v.min()),'finite':bool(np.isfinite(predicted).all()),
                          'mean_clip_fraction':np.mean(np.abs(corrections(m,v,k3,k4)[...,:2].sum(-1))>.25*np.sqrt(v),axis=1).tolist(),
                          'interventions':{key:risk_summary(vals,truth) for key,vals in interventions.items()}}
                report['records'].append(record)
                a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
                np.savez_compressed(a.output.with_name(a.output.stem+f'_{name}_mlp{mid}.npz'),m=m,v=v,k3=k3,k4=k4,prediction=predicted,truth=truth)
                print(json.dumps({'mlp_id':mid,'method':name,'final_mse':record['propagated_mse'][-1],
                                  'final_relative_variance_rmse':record['relative_prevariance_rmse'][-1],
                                  'oracle_local_cross_risk':record['interventions']['oracle_k34']['cross_stream_risk'][-1]}),flush=True)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
