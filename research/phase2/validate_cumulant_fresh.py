"""Fresh fixed-weight networks, offline paired references, unchanged official gate."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import warnings
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.concurrency import apply_thread_limit
from whestbench.generation import sample_mlp
from whestbench.runner import SubprocessRunner,EstimatorEntrypoint
from whestbench.scoring import ContestSpec,ContestData,materialise_predictions
from whestbench.cli import _run_estimator_with_runner
from benchmark_cumulant_audit import load,AUDIT
from diagnose_fourth_transport import cross_risk,paired_delta_interval


def reference(weights,seed,stream,samples,batch,center):
    means = np.zeros(center.shape)
    raw = np.zeros((4,*center.shape))
    rng = np.random.default_rng(np.random.SeedSequence([20261005,seed,stream,913]))
    started = time.perf_counter()
    for offset in range(0,samples,batch):
        x = rng.standard_normal((min(batch,samples-offset),weights.shape[1]),dtype=np.float32)
        for layer,w in enumerate(weights):
            z = x@w
            y = z.astype(np.float64)-center[layer]
            square = y*y
            raw[0,layer] += y.sum(0)
            raw[1,layer] += square.sum(0)
            raw[2,layer] += (square*y).sum(0)
            raw[3,layer] += (square*square).sum(0)
            x = np.maximum(z,0)
            means[layer] += x.sum(0,dtype=np.float64)
    print(json.dumps({'seed':seed,'stream':stream,'reference_wall_s':time.perf_counter()-started}),flush=True)
    return means/samples,raw/samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',default='third_memory16')
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    assert not a.output.exists()
    apply_thread_limit(4)
    names = ('baseline',a.candidate)
    loaded = {name:load(name) for name in names}
    hashes = {name:hashlib.sha256(src.read_bytes()).hexdigest() for name,(mod,src) in loaded.items()}
    report = {'protocol':'Fresh width-1024/depth-16 networks, official He weight sampler with integer RNG seeds 20261051,20261052. Four independent streams of 65536 float32 Gaussian inputs each, batch 2048. One frozen candidate, no fitting or selection from these references. Paired-delta t intervals condition on these two networks and describe sampling uncertainty, not population generalization.',
              'candidate_sha256':hashes,'seed_rule':'sample_mlp rng=fnp.random.default_rng(network_seed); input RNG SeedSequence([20261005,network_seed,stream,913])',
              'streams':4,'samples_per_stream':65536,'records':[],
              'selection':'Require candidate improvement on both fresh networks, aggregate sampling-only 95% paired interval excluding zero, and all official subprocess predictions passing default caps.'}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    mlps,targets,delta_streams = [],[],[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        for seed in (20261051,20261052):
            mlp = sample_mlp(1024,16,fnp.random.default_rng(seed),seed=seed)
            mlps.append(mlp)
            weights = np.asarray(mlp.weights,dtype=np.float32)
            predictions,counts = {},{}
            for name,(mod,src) in loaded.items():
                assert hashlib.sha256(src.read_bytes()).hexdigest()==hashes[name]
                with flops.BudgetContext(flop_budget=2**41,quiet=True) as b:
                    pred = mod.Estimator().predict(mlp,2**41)
                    arr = materialise_predictions(pred,depth=16,width=1024)
                    del pred
                predictions[name] = np.asarray(arr,dtype=np.float64)
                assert np.isfinite(predictions[name]).all()
                counts[name] = b.flops_used
            center = np.zeros((16,1024))
            for layer in range(1,16):
                center[layer] = predictions['baseline'][layer-1]@weights[layer].astype(np.float64)
            streams,powers = [],[]
            for stream in range(4):
                mean,raw = reference(weights,seed,stream,65536,2048,center)
                streams.append(mean); powers.append(raw)
            streams = np.asarray(streams)
            pooled = streams.mean(0)
            targets.append(fnp.asarray(pooled,dtype=fnp.float32))
            errors = {name:predictions[name][None,:,:]-streams for name in names}
            delta = np.mean(errors[a.candidate]**2-errors['baseline']**2,axis=-1)
            delta_streams.append(delta)
            rec = {'seed':seed,'weights_sha256':hashlib.sha256(weights.tobytes()).hexdigest(),
                   'flops':counts,'finite':True,
                   'cross_stream_risk':{name:cross_risk(errors[name]).tolist() for name in names},
                   'pooled_target_mse':{name:np.mean((predictions[name]-pooled)**2,axis=-1).tolist() for name in names},
                   'paired_raw_delta':paired_delta_interval(delta),
                   'paired_adjusted_delta':paired_delta_interval(.1*delta)}
            report['records'].append(rec)
            a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
            np.savez_compressed(a.output.with_name(a.output.stem+f'_seed{seed}.npz'),means=streams,powers=np.asarray(powers),precenter=center,**predictions)
            print(json.dumps({'seed':seed,'final_cross_risk':{name:rec['cross_stream_risk'][name][-1] for name in names},
                              'delta':{k:v[-1] for k,v in rec['paired_raw_delta'].items()}}),flush=True)
        spec = ContestSpec(width=1024,depth=16,n_mlps=2,flop_budget=2**41,ground_truth_samples=262144,seed=20261005)
        data = ContestData(spec,mlps,targets,[t[-1] for t in targets],[0.,0.])
        report['official'] = {}
        for name,(mod,src) in loaded.items():
            # These are the official CLI's unchanged runner and scoring function.
            result = _run_estimator_with_runner(SubprocessRunner(),entrypoint=EstimatorEntrypoint(src),contest_spec=spec,
                                                n_mlps=2,profile=False,detail='raw',output_format='json',contest_data=data)
            out = a.output.with_name(a.output.stem+'_'+name+'_official.json')
            assert not out.exists()
            out.write_text(json.dumps(result,indent=2),encoding='utf-8')
            report['official'][name] = {'report':str(out),'n_failed_mlps':result['results']['n_failed_mlps'],
                                       'adjusted_pooled_target_score':result['results']['adjusted_final_layer_score'],
                                       'per_mlp':[{k:r[k] for k in ('flops_used','wall_time_s','residual_wall_time_s')} for r in result['results']['per_mlp']]}
            print(json.dumps({'official':name,**report['official'][name]}),flush=True)
    combined = paired_delta_interval(np.mean(delta_streams,axis=0))
    risks = {name:float(np.mean([r['cross_stream_risk'][name][-1] for r in report['records']])) for name in names}
    report['summary'] = {'final_raw_cross_stream_risk':risks,
                         'final_adjusted_cross_stream_risk':{name:.1*r for name,r in risks.items()},
                         'raw_improvement_fraction':1-risks[a.candidate]/risks['baseline'],
                         'final_paired_delta_interval':{k:v[-1] for k,v in combined.items()},
                         'positive_on_both_networks':all(r['paired_raw_delta']['mean'][-1]<0 for r in report['records']),
                         'all_official_passed':all(r['n_failed_mlps']==0 for r in report['official'].values())}
    s = report['summary']
    s['promotion_criterion_passed'] = s['positive_on_both_networks'] and s['all_official_passed'] and s['final_paired_delta_interval']['upper_95']<0
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(s,indent=2),flush=True)


if __name__=='__main__':
    main()
