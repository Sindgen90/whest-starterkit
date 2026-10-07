"""Offline fixed paired-fourth transport diagnostic; no submission changes.

Four fixed synthetic width-32, depth-16 networks, no tuning or fitting. Compare
the preserved Gram closure with the mathematically complete paired projection.
Independent Monte Carlo streams measure means, incoming variances and local
transport errors. Oracle moments never enter either predicted recurrence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
from threadpoolctl import threadpool_limits
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
from fourth_transport_candidate.estimator import Estimator
from joint_full_runtime.estimator import Estimator as Frozen
from fourth_transport import transport

ROOT = Path(__file__).resolve().parent
SEEDS = (3103,4103,5103,6103)
CHECKPOINTS = (0,3,7,11,14)


def paired_kstat(raw,count):
    """Unbiased fourth joint k-statistic for repeated indices i,i,j,j.

    raw=(E[Y],E[Y_i Y_j],E[Y_i^2 Y_j],E[Y_i^2 Y_j^2]); a fixed shift
    reduces cancellation and is removed by exact centralization below.
    """
    first,second,third,fourth = raw
    square_mean = first*first
    covariance = second-np.outer(first,first)
    variance = np.diag(covariance)
    marginal_second = np.diag(second)
    central4 = (fourth-2*first[None,:]*third-2*first[:,None]*third.T
                +np.outer(marginal_second,square_mean)
                +np.outer(square_mean,marginal_second)
                +4*np.outer(first,first)*second-3*np.outer(square_mean,square_mean))
    pairings = np.outer(variance,variance)+2*covariance*covariance
    n = float(count)
    return n*n*((n+1)*central4-(n-1)*pairings)/((n-1)*(n-2)*(n-3))


def sample_reference(weights,center,samples,batch,seed):
    depth,n = center.shape
    precenter = np.zeros_like(center)
    for layer in range(1,depth):
        precenter[layer] = center[layer-1]@weights[layer].astype(np.float64)
    pre1 = np.zeros((depth,n))
    pre2 = np.zeros((depth,n))
    means = np.zeros((depth,n))
    raw = {}
    for source in CHECKPOINTS:
        for key in (('post',source),('pre',source+1)):
            raw[key] = [np.zeros(n),*[np.zeros((n,n)) for _ in range(3)]]
    rng = np.random.default_rng(seed)

    def accumulate(key,y):
        square = y*y
        values = raw[key]
        values[0] += y.sum(0)
        values[1] += y.T@y
        values[2] += square.T@y
        values[3] += square.T@square

    for offset in range(0,samples,batch):
        x = rng.standard_normal((min(batch,samples-offset),n),dtype=np.float32)
        for layer,w in enumerate(weights):
            z = x@w
            y = z.astype(np.float64)-precenter[layer]
            pre1[layer] += y.sum(0)
            pre2[layer] += (y*y).sum(0)
            if ('pre',layer) in raw:
                accumulate(('pre',layer),y)
            x = np.maximum(z,0)
            means[layer] += x.sum(0,dtype=np.float64)
            if ('post',layer) in raw:
                accumulate(('post',layer),x.astype(np.float64)-center[layer])
    prevariance = (pre2/samples-(pre1/samples)**2)*samples/(samples-1)
    pairs = {key:paired_kstat([value/samples for value in values],samples)
             for key,values in raw.items()}
    return means/samples,prevariance,pairs


def cross_risk(errors):
    """Independent-stream product removes reference sampling variance."""
    errors = np.asarray(errors,dtype=np.float64)
    streams = len(errors)
    return ((errors.sum(0)**2-np.sum(errors*errors,axis=0))/(streams*(streams-1))).mean(-1)


def paired_delta_interval(values):
    values = np.asarray(values,dtype=np.float64)
    se = values.std(axis=0,ddof=1)/np.sqrt(len(values))
    delta = values.mean(0)
    radius = student_critical_95(len(values)-1)*se
    return {'mean':delta.tolist(),'standard_error':se.tolist(),
            'lower_95':(delta-radius).tolist(),'upper_95':(delta+radius).tolist()}


def student_critical_95(df):
    """Student t quantile by direct density integration; no SciPy dependency.

    With t=sqrt(df)*tan(theta), the density integral is proportional to
    integral cos(theta)^(df-1) dtheta. Its range is bounded even for df=1.
    """
    nodes,weights = np.polynomial.legendre.leggauss(64)
    constant = math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2)) / math.sqrt(math.pi)
    low,high = 0.,math.pi/2
    for _ in range(60):
        theta = (low+high)/2
        integral = theta/2*np.sum(weights*np.cos((nodes+1)*theta/2)**(df-1))
        if .5+constant*integral < .975:
            low = theta
        else:
            high = theta
    return math.sqrt(df)*math.tan((low+high)/2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=262144)
    parser.add_argument('--streams',type=int,default=4)
    parser.add_argument('--batch',type=int,default=4096)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.samples < 4 or args.streams < 2 or args.batch < 1:
        parser.error('At least four samples, two independent streams and positive batch required.')
    if args.output.exists() or any(args.output.parent.glob(args.output.stem+'_seed*.npz')):
        parser.error('Outputs exist; use a new path to preserve evidence.')
    runtime = ROOT/'joint_full_runtime/estimator.py'
    archive = ROOT.parents[1]/'submissions/phase2_joint_full_runtime_20261002.tar.gz'
    frozen = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (runtime,archive)}
    code = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__),ROOT/'fourth_transport.py',ROOT/'fourth_transport_candidate/estimator.py')}
    report = {'date':'2026-10-03','protocol':__doc__, 'width':32,'depth':16,
              'network_seeds':SEEDS,'samples_per_stream':args.samples,'streams':args.streams,
              'source_checkpoints_zero_based':CHECKPOINTS,
              'frozen_submission':frozen,'diagnostic_code_sha256':code,
              'decision_rule':'Only pursue dominant-mode approximation if the complete '
                              'paired projection improves aggregate final mean cross-stream '
                              'risk AND final relative variance RMSE by at least 5%, with '
                              'the paired Monte Carlo 95% interval for final mean MSE change below zero.',
              'limitations':'Synthetic small networks; no hidden-test or full-width claim. '
                            'Cross-stream risk may be negative. Only paired tensor components '
                            'are completed; other fourth-index patterns remain omitted. '
                            'Variance RMSE uses a pooled plug-in reference. Sampling interval '
                            'does not represent uncertainty across new networks.',
              'records':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')

    all_mean_deltas = []
    all_mean_risks = {mode:[] for mode in ('gram','complete')}
    all_var_squared = {mode:[] for mode in ('gram','complete')}
    with threadpool_limits(4),warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        for seed in SEEDS:
            started = time.perf_counter()
            rng = np.random.default_rng(seed)
            weights = np.asarray([rng.normal(size=(32,32)).astype(np.float32)*np.sqrt(2/32)
                                  for _ in range(16)])
            mlp = MLP(32,16,[fnp.asarray(w) for w in weights],seed=seed)
            predictions,prevars,states = {},{},{}
            record = {'seed':seed,'methods':{}}
            for mode in ('gram','complete'):
                estimator = Estimator()
                estimator.fourth_mode = mode
                estimator.collect_diagnostics = True
                with flops.BudgetContext(flop_budget=2**41,quiet=True) as budget:
                    prediction = estimator.predict(mlp,2**41)
                predictions[mode] = np.asarray(prediction,dtype=np.float64)
                prevars[mode] = np.array([np.asarray(row[1]) for row in estimator.trace],dtype=np.float64)
                states[mode] = estimator.state_trace
                record['methods'][mode] = {'flops':budget.flops_used,
                                          'timing_is_eligibility_evidence':False}
            original = Frozen()
            original.collect_diagnostics = True
            parity = float(np.max(np.abs(np.asarray(original.predict(mlp,2**41))-predictions['gram'])))
            assert parity == 0, f'Replica changed baseline: {parity}'
            record['replica_max_abs_difference'] = parity
            reference_means,reference_vars,reference_pairs = [],[],[]
            for stream in range(args.streams):
                stream_seed = 9000001+seed*1009+stream*1000003
                m,v,q = sample_reference(weights,predictions['gram'],args.samples,args.batch,stream_seed)
                reference_means.append(m)
                reference_vars.append(v)
                reference_pairs.append(q)
                print(json.dumps({'network_seed':seed,'stream':stream,
                                  'elapsed_s':time.perf_counter()-started}),flush=True)
            means = np.asarray(reference_means)
            variances = np.asarray(reference_vars)
            pooled_m,pooled_v = means.mean(0),variances.mean(0)
            mean_deltas = []
            variance_deltas = []
            for mode in ('gram','complete'):
                errors = predictions[mode][None,:,:]-means
                risk = cross_risk(errors)
                relative_squared = ((prevars[mode]-pooled_v)/np.maximum(pooled_v,1e-12))**2
                record['methods'][mode].update(
                    mean_cross_stream_risk_by_layer=risk.tolist(),
                    mean_pooled_mse_by_layer=np.mean((predictions[mode]-pooled_m)**2,axis=1).tolist(),
                    prevariance_relative_rmse_by_layer=np.sqrt(relative_squared.mean(1)).tolist())
                all_mean_risks[mode].append(risk)
                all_var_squared[mode].append(relative_squared.mean(1))
            for stream in range(args.streams):
                mean_deltas.append(np.mean((predictions['complete']-means[stream])**2
                                          -(predictions['gram']-means[stream])**2,axis=1))
                # Fixed deterministic normalizer: the reference noise cancels
                # exactly from the paired variance-error difference.
                scale = np.maximum(prevars['gram'],1e-12)
                variance_deltas.append(np.mean(((prevars['complete']-variances[stream])/scale)**2
                                              -((prevars['gram']-variances[stream])/scale)**2,axis=1))
            record['paired_mean_mse_change_by_layer'] = paired_delta_interval(mean_deltas)
            record['paired_normalized_variance_mse_change_by_layer'] = paired_delta_interval(variance_deltas)
            record['reference_variance_stream_spread_by_layer'] = np.sqrt(
                np.mean(variances.var(0,ddof=1)/np.maximum(pooled_v,1e-12)**2,axis=1)).tolist()
            local = []
            for source in CHECKPOINTS:
                mask = ~np.eye(32,dtype=bool)
                transport_errors = {mode:[] for mode in ('gram','complete')}
                for sample in reference_pairs:
                    q = fnp.asarray(sample['post',source],dtype=fnp.float64)
                    w = fnp.asarray(weights[source+1],dtype=fnp.float64)
                    for mode in transport_errors:
                        _,plane = transport(q,w,mode)
                        transport_errors[mode].append((np.asarray(plane)-sample['pre',source+1])[mask])
                local.append({'source_layer':source+1,'target_layer':source+2,
                              'off_diagonal_transport_cross_stream_risk': {
                                  mode:float(cross_risk(errors)) for mode,errors in transport_errors.items()},
                              'predicted_plane_relative_change':float(np.linalg.norm(
                                  np.asarray(states['complete'][source+1]['pre_fourth'])
                                  -np.asarray(states['gram'][source+1]['pre_fourth']))/
                                  max(np.linalg.norm(np.asarray(states['gram'][source+1]['pre_fourth'])),1e-12))})
            record['local_oracle_paired_transport_only'] = local
            record['wall_s'] = time.perf_counter()-started
            report['records'].append(record)
            all_mean_deltas.append(np.asarray(mean_deltas))
            np.savez_compressed(args.output.with_name(args.output.stem+f'_seed{seed}.npz'),
                                weights=weights,reference_means=means,reference_prevariances=variances,
                                gram_means=predictions['gram'],complete_means=predictions['complete'],
                                gram_prevariances=prevars['gram'],complete_prevariances=prevars['complete'],
                                **{f'{mode}_postQ_{source+1}':np.asarray(states[mode][source]['post_fourth'])
                                   for mode in states for source in CHECKPOINTS},
                                **{f'oracle_{kind}Q_{layer+1}':np.array([q[kind,layer] for q in reference_pairs])
                                   for kind,layer in reference_pairs[0]})
            save()
            print(json.dumps({'seed':seed,'final_mean_risk':{
                mode:record['methods'][mode]['mean_cross_stream_risk_by_layer'][-1] for mode in predictions},
                'final_variance_relative_rmse':{
                mode:record['methods'][mode]['prevariance_relative_rmse_by_layer'][-1] for mode in predictions}}),flush=True)
    risk = {mode:np.mean(rows,axis=0) for mode,rows in all_mean_risks.items()}
    var_rmse = {mode:np.sqrt(np.mean(rows,axis=0)) for mode,rows in all_var_squared.items()}
    # Each aggregated stream consists of independent streams across all networks.
    interval = paired_delta_interval(np.mean(all_mean_deltas,axis=0))
    improvement_m = 1-risk['complete'][-1]/risk['gram'][-1]
    improvement_v = 1-var_rmse['complete'][-1]/var_rmse['gram'][-1]
    report['summary'] = {'mean_cross_stream_risk_by_layer':{mode:rows.tolist() for mode,rows in risk.items()},
                         'prevariance_relative_rmse_by_layer':{mode:rows.tolist() for mode,rows in var_rmse.items()},
                         'final_mean_risk_relative_improvement':float(improvement_m),
                         'final_variance_rmse_relative_improvement':float(improvement_v),
                         'paired_mean_mse_change_by_layer':interval,
                         'pursue_dominant_modes':bool(risk['gram'][-1]>0 and improvement_m>=.05
                                                     and improvement_v>=.05 and interval['upper_95'][-1]<0)}
    for path,digest in frozen.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, f'Frozen submission changed: {path}'
    report['frozen_submission_unchanged'] = True
    save()
    print(json.dumps(report['summary'],indent=2),flush=True)


if __name__ == '__main__':
    main()
