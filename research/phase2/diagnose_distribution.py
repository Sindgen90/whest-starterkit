"""OFFLINE oracle interventions. Reference targets must NEVER enter submissions.

Separates propagated moment error from final marginal Gaussian closure error.
Four independent MC streams support cross-stream squared-error estimates;
checkpoint restarts measure regeneration of error after repairing the state.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace
import warnings
import numpy as np
from datasets import Dataset
from threadpoolctl import threadpool_limits
import flopscope.numpy as fnp
import flopscope as flops
from whestbench import MLP

ROOT = Path(__file__).parent
SOURCE = ROOT/"venture_baseline"/"estimator.py"
spec = importlib.util.spec_from_file_location("oracle_baseline",SOURCE)
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)


def ndtr(values):
    return np.asarray(flops.stats.norm.cdf(fnp.asarray(values)))


def gaussian_relu(mean, variance):
    variance = np.maximum(variance,1e-12)
    sigma = np.sqrt(variance)
    alpha = mean/sigma
    cdf = ndtr(alpha)
    pdf = np.exp(-.5*alpha*alpha)/np.sqrt(2*np.pi)
    return mean*cdf+sigma*pdf


def closure(weights, initial_mean=None, initial_cov=None, start=0, mean_correction=None,
            second_moment_correction=None):
    """8-node research replica; optional mean feedback retains Gaussian covariance."""
    n = weights.shape[1]
    mu = np.zeros(n,dtype=np.float32) if initial_mean is None else initial_mean.astype(np.float32)
    cov = np.eye(n,dtype=np.float32) if initial_cov is None else initial_cov.astype(np.float32)
    rows,premeans,prevars,masses = [],[],[],[]
    for layer,w in enumerate(weights[start:],start=start):
        m = mu@w
        s = w.T@cov@w
        v = np.maximum(np.diag(s),1e-12)
        sigma = np.sqrt(v)
        a = m/sigma
        phi = (np.exp(-.5*a.astype(np.float64)**2)/np.sqrt(2*np.pi)).astype(np.float32)
        cdf = ndtr(a.astype(np.float64)).astype(np.float32)
        mu = m*cdf+sigma*phi
        postvar = np.maximum((m*m+v)*cdf+m*sigma*phi-mu*mu,0)
        outer = np.outer(sigma,sigma)
        rho = np.clip(s/np.maximum(outer,1e-30),-.999999,.999999)
        a2 = a*a
        pair = a2[:,None]+a2[None,:]
        ab = np.outer(a,a)
        correction = np.zeros_like(s)
        for node,weight in zip(source._QUAD_NODES01,source._QUAD_WEIGHTS01):
            t = rho*node
            denominator = np.maximum(1-t*t,1e-12)
            density = np.exp(-(pair-2*t*ab)/(2*denominator))/(2*np.pi*np.sqrt(denominator))
            correction += (rho*weight)*(rho-t)*density
        cov = np.outer(cdf,cdf)*s + outer*correction
        np.fill_diagonal(cov,postvar)
        cov = .5*(cov+cov.T)
        if mean_correction is not None:
            previous_mu = mu
            mu = (mu+mean_correction(m,v,layer)).astype(np.float32)
            if second_moment_correction is not None:
                corrected_var = postvar+second_moment_correction(m,v,layer)-(mu*mu-previous_mu*previous_mu)
                np.fill_diagonal(cov,np.maximum(corrected_var,1e-12))
        rows.append(mu.copy())
        premeans.append(m.copy())
        prevars.append(v.copy())
        masses.append((np.abs(s).sum(1)-np.abs(v))/n)
    return {"mean":np.array(rows),"premean":np.array(premeans),
            "prevar":np.array(prevars),"mass":np.array(masses)}


def sample_statistics(weights, truth, samples, seed, batch, checkpoints):
    """Independent iid streams; center moments on baked means, not sample means."""
    depth,n = truth.shape
    true_pre = np.vstack((np.zeros(n),truth[:-1]))
    true_pre = np.stack([true_pre[l]@weights[l].astype(np.float64) for l in range(depth)])
    variance_sum = np.zeros((depth,n))
    mean_sum = np.zeros((depth,n))
    cov_sum = {layer:np.zeros((n,n)) for layer in checkpoints}
    rng = np.random.default_rng(seed)
    for offset in range(0,samples,batch):
        x = rng.standard_normal((min(batch,samples-offset),n),dtype=np.float32)
        for layer,w in enumerate(weights,start=1):
            z = x@w
            centered = z.astype(np.float64)-true_pre[layer-1]
            variance_sum[layer-1] += np.sum(centered*centered,axis=0)
            x = np.maximum(z,0)
            mean_sum[layer-1] += np.sum(x,axis=0,dtype=np.float64)
            if layer in checkpoints:
                centered_post = x-truth[layer-1].astype(np.float32)
                cov_sum[layer] += centered_post.T@centered_post
    return true_pre,variance_sum/samples,mean_sum/samples,{k:v/samples for k,v in cov_sum.items()}


def risk_summary(predictions,truth):
    """Cross-stream products remove independent variance, not finite-N bias."""
    errors = np.asarray(predictions,dtype=np.float64)-truth
    streams = len(errors)
    direct = np.mean(errors*errors,axis=-1)
    cross = ((errors.sum(0)**2-np.sum(errors*errors,axis=0))/(streams*(streams-1))).mean(-1)
    spread = np.var(errors,axis=0,ddof=1).mean(-1)
    return {"mean_single_stream_mse":np.mean(direct,axis=0).tolist(),
            "cross_stream_risk":cross.tolist(),"estimated_single_stream_noise_mse":spread.tolist(),
            "pooled_prediction_mse":np.mean(errors.mean(0)**2,axis=-1).tolist(),
            "per_stream_mse":direct.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arrow",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--limit",type=int,default=4)
    p.add_argument("--samples",type=int,default=65536)
    p.add_argument("--streams",type=int,default=4)
    p.add_argument("--batch",type=int,default=2048)
    args = p.parse_args()
    if args.streams<2 or args.samples<1:
        p.error("At least two independent streams and positive sample count required")
    report = {"protocol":"Frozen oracle diagnostic, 2026-09-30; no deployable estimator tuning",
        "dataset":str(args.arrow),"network_selection":"first four exposed development networks",
        "samples_per_stream":args.samples,"streams":args.streams,"checkpoints":[4,8,12],
        "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),"records":[],
        "caveat":"Reference means have finite 1e9-sample error; sampled covariance is approximate. Cross-stream risk removes independent sampling variance but not plug-in bias or shared target noise. Negative estimates are possible. No inference samples reuse dataset seeds."}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    data = Dataset.from_file(str(args.arrow)).with_format("numpy")
    with threadpool_limits(4), warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        for i in range(min(args.limit,len(data))):
            start = time.perf_counter()
            row = data[i]
            weights = np.asarray(row["weights"],dtype=np.float32)
            truth = np.asarray(row["all_layer_means"],dtype=np.float64)
            state = closure(weights)
            official = source.Estimator()
            official.setup(SimpleNamespace(submission_dir=SOURCE.parent,seed=20260930))
            mlp = MLP(weights.shape[1],len(weights),[fnp.asarray(w) for w in weights],seed=20260930)
            calibrated = np.asarray(official.predict(mlp,2**41),dtype=np.float64)
            official._calibration = None
            original = np.asarray(official.predict(mlp,2**41),dtype=np.float64)
            parity = float(np.max(np.abs(original-state["mean"])))
            if parity>5e-6:
                raise AssertionError(f"Research replica mismatch {parity}")
            predictions = {name:[] for name in ("mean_only","variance_only","both_moments")}
            restarts = {str(layer):[] for layer in (4,8,12)}
            prevars = []
            rec = {"mlp_id":int(row["mlp_id"]),"replica_max_abs_difference":parity,
                "baseline_raw_mse_by_layer":np.mean((original-truth)**2,axis=1).tolist(),
                "calibrated_raw_mse_by_layer":np.mean((calibrated-truth)**2,axis=1).tolist()}
            for stream in range(args.streams):
                stream_start = time.perf_counter()
                seed = 20260930+1009*i+100003*stream
                true_m,var,sampled,covs = sample_statistics(weights,truth,args.samples,seed,args.batch,(4,8,12))
                prevars.append(var)
                predictions["mean_only"].append(gaussian_relu(true_m,state["prevar"]))
                predictions["variance_only"].append(gaussian_relu(state["premean"],var))
                predictions["both_moments"].append(gaussian_relu(true_m,var))
                for layer in (4,8,12):
                    suffix = closure(weights,truth[layer-1],covs[layer],start=layer)
                    restarts[str(layer)].append(suffix["mean"])
                print(json.dumps({"mlp_id":rec["mlp_id"],"stream":stream,
                      "seconds":time.perf_counter()-stream_start,
                      "both_moments_final_mse":float(np.mean((predictions["both_moments"][-1][-1]-truth[-1])**2))}),flush=True)
            rec["interventions"] = {k:risk_summary(v,truth) for k,v in predictions.items()}
            rec["restarts"] = {k:risk_summary(v,truth[int(k):]) for k,v in restarts.items()}
            # Half-size vs paired doubled-size estimates expose plug-in bias/noise.
            if args.streams==4:
                paired_vars = [np.mean(prevars[:2],axis=0),np.mean(prevars[2:],axis=0)]
                rec["both_moments_double_samples"] = risk_summary(
                    [gaussian_relu(true_m,v) for v in paired_vars],truth)
            rec["premean_error_mse_by_layer"] = np.mean((state["premean"]-true_m)**2,axis=1).tolist()
            rec["prevariance_relative_rmse_by_layer"] = np.sqrt(np.mean(
                ((state["prevar"]-np.mean(prevars,axis=0))/np.maximum(np.mean(prevars,axis=0),1e-12))**2,axis=1)).tolist()
            rec["wall_s"] = time.perf_counter()-start
            report["records"].append(rec)
            args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
            np.savez_compressed(args.output.with_name(args.output.stem+f"_mlp{rec['mlp_id']}.npz"),
                truth=truth,baseline=original,calibrated=calibrated,reference_premean=true_m,
                propagated_premean=state["premean"],propagated_prevariance=state["prevar"],
                sampled_prevariances=np.asarray(prevars),**{k:np.asarray(v) for k,v in predictions.items()},
                **{f"restart_{k}":np.asarray(v) for k,v in restarts.items()})
            print(json.dumps({"mlp_id":rec["mlp_id"],"done_seconds":rec["wall_s"],
                  "final_cross_stream_risk":{k:v["cross_stream_risk"][-1] for k,v in rec["interventions"].items()}}),flush=True)


if __name__=="__main__":
    main()
