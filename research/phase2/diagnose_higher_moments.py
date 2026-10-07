"""Offline K3/K4 oracle study. Reference information must not enter submissions.

Primary comparisons are fixed analytical corrections. A secondary two-feature
fit uses only development networks and is frozen before validation is loaded.
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
from diagnose_distribution import gaussian_relu, risk_summary


def cumulants(raw, count):
    """Raw powers about a fixed reference; translation-invariant k-statistics."""
    r1, r2, r3, r4 = raw
    central2 = r2-r1*r1
    central3 = r3-3*r1*r2+2*r1**3
    central4 = r4-4*r1*r3+6*r1*r1*r2-3*r1**4
    n = float(count)
    k3 = n*n/((n-1)*(n-2))*central3
    k4 = n*n*((n+1)*central4-3*(n-1)*central2**2)/((n-1)*(n-2)*(n-3))
    # r2 estimates variance about the accurate reference mean, as in the prior study.
    return np.maximum(r2,1e-12), k3, k4


def corrections(mean, variance, k3, k4):
    sigma = np.sqrt(np.maximum(variance,1e-12))
    a = mean/sigma
    phi = np.exp(-a*a/2)/np.sqrt(2*np.pi)
    skew = k3/sigma**3
    excess = k4/sigma**4
    return np.stack((-sigma*phi*a*skew/6,
                     sigma*phi*(a*a-1)*excess/24,
                     sigma*phi*(a**4-6*a*a+3)*skew**2/72),axis=-1)


def second_raw_correction(mean,variance,skew,excess):
    """K3/K4 correction to E[ReLU(Z)^2], before recentering the variance."""
    a = mean/np.sqrt(np.maximum(variance,1e-12))
    phi = np.exp(-a*a/2)/np.sqrt(2*np.pi)
    return variance*phi*(skew/3-a*excess/12)


def predict_variants(mean, raw, count, coefficients=None):
    variance,k3,k4 = cumulants(raw,count)
    base = gaussian_relu(mean,variance)
    terms = corrections(mean,variance,k3,k4)
    out = {"gaussian":base,"k3":base+terms[...,0],"k4":base+terms[...,1],
           "k34":base+terms[...,0]+terms[...,1],"k34_k33":base+terms.sum(-1)}
    if coefficients is not None:
        out["fitted_k34"] = base+np.sum(terms[...,:2]*coefficients[:,None,:],axis=-1)
    return out


def sample_powers(weights, mean, count, seed, batch):
    sums = np.zeros((4,*mean.shape),dtype=np.float64)
    rng = np.random.default_rng(seed)
    for offset in range(0,count,batch):
        x = rng.standard_normal((min(batch,count-offset),weights.shape[1]),dtype=np.float32)
        for layer,w in enumerate(weights):
            z = x@w
            y = z.astype(np.float64)-mean[layer]
            square = y*y
            sums[0,layer] += y.sum(0)
            sums[1,layer] += square.sum(0)
            sums[2,layer] += (square*y).sum(0)
            sums[3,layer] += (square*square).sum(0)
            x = np.maximum(z,0)
    return sums/count


def fit_coefficients(networks, count):
    """Per-layer normalized ridge, fixed 1e-3 regularizer toward analytic (1,1)."""
    features, residuals = [],[]
    for net in networks:
        raw = net["powers"].mean(0)
        v,k3,k4 = cumulants(raw,count)
        features.append(corrections(net["premean"],v,k3,k4)[...,:2])
        residuals.append(net["truth"]-gaussian_relu(net["premean"],v))
    coefficients = []
    for layer in range(networks[0]["truth"].shape[0]):
        x = np.concatenate([f[layer] for f in features])
        y = np.concatenate([r[layer] for r in residuals])
        scale = np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-12)
        a = x/scale
        beta = np.linalg.solve(a.T@a/len(a)+1e-3*np.eye(2),
                               a.T@y/len(a)+1e-3*scale)
        coefficients.append(beta/scale)
    # First preactivation is exactly Gaussian; retain analytical weights for this sanity row.
    coefficients[0] = np.ones(2)
    return np.asarray(coefficients)


def evaluate(net,count,coefficients):
    draws = net["powers"]
    predictions = [predict_variants(net["premean"],r,count,coefficients) for r in draws]
    methods = {name:risk_summary([p[name] for p in predictions],net["truth"])
               for name in predictions[0]}
    paired = [predict_variants(net["premean"],draws[j:j+2].mean(0),2*count,coefficients)
              for j in (0,2)]
    double = {name:risk_summary([p[name] for p in paired],net["truth"])
              for name in paired[0]}
    v,k3,k4 = cumulants(draws.mean(0),count*len(draws))
    return {"mlp_id":net["mlp_id"],"methods":methods,"double_samples":double,
            "shape_by_layer":{"skew_rms":np.sqrt(np.mean(k3*k3/v**3,axis=-1)).tolist(),
                              "excess_kurtosis_mean":np.mean(k4/v**2,axis=-1).tolist(),
                              "excess_kurtosis_rms":np.sqrt(np.mean(k4*k4/v**4,axis=-1)).tolist()}}


def aggregate(records):
    return {group:{method:{metric:np.mean([r[group][method][metric] for r in records],axis=0).tolist()
               for metric in ("cross_stream_risk","pooled_prediction_mse","mean_single_stream_mse")}
               for method in records[0][group]} for group in ("methods","double_samples")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development",type=Path,required=True)
    parser.add_argument("--validation",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--limit",type=int,default=4)
    parser.add_argument("--samples",type=int,default=65536)
    parser.add_argument("--batch",type=int,default=2048)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report = {"date":"2026-10-01","protocol":"Fixed K3, K4, K3+K4, K3+K4+K3-squared; secondary ridge fit on development only",
              "samples_per_stream":args.samples,"streams":4,"limit_per_split":args.limit,
              "development":str(args.development),"validation":str(args.validation),
              "seed_rule":"20261001+1009*mlp_id+100003*stream; independent of dataset seeds",
              "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "records":{"development":[],"validation":[]},
              "caveat":"Offline oracle marginal readouts, not propagated cumulants or competition scores. Cross-stream risk removes independent variance, not finite-N plug-in bias or reference-target noise. Validation is excluded from fitting, but has been examined in earlier experiments."}
    save = lambda:args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    save()
    coefficients = None
    ids = set()
    with threadpool_limits(4),warnings.catch_warnings():
        warnings.simplefilter("ignore",UserWarning)
        for split,path in (("development",args.development),("validation",args.validation)):
            data = Dataset.from_file(str(path)).with_format("numpy")
            networks = []
            for i in range(min(args.limit,len(data))):
                row = data[i]
                mid = int(row["mlp_id"])
                assert mid not in ids, "Development and validation must be disjoint"
                ids.add(mid)
                w = np.asarray(row["weights"],dtype=np.float32)
                truth = np.asarray(row["all_layer_means"],dtype=np.float64)
                m = np.vstack((np.zeros(w.shape[1]),truth[:-1]))
                m = np.stack([m[j]@w[j].astype(np.float64) for j in range(len(w))])
                powers = []
                for stream in range(4):
                    start = time.perf_counter()
                    raw = sample_powers(w,m,args.samples,20261001+1009*mid+100003*stream,args.batch)
                    powers.append(raw)
                    p = predict_variants(m,raw,args.samples)
                    print(json.dumps({"split":split,"mlp_id":mid,"stream":stream,
                         "seconds":time.perf_counter()-start,
                         "final_single_stream_mse":{k:float(np.mean((v[-1]-truth[-1])**2)) for k,v in p.items()}}),flush=True)
                net = {"mlp_id":mid,"premean":m,"truth":truth,"powers":np.array(powers)}
                networks.append(net)
                np.savez_compressed(args.output.with_name(args.output.stem+f"_mlp{mid}.npz"),**net)
                rec = evaluate(net,args.samples,coefficients)
                report["records"][split].append(rec)
                save()
                print(json.dumps({"complete_mlp_id":mid,"final_cross_risk":{
                    k:v["cross_stream_risk"][-1] for k,v in rec["methods"].items()}}),flush=True)
            if split=="development":
                coefficients = fit_coefficients(networks,args.samples*4)
                report["coefficients_frozen_before_validation"] = coefficients.tolist()
                report["records"][split] = [evaluate(net,args.samples,coefficients) for net in networks]
                save()
            report.setdefault("summary",{})[split] = aggregate(report["records"][split])
            save()
    print(json.dumps({split:{k:v["cross_stream_risk"][-1] for k,v in s["methods"].items()}
                      for split,s in report["summary"].items()},indent=2),flush=True)


if __name__=="__main__":
    main()
