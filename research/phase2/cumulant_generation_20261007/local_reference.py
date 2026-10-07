"""Teacher-forced marginal errors vs independent full-width public moments."""
import json,hashlib
from pathlib import Path
import numpy as np
from numpy_marginal import update,cumulants
R=Path(__file__).resolve().parent
def main():
    out=R/'local_reference.json';assert not out.exists();report={'reference':'keenanpepper Phase2 mini, 1e8 fresh draws; mean target copied from official 1e9 bake. Raw marginal moments are f64; full joint tensor absent. Single-stream cumulant MSE includes reference noise.','records':[],'checks':[]}
    for idx in (0,1):
        path=R/'public/moments'/f'mlp_{idx:05d}.npz';d=np.load(path);pre=cumulants(d,True);post=cumulants(d)
        assert pre.shape==(4,16,1024) and int(d['n_samples'])==100000000
        report['checks'].append(dict(id=idx,n_samples=int(d['n_samples']),global_index=int(d['global_index']),mean_to_baked_mse=float(np.mean((d['mean'][-1]-d['gt_mean'][-1])**2)),pre_gaussian_layer0_third_rms=float(np.sqrt(np.mean(pre[2,0]**2))),pre_gaussian_layer0_fourth_rms=float(np.sqrt(np.mean(pre[3,0]**2))),keys=d.files))
        for layer in range(16):
            for name in ('baseline_snapshot','unclip34','centered','coherent'):
                pred,mask,lam=update(name,*pre[:,layer]);target=post[:,layer];risk=np.mean((pred-target)**2,axis=1)
                report['records'].append(dict(id=idx,layer=layer,method=name,mse_mean_variance_third_fourth=risk.tolist(),mean_mse_to_baked=float(np.mean((pred[0]-d['gt_mean'][layer])**2)),raw_clip_fractions=mask.mean(axis=1).tolist(),attenuation=lam,min_variance=float(pred[1].min()),min_fourth_plus_two_variance_squared=float((pred[3]+2*pred[1]**2).min()),finite=bool(np.isfinite(pred).all())))
    report['summary']={name:{'mse_mean_variance_third_fourth':np.mean([r['mse_mean_variance_third_fourth'] for r in report['records'] if r['method']==name],axis=0).tolist(),'average_raw_clip_fractions':np.mean([r['raw_clip_fractions'] for r in report['records'] if r['method']==name],axis=0).tolist(),'min_attenuation':min(r['attenuation'] for r in report['records'] if r['method']==name)} for name in ('baseline_snapshot','unclip34','centered','coherent')};out.write_text(json.dumps(report,indent=2));print(json.dumps(report['summary']),flush=True)
if __name__=='__main__':main()
