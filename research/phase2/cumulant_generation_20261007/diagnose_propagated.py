import json
from pathlib import Path
import numpy as np
from numpy_marginal import cumulants,update
R=Path(__file__).resolve().parent
out=R/'propagated_diagnostics.json';assert not out.exists();report={'records':[],'note':'Diagnostic copies execute full final affine statistics; their cost is not submission cost. Reference 1e8 marginal moment MSE includes noise.'}
for idx in (0,1):
    a=np.load(R/f'diagnostic_states_id{idx}.npz');d=np.load(R/'public/moments'/f'mlp_{idx:05d}.npz');pre=cumulants(d,True);post=cumulants(d)
    for name in ('baseline_snapshot','unclip34','centered','coherent'):
        nm='debug_'+name
        for layer in range(16):
            get=lambda key:a[f'{nm}_{layer}_{key}']
            predpre=np.array([get(k) for k in ('m','v','d3','d4')]);_,mask,lam=update(name,*predpre)
            r=dict(id=idx,layer=layer,method=name,pre_mse_mean_variance_third_fourth=np.mean((predpre-pre[:,layer])**2,axis=1).tolist(),raw_clip_fractions=mask.mean(axis=1).tolist(),attenuation_recomputed=lam,pre_min_variance=float(predpre[1].min()),finite_pre=bool(np.isfinite(predpre).all()))
            if layer<15:
                predpost=np.array([get(k) for k in ('mu','post_var','post3','post4')]);r.update(post_mse_mean_variance_third_fourth=np.mean((predpost-post[:,layer])**2,axis=1).tolist(),min_fourth_plus_two_variance_squared=float((predpost[3]+2*predpost[1]**2).min()),finite_post=bool(np.isfinite(predpost).all()))
            report['records'].append(r)
report['summary']={name:{'mean_pre_mse':np.mean([x['pre_mse_mean_variance_third_fourth'] for x in report['records'] if x['method']==name],axis=0).tolist(),'mean_post_mse':np.mean([x['post_mse_mean_variance_third_fourth'] for x in report['records'] if x['method']==name and 'post_mse_mean_variance_third_fourth' in x],axis=0).tolist(),'mean_raw_clip_fractions':np.mean([x['raw_clip_fractions'] for x in report['records'] if x['method']==name],axis=0).tolist(),'finite':all(x['finite_pre'] and x.get('finite_post',True) for x in report['records'] if x['method']==name),'min_marginal_moment_slack':min(x['min_fourth_plus_two_variance_squared'] for x in report['records'] if x['method']==name and 'min_fourth_plus_two_variance_squared' in x)} for name in ('baseline_snapshot','unclip34','centered','coherent')};out.write_text(json.dumps(report,indent=2));print(json.dumps(report['summary']))
