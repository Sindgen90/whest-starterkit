import json
from pathlib import Path
import numpy as np
from numpy_marginal import gaussian,cumulants
R=Path(__file__).resolve().parent
def mean_update(m,v,t,k):
    raw,_,s,ph,_=gaussian(m,v);a=m/s
    return raw[1]-t*a*ph/(6*v)+k*(a*a-1)*ph/(24*v*s)+t*t*(a**4-6*a*a+3)*ph/(72*s**5)+t*k*(-a**5+10*a**3-15*a)*ph/(144*s**6)
def main():
    out=R/'public_propagation_diagnostics_v2.json';assert not out.exists();report={'records':[],'reference':'Phase2 1e8 independent moments plus official 1e9 mean. G-derived g4row compared as an effective marginal cumulant; G itself is not Q. Teacher-forced interventions are diagnostic only; no targets enter estimator. C snapshot is after the source zeroes its diagonal in place; compare off-diagonal entries only. v1 covariance_mse erroneously included that zero diagonal; all other v1 metrics remain valid.'}
    for idx in (0,1):
        a=np.load(R/f'public_diagnostic_states_id{idx}.npz');d=np.load(R/'public/moments'/f'mlp_{idx:05d}.npz');pre=cumulants(d,True);post=cumulants(d)
        for layer in range(16):
            get=lambda key:a[f'debug_public_v29_{layer}_{key}'];p=np.array([get(k) for k in ('m','v','d3','d4')]);target=d['gt_mean'][layer];vectors={'propagated_formula':mean_update(*p),'oracle_all':mean_update(*pre[:,layer]),'oracle_mean_variance':mean_update(*pre[:2,layer],*p[2:]),'oracle_third':mean_update(*p[:2],pre[2,layer],p[3]),'oracle_fourth':mean_update(*p[:3],pre[3,layer])}
            rec=dict(id=idx,layer=layer,pre_mse_mean_variance_third_effective_fourth=np.mean((p-pre[:,layer])**2,axis=1).tolist(),mean_mse={k:float(np.mean((v-target)**2)) for k,v in vectors.items()},predicted_mean_mse=float(np.mean((get('mu')-target)**2)),formula_parity_maxabs=float(np.max(np.abs(vectors['propagated_formula']-get('mu')))),finite=bool(np.isfinite(p).all()))
            if layer<15:
                mu=pre[0,layer];m2=d['pre_m2'][layer];c=d['pre_M11'][layer].astype(float)-np.outer(mu,mu);t=d['pre_M21'][layer].astype(float)-2*mu[:,None]*d['pre_M11'][layer]-m2[:,None]*mu[None,:]+2*mu[:,None]**2*mu[None,:];np.fill_diagonal(t,0)
                np.fill_diagonal(c,0)
                predpost=np.array([get(k) for k in ('mu','post_var','post3','post4')]);rec.update(covariance_offdiagonal_mse=float(np.sum((get('C')-c)**2)/(1024*1023)),repeated_third_offdiagonal_mse=float(np.sum((get('P')-t)**2)/(1024*1023)),post_mse_mean_variance_third_fourth=np.mean((predpost-post[:,layer])**2,axis=1).tolist(),min_marginal_slack=float((predpost[3]+2*predpost[1]**2).min()))
            report['records'].append(rec)
    report['summary']={'mean_pre_mse':np.mean([r['pre_mse_mean_variance_third_effective_fourth'] for r in report['records']],axis=0).tolist(),'mean_post_mse':np.mean([r['post_mse_mean_variance_third_fourth'] for r in report['records'] if 'post_mse_mean_variance_third_fourth' in r],axis=0).tolist(),'mean_mse_controls':{key:float(np.mean([r['mean_mse'][key] for r in report['records']])) for key in report['records'][0]['mean_mse']},'final_mse_controls':{key:float(np.mean([r['mean_mse'][key] for r in report['records'] if r['layer']==15])) for key in report['records'][0]['mean_mse']},'formula_parity_maxabs':max(r['formula_parity_maxabs'] for r in report['records'])};out.write_text(json.dumps(report,indent=2));print(json.dumps(report['summary']))
if __name__=='__main__':main()
