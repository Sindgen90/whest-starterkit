"""Focused affine-completion controls, no selection or full-width claims."""
import hashlib,json,warnings
import numpy as np
import flopscope.numpy as fnp
from whestbench.concurrency import apply_thread_limit
from whestbench.generation import sample_mlp
from common import ROOT,load,measure,crossrisk
from oracle_study import NETWORKS
def main():
    path=ROOT/'exact_transport_controls.json';assert not path.exists()
    report=dict(purpose='Exact within-model affine fourth transport versus fast global Q, with identical nonlinear closure and unchanged third algorithm. Fixed b8, QR/E/F, four small networks only. Not used for new full-width selection.',records=[])
    rule=json.loads((ROOT/'partition_selection.json').read_text())['selected_rule']
    for n,seed in NETWORKS:
        data=np.load(ROOT/f'reference_n{n}_seed{seed}.npz');mlp=sample_mlp(n,16,fnp.random.default_rng(seed),seed=seed)
        predictions={}
        for name in ('qr','qblock8','qrblock8'):
            est=load(name).Estimator();est.complete_pair=True;est.partition_rule=rule
            try:
                pred,cost,states=measure(est,mlp,True)
                rec=dict(n=n,seed=seed,method=name,status='run',cost=cost,source_sha256=hashlib.sha256((ROOT/name/'estimator.py').read_bytes()).hexdigest(),mean_risks=[crossrisk(pred[i][None]-data['means'][:,i]) for i in range(16)],cov_risks=[crossrisk(s['cov'][None]-data['covs'][:,i]) for i,s in enumerate(states)])
                predictions[name]=pred
            except Exception as e:rec=dict(n=n,seed=seed,method=name,status='failed',exception=repr(e))
            report['records'].append(rec);path.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:rec[k] for k in ('n','seed','method','status')}),flush=True)
        np.savez_compressed(ROOT/f'exact_controls_n{n}_seed{seed}.npz',**predictions)
if __name__=='__main__':
    apply_thread_limit(4)
    with warnings.catch_warnings():warnings.simplefilter('ignore');main()
