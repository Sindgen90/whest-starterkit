"""Bounded width-1024 kernels before any full candidate propagation."""
import argparse,hashlib,json,time,warnings
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench.concurrency import apply_thread_limit
from common import ROOT,peak_memory
import kernels as k

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='kernel_profiles.json');p.add_argument('--transport-only',action='store_true');args=p.parse_args()
    path=ROOT/args.output;assert not path.exists()
    report=dict(width=1024,threads=4,seed=20261007,kernels_sha256=hashlib.sha256((ROOT/'kernels.py').read_bytes()).hexdigest(),records=[],timing='In-process bounded kernels. No subprocess eligibility or accuracy claims. Process peak working set is cumulative within this fresh profiler process.')
    rng=np.random.default_rng(20261007);n=1024
    w=fnp.asarray(rng.normal(0,np.sqrt(2/n),(n,n)).astype(np.float32))
    cov=fnp.asarray((np.eye(n)+.002*np.ones((n,n))).astype(np.float32))
    q=fnp.asarray((.001*np.ones((n,n))+.05*np.eye(n)).astype(np.float32));r=fnp.asarray((.002*np.ones((n,n))+.05*np.eye(n)).astype(np.float32))
    aa=fnp.asarray(rng.normal(0,.01,(n,n)).astype(np.float32));bb=w
    def run(label,fn):
        started=time.perf_counter()
        try:
            with flops.BudgetContext(flop_budget=2**43,quiet=True) as budget:result=fn()
            summary=budget.summary_dict()
            rec=dict(kernel=label,status='run',flops=int(budget.flops_used),wall_s=time.perf_counter()-started,residual_s=summary['residual_wall_time_s'],backend_s=summary['flopscope_backend_time_s'],operations=summary['operations'],peak_working_set_bytes=peak_memory())
        except Exception as e:rec=dict(kernel=label,status='failed',exception=repr(e),wall_s=time.perf_counter()-started);result=None
        report['records'].append(rec);path.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(rec),flush=True)
        return result
    if not args.transport_only:run('global_QR_planes',lambda:k.global_planes(q,r,w))
    for b in (8,16):
        parts=run(f'partition_contiguous_b{b}',lambda:k.partition(cov,b,'contiguous',0))
        run(f'partition_random_b{b}',lambda:k.partition(cov,b,'random',0))
        run(f'partition_correlation_b{b}',lambda:k.partition(cov,b,'correlation',0))
        masks=k.pattern_masks(b)
        localq=k.local_plane(q,parts);localr=k.local_plane(r,parts)
        core=run(f'construct_core_b{b}',lambda:k.projected_core(localq,localr,masks))
        if args.transport_only:
            run(f'block_to_blocks_b{b}',lambda:k.block_to_blocks(core,parts,w));continue
        run(f'block_planes_b{b}',lambda:k.block_planes(core,parts,w))
        run(f'block_to_blocks_b{b}',lambda:k.block_to_blocks(core,parts,w))
        run(f'global_Q_to_blocks_b{b}',lambda:k.global_to_blocks(q,None,parts,w))
        run(f'global_QR_to_blocks_b{b}',lambda:k.global_to_blocks(q,r,parts,w))
        third=run(f'third_core_b{b}',lambda:k.third_core(aa,bb,parts))
        lc=k.local_plane(cov,parts);gain=fnp.ones((n//b,b),dtype=fnp.float32)*.6;second=gain*.4;thirdresp=gain*-.1
        run(f'nonlinear_core_b{b}',lambda:k.nonlinear_core(core,lc,third,gain,second,thirdresp))
        run(f'consistency_repair_b{b}',lambda:k.repair_core(core,lc))
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':
    apply_thread_limit(4)
    with warnings.catch_warnings():warnings.simplefilter('ignore');main()
