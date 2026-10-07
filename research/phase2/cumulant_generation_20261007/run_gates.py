"""Serial unchanged official helper, with read-only Windows memory observation."""
import argparse,ast,ctypes,json,os,subprocess,sys,time
from ctypes import wintypes as wt
from pathlib import Path
R=Path(__file__).resolve().parent
source=ast.parse((R.parent/'fourth_representation_20261007/gate_all.py').read_text());function=next(x for x in source.body if isinstance(x,ast.FunctionDef) and x.name=='process_memory_tree');exec(compile(ast.Module(body=[function],type_ignores=[]),'preserved_memory_observer','exec'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',required=True);p.add_argument('--n-mlps',type=int,default=2);p.add_argument('--seed',type=int,default=20261071);p.add_argument('--suffix',default='gate');a=p.parse_args()
    for name in a.names:
        prefix=R/(name+'_'+a.suffix);log=R/(name+'_'+a.suffix+'_stdout.txt');assert not log.exists()
        cmd=[sys.executable,str(R.parent/'benchmark_joint_official.py'),'--candidate',R.name+'/'+name,'--seeds',str(a.seed),'--n-mlps',str(a.n_mlps),'--prefix',str(prefix)]
        with log.open('w') as out:
            proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONIOENCODING='utf-8'));seen={};peak=0;count=0
            while proc.poll() is None:
                rows=process_memory_tree(proc.pid);count+=1;peak=max(peak,sum(r['working_set_bytes'] for r in rows))
                for row in rows:
                    pid=row['pid'];seen[pid]=dict(row,peak_working_set_bytes=max(row['peak_working_set_bytes'],seen.get(pid,{}).get('peak_working_set_bytes',0)))
                time.sleep(.5)
        rec={'method':name,'returncode':proc.returncode,'sampled_tree_peak_bytes':peak,'observations':count,'period_s':.5,'processes':list(seen.values()),'limitation':'OS process lifetime peaks and sampled simultaneous RSS, including CLI/worker/backend/helper; can miss short-lived children. Official Windows gate is unchanged; Linux RLIMIT_AS is not reproduced.'};(R/(name+'_'+a.suffix+'_memory.json')).write_text(json.dumps(rec,indent=2));print(json.dumps({k:v for k,v in rec.items() if k not in ('processes','limitation')}),flush=True)
if __name__=='__main__':main()
