"""Same official seed/spec/source, local versus preserved subprocess results."""
import json,subprocess,sys,os
from common import ROOT
def main():
    out=[]
    for name in ('baseline_snapshot','selective_block8','qr'):
        prefix=ROOT/(name+'_local_timing')
        cmd=[sys.executable,str(ROOT.parent/'benchmark_joint_official.py'),'--candidate','fourth_representation_20261007/'+name,'--runners','local','--seeds','20261071','--n-mlps','2','--prefix',str(prefix)]
        log=ROOT/(name+'_local_timing_stdout.txt');assert not log.exists()
        with log.open('w',encoding='utf-8') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONIOENCODING='utf-8'))
        local=json.loads(prefix.with_suffix('.json').read_text());remote=json.loads((ROOT/(name+'_gate.json')).read_text())
        assert local['source_sha256']==remote['source_sha256']
        out.append(dict(method=name,local=local['records'][0],subprocess=remote['records'][0],same_seed_spec_source=True))
        (ROOT/'timing_pair_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
        print(json.dumps(dict(method=name,local_returncode=result.returncode)),flush=True)
if __name__=='__main__':main()
