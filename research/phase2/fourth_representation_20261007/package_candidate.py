"""Package only after preserved official gate passes; no upload."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args()
    gate=json.loads((ROOT/(a.name+'_gate.json')).read_text())
    assert all(r['returncode']==0 and r['summary']['n_failed_mlps']==0 for r in gate['records'])
    src=ROOT/a.name/'estimator.py';digest=hashlib.sha256(src.read_bytes()).hexdigest()
    assert digest==gate['source_sha256']
    archive=REPO/'submissions'/f'phase2_EXPERIMENTAL_{a.name}_20261007.tar.gz'
    assert not archive.exists()
    provenance=src.parent/'PROVENANCE.md';assert not provenance.exists()
    provenance.write_text(f'# EXPERIMENTAL — NOT RECOMMENDED\n\nFourth-order representation study: {a.name}. Baseline and previous archives preserved. No upload.\n\nEstimator SHA-256: {digest}\n\nFrozen protocol: research/phase2/fourth_representation_20261007/PROTOCOL.md.\nAll inference preparation is metered. Reference moments and rankings remain offline. Full derivations, ablations and failures are preserved in that research directory. Packaging follows the unchanged official subprocess gate at seed 20261071, two networks, four threads, default caps.\n\nUser-reported baseline leaderboard score: 8.62e-8; this is not a local measurement. Passing execution does not establish competitive accuracy. This experimental archive is not a promoted incumbent.\n',encoding='utf-8')
    with provenance.open('a',encoding='utf-8') as f:
        f.write('\nConfiguration: full covariance; unchanged one-generation third-source algorithm; full fourth-order cores in disjoint blocks of eight; correlation partition from first propagated covariance; enriched source layers 0 and 7 only. Width-four contract validation uses block width min(8,n); width-1024 arithmetic is identical to the original selective experiment. No fifth or higher cumulant state.\n')
        f.write('\nProtocol SHA-256: '+hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest()+'\n')
        f.write('\nOfficial execution summary: '+json.dumps(gate['records'][0]['summary'])+'\n')
        f.write('\nPaired public development nominal adjusted error: 1.36023048e-7, versus incumbent 9.13640801e-8. This is worse; no fresh held-out promotion was attempted. References are external offline evidence and are absent from this package.\n')
    shutil.copyfile(REPO/'LICENSE',src.parent/'LICENSE')
    exe=Path(sys.executable).with_name('whest.exe'); records=[]
    commands=[['validate','--estimator',str(src),'--seed','20261071','--format','json'],['package','--estimator',str(src.parent),'--output',str(archive),'--yes','--format','json'],['validate-package',str(archive),'--format','json']]
    for label,args in zip(('validate','package','validate_package'),commands):
        out=ROOT/f'{a.name}_{label}.json';err=ROOT/f'{a.name}_{label}.stderr.txt'
        assert not out.exists() and not err.exists()
        cmd=[str(exe),*args];result=subprocess.run(cmd,capture_output=True,encoding='utf-8',env=dict(os.environ,PYTHONIOENCODING='utf-8'))
        out.write_text(result.stdout,encoding='utf-8');err.write_text(result.stderr,encoding='utf-8')
        records.append({'command':cmd,'returncode':result.returncode,'report':str(out)})
        (ROOT/f'{a.name}_packaging_manifest.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(records[-1]),flush=True)
        assert result.returncode==0,result.stdout+result.stderr
    with tarfile.open(archive) as tf:
        members=[m for m in tf.getmembers() if m.name.endswith('estimator.py')]
        assert len(members)==1
        assert hashlib.sha256(tf.extractfile(members[0]).read()).hexdigest()==digest
        inventory=[m.name for m in tf.getmembers()]
    final={'archive':str(archive.resolve()),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'source_sha256':digest,'bytes':archive.stat().st_size,'members':inventory,'experimental_not_recommended':True,'official_gate_passed':True,'packaging_passed':True}
    (ROOT/f'{a.name}_archive.json').write_text(json.dumps(final,indent=2));print(json.dumps(final,indent=2),flush=True)
if __name__=='__main__':main()
