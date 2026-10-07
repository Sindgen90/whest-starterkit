"""Summarize frozen evidence, check preservation, and inventory artifacts."""
from pathlib import Path
import hashlib,json,subprocess,sys,tarfile
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=ROOT/'comparison.json';assert not out.exists()
    screens={key:json.loads((ROOT/file).read_text()) for key,file in [('surrogate','surrogate_screen_v2.json'),('allocation','allocation_screen.json'),('conditional','conditional_screen.json')]}
    sources={name:digest(ROOT/name/'estimator.py') for name in ('baseline_snapshot','surrogate','allocation','conditional','random_control','repair_diagnostic')}
    records=[]
    for name,screen in [('baseline',screens['surrogate']),('surrogate',screens['surrogate']),('norm_control',screens['allocation']),('allocation',screens['allocation']),('conditional',screens['conditional']),('random_control',screens['conditional'])]:
        summary=screen['summary'][name];rec={'method':name,**summary}
        selected=[r for r in screen['records'] if r['method']==name]
        rec['per_case']=[{k:r[k] for k in ('mlp_id','random_seed','raw_mse','nominal_adjusted','finite')} for r in selected]
        if name!='norm_control':
            gate=json.loads((ROOT/(name+'_gate.json')).read_text());g=gate['records'][0]['summary']
            assert gate['source_sha256']==(digest(ROOT.parent/'joint_full_runtime/estimator.py') if name=='baseline' else sources[name])
            rec['official_gate']=g;rec['execution_eligible']=g['n_failed_mlps']==0
            rec['subprocess_wall_range']=[min(m['wall_time_s'] for m in g['per_mlp']),max(m['wall_time_s'] for m in g['per_mlp'])]
            rec['subprocess_residual_range']=[min(m['residual_wall_time_s'] for m in g['per_mlp']),max(m['residual_wall_time_s'] for m in g['per_mlp'])]
        records.append(rec)
    ledger=json.loads((ROOT/'preservation.json').read_text())
    changed=[p for p,sha in ledger.items() if not (REPO/p).is_file() or digest(REPO/p)!=sha]
    preservation={'checked':len(ledger),'changed':changed,'all_unchanged':not changed,'baseline_snapshot_equal':digest(ROOT/'baseline_snapshot/estimator.py')==digest(ROOT.parent/'joint_full_runtime/estimator.py')}
    (ROOT/'preservation_verification.json').write_text(json.dumps(preservation,indent=2));assert not changed and preservation['baseline_snapshot_equal']
    best=REPO/'submissions/phase2_joint_full_runtime_20261002.tar.gz'
    with tarfile.open(best) as tf:
        member=[m for m in tf.getmembers() if m.name.endswith('estimator.py')];assert len(member)==1
        assert hashlib.sha256(tf.extractfile(member[0]).read()).hexdigest()==sources['baseline_snapshot']
    cmd=[str(Path(sys.executable).with_name('whest.exe')),'validate-package',str(best),'--format','json']
    p=subprocess.run(cmd,capture_output=True,text=True);assert p.returncode==0
    (ROOT/'baseline_archive_validation.json').write_text(p.stdout)
    (ROOT/'baseline_archive_validation.stderr.txt').write_text(p.stderr)
    (ROOT/'baseline_archive_validation_command.json').write_text(json.dumps({'command':cmd,'returncode':p.returncode},indent=2))
    archives=[json.loads((ROOT/(name+'_archive.json')).read_text()) for name in ('surrogate','allocation')]
    for a in archives:assert digest(Path(a['archive']))==a['sha256']
    report={'protocol_sha256':digest(ROOT/'PROTOCOL.md'),'source_sha256':sources,'records':records,'new_archives':archives,'best_validated_archive':{'path':str(best),'sha256':digest(best),'recommendation':'Retain existing best. No fresh-reference trigger or promotion criterion met. User-reported leaderboard score 8.62e-8 is not a local result.'},'preservation':preservation,'fresh_networks_generated':False,'uploaded':False}
    out.write_text(json.dumps(report,indent=2))
    paths=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='final_manifest.json']
    manifest={'artifacts':{str(p.relative_to(REPO)):digest(p) for p in paths},'new_archives':{a['archive']:a['sha256'] for a in archives},'source_sha256':sources}
    (ROOT/'final_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'preservation':preservation,'artifact_count':len(paths),'new_archives':[{k:a[k] for k in ('archive','sha256')} for a in archives],'best':report['best_validated_archive']},indent=2))
if __name__=='__main__':main()
