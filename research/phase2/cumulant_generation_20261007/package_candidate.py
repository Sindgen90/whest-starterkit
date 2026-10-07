"""Gate, held-out objective, unchanged contract validation, package. Never upload."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parent.parent.parent
def main():
    p=argparse.ArgumentParser();p.add_argument('name',choices=['unclip34','centered','public_v29']);a=p.parse_args();name=a.name
    report=json.loads((R/'confirmation.json').read_text());summary=report['summary'];assert summary[name]['nominal_adjusted']<summary['baseline_snapshot']['nominal_adjusted']
    gate=json.loads((R/(name+'_gate.json')).read_text());assert all(x['returncode']==0 and x['summary']['n_failed_mlps']==0 for x in gate['records'])
    src=R/name/'estimator.py';digest=hashlib.sha256(src.read_bytes()).hexdigest();freeze=json.loads((R/'confirmation_freeze.json').read_text());assert digest==freeze['methods'][name]==gate['source_sha256']==report['source_sha256'][name]
    public=name=='public_v29';kind='REPRODUCTION_504aldo_V29' if public else 'joint_'+name
    if public:
        extra=json.loads((R/'public_v29_confirmation_gate.json').read_text());assert extra['source_sha256']==digest and all(x['returncode']==0 and x['summary']['n_failed_mlps']==0 for x in extra['records'])
        mem=json.loads((R/'public_v29_confirmation_gate_memory.json').read_text());assert max(x['peak_working_set_bytes'] for x in mem['processes'])<8192*1024*1024,'Observed per-process resident peak exceeds published memory cap; do not package for promotion.'
    archive=REPO/'submissions'/f'phase2_{kind}_20261007.tar.gz';assert not archive.exists()
    if not (src.parent/'LICENSE').exists():shutil.copyfile(REPO/'LICENSE',src.parent/'LICENSE')
    provenance=src.parent/'PROVENANCE.md';assert not provenance.exists()
    text=f'# {kind}\n\nSource SHA256: {digest}\n\nFrozen protocol and complete results: research/phase2/cumulant_generation_20261007/. Incumbent and all earlier archives preserved. No AIcrowd upload. Official subprocess CLI defaults, four threads, two smoke networks at seed20261071 passed. Accuracy uses distinct reserved public mini IDs80..83, separate from development IDs0/1. Official smoke targets are not accuracy evidence.\n\nHeld-out aggregate nominal adjusted objective: {summary[name]["nominal_adjusted"]:.12g}; incumbent: {summary["baseline_snapshot"]["nominal_adjusted"]:.12g}. Only four confirmation networks: this is a local comparison, not a leaderboard score or a guarantee.\n\nNo sampled/reference arrays, evaluation targets, or offline diagnostics are included. All estimator inference preparation uses metered operations.\n'
    if public:
        text+='\nThis is a complete, byte-identical reproduction of 504aldo estimator_v29.py, MIT copyright(c)2026 504aldo. Original repository https://github.com/504aldo/whest-p2-cumulant-k3 at commit1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d. Original MIT notice is included in LICENSE. No algorithmic changes, new fitted parameters, or claim of original prize-eligible work. Published LAM/REF_R parameters are retained; published fitted mean correction is disabled by default. The Oct5 organizer clarification requires substantial original changes for prize consideration; this archive is a reproduction benchmark. A further four official smoke networks at seed20261171 passed. Windows resident-memory observation is not a reproduction of Linux RLIMIT_AS, and a six-network smoke check does not establish long-suite memory safety.\n'
    else:
        text+='\nOriginal local change: remove independent clipping of raw M3/M4 corrections. '+('Compute marginal moments about max(incoming_mean,0) and recenter consistently, preserving M1/M2 guards; avoid large positive-mean raw cancellation. Off-diagonal closure remains unchanged.\n' if name=='centered' else 'Keep every other incumbent operation and guard fixed.\n')
    provenance.write_text(text,encoding='utf-8')
    exe=Path(sys.executable).with_name('whest.exe');records=[]
    for label,args in [('validate',['validate','--estimator',str(src),'--seed','20261071','--format','json']),('package',['package','--estimator',str(src.parent),'--output',str(archive),'--yes','--format','json']),('validate_package',['validate-package',str(archive),'--format','json'])]:
        out=R/f'{name}_{label}.json';err=R/f'{name}_{label}.stderr.txt';assert not out.exists() and not err.exists();cmd=[str(exe),*args];result=subprocess.run(cmd,capture_output=True,encoding='utf-8',env=dict(os.environ,PYTHONIOENCODING='utf-8'));out.write_text(result.stdout);err.write_text(result.stderr);records.append({'command':cmd,'returncode':result.returncode,'report':str(out)});(R/f'{name}_packaging_manifest.json').write_text(json.dumps(records,indent=2));assert result.returncode==0,result.stdout+result.stderr
    with tarfile.open(archive) as tf:
        inventory=[m.name for m in tf.getmembers()];members=[m for m in tf.getmembers() if m.name.endswith('estimator.py')];assert len(members)==1 and hashlib.sha256(tf.extractfile(members[0]).read()).hexdigest()==digest;assert not any(x.endswith(('.npz','.arrow','.npy')) for x in inventory)
    rec={'archive':str(archive.resolve()),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'source_sha256':digest,'bytes':archive.stat().st_size,'members':inventory,'public_reproduction':public,'heldout_nominal_adjusted':summary[name]['nominal_adjusted']};(R/f'{name}_archive.json').write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
