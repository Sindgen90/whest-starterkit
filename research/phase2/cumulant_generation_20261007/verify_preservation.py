import hashlib,importlib.metadata,json
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parent.parent.parent
ledger=json.loads((R/'preservation.json').read_text());checked={};mismatch=[]
for name,digest in ledger.items():
    p=REPO/name;got=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None;checked[name]=got
    if got!=digest:mismatch.append(name)
old=json.loads((R.parent/'three_approaches_20261005/preservation.json').read_text());sdk={}
for name,digest in old.items():
    if str(name).startswith('.venv-phase2'):
        p=REPO/name;got=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None;sdk[name]={'expected':digest,'actual':got}
        if got!=digest:mismatch.append(name)
record={'existing_files_checked':len(ledger),'preexisting_sha256_mismatches':mismatch,'sdk_against_prior_study':sdk,'versions':{p:importlib.metadata.version(p) for p in ('whestbench','flopscope','numpy','datasets','threadpoolctl')},'baseline_sha256':hashlib.sha256((R.parent/'joint_full_runtime/estimator.py').read_bytes()).hexdigest()};(R/'preservation_verification.json').write_text(json.dumps(record,indent=2));assert not mismatch,mismatch;print('Preserved',len(ledger),'existing files and',len(sdk),'SDK sources')
