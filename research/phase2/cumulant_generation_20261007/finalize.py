import hashlib,json,tarfile
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parent.parent.parent
freeze=json.loads((R/'confirmation_freeze.json').read_text())
for name,digest in freeze['methods'].items():assert hashlib.sha256((R/name/'estimator.py').read_bytes()).hexdigest()==digest
for name in ('centered','public_v29'):
    rec=json.loads((R/(name+'_archive.json')).read_text());path=Path(rec['archive']);assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256']
    with tarfile.open(path) as tar:
        assert sorted(x.name for x in tar.getmembers())==['LICENSE','PROVENANCE.md','estimator.py','manifest.json'];assert hashlib.sha256(tar.extractfile('estimator.py').read()).hexdigest()==rec['source_sha256']
    assert json.loads((R/(name+'_validate.json')).read_text())['ok']
files={}
for p in R.rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts or p.name=='inventory.json':continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(4*1024*1024):h.update(chunk)
    files[str(p.relative_to(R))]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
out=R/'inventory.json';assert not out.exists();out.write_text(json.dumps(files,indent=2));print('Final frozen sources, archives and',len(files),'study artifacts verified')
