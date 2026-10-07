import hashlib,json,shutil
from pathlib import Path
R=Path(__file__).resolve().parent; P=R.parent; REPO=P.parent.parent
assert not (R/'preservation.json').exists()
files=list(P.rglob('*'))+list((REPO/'submissions').rglob('*'))
ledger={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file() and R not in p.parents and '__pycache__' not in p.parts}
(R/'preservation.json').write_text(json.dumps(ledger,indent=2))
(R/'baseline_snapshot').mkdir()
shutil.copyfile(P/'joint_full_runtime/estimator.py',R/'baseline_snapshot/estimator.py')
assert hashlib.sha256((R/'baseline_snapshot/estimator.py').read_bytes()).hexdigest()=='bd4c808d47068dcf2b0515472925598f56c43b2b603a2159b982d4d1f9dbce7e'
print('Frozen',len(ledger),'existing files')
