"""Final read-only hash comparison plus inventory of this append-only study."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    ledger=json.loads((ROOT/'preservation.json').read_text());changed=[];missing=[]
    for rel,sha in ledger.items():
        p=REPO/rel
        if not p.exists():missing.append(str(p))
        elif digest(p)!=sha:changed.append(str(p))
    check=dict(files_checked=len(ledger),changed=changed,missing=missing,all_preserved=not changed and not missing)
    (ROOT/'preservation_verification.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    excluded={'inventory.json','finish_jobs_stdout.txt','preservation_stdout.txt'}
    files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in excluded}
    (ROOT/'inventory.json').write_text(json.dumps(dict(root=str(ROOT),excluded_active_logs=sorted(excluded),files=files),indent=2),encoding='utf-8')
    print(json.dumps(check),flush=True);assert check['all_preserved']
if __name__=='__main__':main()
