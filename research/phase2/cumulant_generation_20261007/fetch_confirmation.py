import hashlib,json,urllib.request
from pathlib import Path
R=Path(__file__).resolve().parent/'public'
repo='aicrowd/arc-whestbench-public-2026'
meta=json.load(urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}/revision/v2-phase2',timeout=120));sha=meta['sha'];name='prepared/mini/data-00011-of-00014.arrow';path=R/'confirmation_shard11.arrow';assert not path.exists()
url=f'https://huggingface.co/datasets/{repo}/resolve/{sha}/{name}';h=hashlib.sha256()
with urllib.request.urlopen(url,timeout=120) as src,path.open('wb') as dst:
    while chunk:=src.read(4*1024*1024):dst.write(chunk);h.update(chunk)
tree=json.loads((R/'official_tree.json').read_text());expected=next(x['lfs']['oid'] for x in tree if x['path']==name);assert h.hexdigest()==expected
(R/'confirmation_provenance.json').write_text(json.dumps({'repository':repo,'commit':sha,'url':url,'sha256':h.hexdigest(),'bytes':path.stat().st_size,'reserved_global_ids':[80,81,82,83],'note':'Shard contains IDs79..85; confirmation uses only frozen80..83.'},indent=2));print(path,path.stat().st_size,flush=True)
