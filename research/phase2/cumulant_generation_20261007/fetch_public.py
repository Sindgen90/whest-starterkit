"""Download pinned public research artifacts only; never execute downloaded scripts."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
R=Path(__file__).resolve().parent
def read(url):
    req=urllib.request.Request(url,headers={'User-Agent':'ARC-focused-reproduction'})
    with urllib.request.urlopen(req,timeout=120) as s:return s.read()
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['code','moments']);p.add_argument('--ids',nargs='+',type=int,default=[0,1,80,81]);a=p.parse_args()
    d=R/'public';d.mkdir(exist_ok=True)
    if a.mode=='code':
        sha='1a4083f89184d6b1cc55d2e55a5a561d2ae50a8d';names=['LICENSE','README.md','estimators/estimator_v25.py','estimators/estimator_v29.py','docs/derivation_code_map.md','docs/community_post.md']
        record={'repository':'https://github.com/504aldo/whest-p2-cumulant-k3','commit':sha,'files':{}}
        for name in names:
            path=d/'504aldo'/name;assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);url=f'https://raw.githubusercontent.com/504aldo/whest-p2-cumulant-k3/{sha}/{name}';path.write_bytes(read(url));record['files'][name]={'url':url,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};print(name,path.stat().st_size,flush=True)
        (d/'504aldo_provenance.json').write_text(json.dumps(record,indent=2))
        for label,url in [('phase2_dataset','https://huggingface.co/datasets/keenanpepper/arc-whestbench-p2-higher-moments-2026/raw/main/README.md'),('phase1_dataset','https://huggingface.co/datasets/keenanpepper/arc-whestbench-higher-moments-2026/raw/main/README.md'),('phase2_tree','https://huggingface.co/api/datasets/keenanpepper/arc-whestbench-p2-higher-moments-2026/tree/main?recursive=true'),('official_tree','https://huggingface.co/api/datasets/aicrowd/arc-whestbench-public-2026/tree/v2-phase2?recursive=true')]:
            (d/(label+('.json' if 'tree' in label else '.md'))).write_bytes(read(url))
    else:
        meta=json.loads(read('https://huggingface.co/api/datasets/keenanpepper/arc-whestbench-p2-higher-moments-2026'));sha=meta['sha'];(d/'moments_metadata.json').write_text(json.dumps(meta,indent=2))
        for idx in a.ids:
            name=f'moments/mlp_{idx:05d}.npz';path=d/name;assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);url=f'https://huggingface.co/datasets/keenanpepper/arc-whestbench-p2-higher-moments-2026/resolve/{sha}/{name}';req=urllib.request.Request(url,headers={'User-Agent':'ARC-focused-reproduction'});h=hashlib.sha256()
            with urllib.request.urlopen(req,timeout=120) as src,path.open('wb') as dst:
                while chunk:=src.read(4*1024*1024):dst.write(chunk);h.update(chunk)
            (d/f'moments_id{idx}_provenance.json').write_text(json.dumps({'repository':meta['id'],'commit':sha,'url':url,'bytes':path.stat().st_size,'sha256':h.hexdigest()},indent=2));print('Downloaded',idx,path.stat().st_size,flush=True)
if __name__=='__main__':main()
