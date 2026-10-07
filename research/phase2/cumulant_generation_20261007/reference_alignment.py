import hashlib,json
from pathlib import Path
import numpy as np
from datasets import Dataset
R=Path(__file__).resolve().parent
first=Dataset.from_file('../tmp/phase2_mini_first8.arrow').with_format('numpy');held=Dataset.from_file(str(R/'public/confirmation_shard11.arrow')).with_format('numpy');rows=[];tree=json.loads((R/'public/phase2_tree.json').read_text())
for idx in (0,1,80,81):
    row=first[idx] if idx<2 else held[idx-79];path=R/'public/moments'/f'mlp_{idx:05d}.npz';d=np.load(path);provenance=json.loads((R/f'public/moments_id{idx}_provenance.json').read_text());h=hashlib.sha256()
    with path.open('rb') as f:
        while chunk:=f.read(4*1024*1024):h.update(chunk)
    assert provenance['sha256']==h.hexdigest();assert int(row['mlp_id'])==int(d['global_index'])==idx
    difference=float(np.max(np.abs(np.asarray(row['all_layer_means'],dtype=float)-d['gt_mean'])));assert difference<1e-6,difference
    ww=np.asarray(row['weights'],dtype=np.float64);prevar=d['pre_m2'][0]-d['pre_mean'][0]**2;analytic=np.sum(ww[0]**2,axis=0)
    item=next(x for x in tree if x['path']==f'moments/mlp_{idx:05d}.npz');remote=item.get('lfs',{}).get('oid');assert remote is None or remote==h.hexdigest()
    rows.append({'id':idx,'width':ww.shape[1],'depth':len(ww),'n_reference_samples':int(d['n_samples']),'baked_mean_copy_maxabs':difference,'reference_mean_final_mse_to_baked':float(np.mean((d['mean'][-1]-d['gt_mean'][-1])**2)),'first_prevariance_rms_to_exact':float(np.sqrt(np.mean((prevar-analytic)**2))),'weights_sha256':hashlib.sha256(ww.astype(np.float32).tobytes()).hexdigest(),'moment_file_sha256':h.hexdigest(),'matches_remote_lfs_hash':remote==h.hexdigest()})
(R/'reference_alignment.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
