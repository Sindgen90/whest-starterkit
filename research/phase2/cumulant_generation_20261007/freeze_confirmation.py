import hashlib,json,os
from pathlib import Path
R=Path(__file__).resolve().parent;names=('baseline_snapshot','unclip34','centered','coherent','public_v25','public_v29');path=R/'confirmation_freeze.json';assert not path.exists()
env={k:v for k,v in os.environ.items() if k.startswith(('V17_','V18_','V19_','V21_','V24_','V25_','V26_','V27_','V28_','V29_','STRASSEN_'))};assert not env,env
record=dict(methods={n:hashlib.sha256((R/n/'estimator.py').read_bytes()).hexdigest() for n in names},heldout_global_ids=[80,81,82,83],development_global_ids=[0,1],rule='Evaluate all frozen variants once; require lower aggregate held-out adjusted objective and official subprocess pass. No changes after targets are examined. Public reproductions are benchmarks; no claim of original prize eligibility.',public_environment_overrides=env)
path.write_text(json.dumps(record,indent=2));print(json.dumps(record))
