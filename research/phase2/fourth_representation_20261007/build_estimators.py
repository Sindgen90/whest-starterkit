"""Generate self-contained experimental files without changing the incumbent."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
base=(ROOT/'baseline_snapshot/estimator.py').read_text()
helpers=base[:base.index('class Estimator(BaseEstimator):')]
kernel=(ROOT/'kernels.py').read_text()
closure=(ROOT/'closure.py').read_text()
selection=ROOT/'partition_selection.json'
rule=json.loads(selection.read_text())['selected_rule'] if selection.exists() else 'contiguous'
for name,family,b in [('qr','QR',8),('block8','D',8),('block16','D',16),('qblock8','E',8),('qblock16','E',16),('qrblock8','F',8),('qrblock16','F',16),('exact_q_control','Q',8)]:
    folder=ROOT/name;folder.mkdir(exist_ok=True)
    text=helpers+'\n'+kernel+'\n'+closure.replace("family='QR'",f"family='{family}'").replace('block_size=8',f'block_size={b}').replace("partition_rule='contiguous'",f"partition_rule='{rule}'")
    if name=='exact_q_control':text=text.replace('complete_pair=False','complete_pair=True')
    path=folder/'estimator.py'
    if path.exists() and path.read_text()!=text:
        old=path.read_bytes();history=ROOT/'source_history';history.mkdir(exist_ok=True)
        saved=history/(name+'_'+hashlib.sha256(old).hexdigest()+'.py')
        if not saved.exists():saved.write_bytes(old)
    path.write_text(text,encoding='utf-8')
# Baseline diagnostic copy: post-layer moments are observed without changing arithmetic.
diagnostic=base.replace('        self.trace = []','        self.trace = []\n        self.states = []').replace('            cov = flops.as_symmetric(cov,symmetry=(0,1))','            cov = flops.as_symmetric(cov,symmetry=(0,1))\n            self.states.append(dict(mu=mu,cov=cov,q=fourth,r=None,core=None,parts=None,pre=pre,d3=d3,d4=d4,alpha=None))')
folder=ROOT/'baseline_diagnostic';folder.mkdir(exist_ok=True)
(folder/'estimator.py').write_text(diagnostic,encoding='utf-8')
