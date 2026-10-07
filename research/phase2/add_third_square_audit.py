"""One extra analytical ablation identified by the audit; no coefficient fitting."""
import hashlib
import json
from benchmark_cumulant_audit import ROOT,AUDIT
from start_cumulant_audit import replace_once

source = (ROOT/'joint_full_runtime/estimator.py').read_text()
source = replace_once(source,'    kernels = fnp.stack((pair4,pair3,pair3.T,pc,pc.T,pair3*pair3.T,dc,dc.T))',
                      '    square3 = pair3*pair3\n    kernels = fnp.stack((pair4,pair3,pair3.T,pc,pc.T,pair3*pair3.T,dc,dc.T,square3,square3.T))')
source = replace_once(source,'    orders = ((2,2),(2,1),(1,2),(3,2),(2,3),(3,3),(4,1),(1,4))',
                      '    coefficients = fnp.concatenate((coefficients,fnp.asarray((1/8,1/8),dtype=fnp.float32)[:,None]),axis=0)\n    orders = ((2,2),(2,1),(1,2),(3,2),(2,3),(3,3),(4,1),(1,4),(4,2),(2,4))')
folder = AUDIT/'third_square'
assert not folder.exists()
folder.mkdir()
(folder/'estimator.py').write_text(source,encoding='utf-8')
(AUDIT/'third_square_protocol.json').write_text(json.dumps({
    'source_sha256':hashlib.sha256((folder/'estimator.py').read_bytes()).hexdigest(),
    'timing':'Added after original screen and memory confirmation, before any fresh-seed reference draws.',
    'derivation':'The mixed third operator is D3=Pij/2*d_i^2*d_j+Pji/2*d_i*d_j^2. In D3^2/2, Pij*Pji/4*d_i^3*d_j^3 is already retained. Pij^2/8*d_i^4*d_j^2 and Pji^2/8*d_i^2*d_j^4 were missing. Add both to all three covariance planes; no fifth-to-eighth cumulant state.',
    'selection':'One fixed ablation on identical networks 0-3. Only if it outperforms compressed history advance to excluded IDs 8-11 and official gate; freeze one choice before fresh-seed reference sampling.',
    'parameters':'Analytical 1/8, no fitting, no combination, no grid.'},indent=2),encoding='utf-8')
print('Created',folder)
