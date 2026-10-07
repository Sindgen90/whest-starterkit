"""Freeze a standalone 12-layer joint closure and development-only readout."""
import ast
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument('--full-q2',action='store_true')
p.add_argument('--full-q1',action='store_true')
args = p.parse_args()
full = args.full_q2 or args.full_q1
report = json.loads((ROOT/'results/joint_cumulant_optimized_20261001.json').read_text())
method = 'k34_prefix12_q4'
source = (ROOT/'joint_cumulant.py').read_text()
helper = (ROOT/'puffi_reference.py').read_text()
functions = '\n\n'.join(ast.get_source_segment(helper,node) for node in ast.parse(helper).body
                        if isinstance(node,ast.FunctionDef) and node.name in ('responses','third_slices'))
baseline = (ROOT/'venture_runtime/estimator.py').read_text()
constants = '\n'.join(ast.get_source_segment(baseline,node) for node in ast.parse(baseline).body
                      if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in
                      ('_QUAD_NODES01','_QUAD_WEIGHTS01') for t in node.targets))
source = source.replace('from puffi_reference import responses,third_slices,new_third_source',functions)
source = source.replace('from venture_runtime.estimator import _QUAD_NODES01,_QUAD_WEIGHTS01',constants)
if full:
    source = source.replace('    quadrature_order = 8','    quadrature_order = '+('1' if args.full_q1 else '2'))
else:
    source = source.replace('    rich_layers = 16','    rich_layers = 12').replace('    quadrature_order = 8','    quadrature_order = 4')
    source = source.replace('        rows = []','        rows = []\n        designs = []')
    source = source.replace('            rows.append(mu)','            rows.append(mu)\n            mass = (fnp.sum(fnp.abs(pre),axis=1)-v)/n\n            designs.append(fnp.stack((mu,v,a,mass,fnp.ones(n,dtype=fnp.float32)),axis=-1))')
    source = source.replace('        return fnp.stack(rows)',
        '        coefficients = fnp.asarray(CALIBRATION,dtype=fnp.float32)\n'
        "        return fnp.stack(rows)+fnp.einsum('lnk,lk->ln',fnp.stack(designs),coefficients)")
    source += '\n\nCALIBRATION = '+repr(report['coefficients_frozen'][method])+'\n'
    source = source.replace('Experimental joint moment closure; no oracle targets or sampled moments.',
                        'Frozen joint moment candidate: twelve rich layers, four quadrature nodes.')
    source = source.replace('No fitted correction by default.',
                        'Fixed output-only readout fitted on public networks 0 and 1; never fed forward.')
folder = ROOT/('joint_q1_candidate' if args.full_q1 else 'joint_full_candidate' if full else 'joint_candidate')
folder.mkdir(exist_ok=True)
(folder/'estimator.py').write_text(source,encoding='utf-8')
(folder/'LICENSE').write_bytes((ROOT/'venture_runtime/LICENSE').read_bytes())
(folder/'PROVENANCE.md').write_text('''# Joint cumulant candidate, 1 October 2026

Standalone local extension of the Gaussian covariance baseline. Upstream
venture_agent_ai MIT attribution is preserved in LICENSE. Baseline repository:
https://github.com/agentaiventure-dot/arc-whest-estimator,
commit fb33bb5a66b4b0aed3022e6a70a9274d5957d61d.

Mixed cumulant response terms were inspired by the Puffi A716 write-up:
https://discourse.aicrowd.com/uploads/short-url/iDdUNo66HzI7ovRqhQqpLOFFugI.pdf
This is our approximate closure, not a reproduction of the published algorithm.

Retains full covariance, compressed third cumulants and paired fourth cumulants
for the first twelve layers; Gaussian propagation for the last four. Four-node
quadrature. Fixed five-feature readout coefficients fitted on public networks
0 and 1 only, using ridge 0.001. The readout does not affect propagated state.
No targets, sampled moments, datasets or inference-time fitting are bundled.
See ../JOINT_PROPAGATION.md for validation and limitations.
''',encoding='utf-8')
print(json.dumps({'source':str(folder/'estimator.py'),'sha256':hashlib.sha256((folder/'estimator.py').read_bytes()).hexdigest()}))
if full:
    provenance = (folder/'PROVENANCE.md').read_text()
    start = provenance.index('for the first twelve layers;')
    provenance = provenance[:start]+'''for all sixteen layers. Two-node quadrature, one third-cumulant source history.
No fitted calibration or learned coefficient is applied. No targets, sampled
moments, datasets or inference-time fitting are bundled.
See ../JOINT_PROPAGATION.md for validation and limitations.
'''
    if args.full_q1:
        provenance = provenance.replace('Two-node quadrature','One-node quadrature')
    (folder/'PROVENANCE.md').write_text(provenance,encoding='utf-8')
