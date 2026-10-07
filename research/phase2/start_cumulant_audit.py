"""Freeze existing evidence and create three independent, standalone ablations."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'cumulant_audit_20261005'


def replace_once(text, before, after):
    assert text.count(before) == 1, before[:100]
    return text.replace(before, after)


def main():
    assert not OUT.exists(), 'Preserve prior runs: choose a new directory.'
    frozen = {}
    paths = list(ROOT.rglob('*')) + list((ROOT.parents[1] / 'submissions').glob('*.tar.gz'))
    paths += list((ROOT.parents[1] / '.venv-phase2/Lib/site-packages/whestbench').rglob('*.py'))
    paths += list((ROOT.parents[1] / '.venv-phase2/Lib/site-packages/flopscope').rglob('*.py'))
    for path in paths:
        if path.is_file() and '__pycache__' not in path.parts and path != Path(__file__):
            frozen[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    OUT.mkdir()
    (OUT / 'preserved.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    baseline = (ROOT / 'joint_full_runtime/estimator.py').read_text(encoding='utf-8')
    assert hashlib.sha256((ROOT / 'joint_full_runtime/estimator.py').read_bytes()).hexdigest() == 'bd4c808d47068dcf2b0515472925598f56c43b2b603a2159b982d4d1f9dbce7e'
    memory = replace_once(baseline, '    history = 1', '    history = 16')
    before = '''                sources = [(born,gain[:,None]*aa,gain[:,None]*bb)
                           for born,aa,bb in sources if layer-born<self.history]'''
    after = '''                sources = [(born,gain[:,None]*aa,gain[:,None]*bb)
                           for born,aa,bb in sources if layer-born<self.history]
                # Retain exactly 16 older CP columns, ranked by a tensor-norm bound.
                # Selection uses only the current inferred state and supplied weights.
                if sources:
                    aa = fnp.concatenate([aa for born,aa,bb in sources],axis=1)
                    bb = fnp.concatenate([bb for born,aa,bb in sources],axis=1)
                    scores = fnp.sqrt(fnp.sum(aa*aa,axis=0))*fnp.sum(bb*bb,axis=0)
                    selected = fnp.argsort(scores)[-16:]
                    sources = [(layer,aa[:,selected],bb[:,selected])]'''
    memory = replace_once(memory, before, after)
    # Birth is reset for the compressed pool; all older selected columns can survive.
    cross_helper = '''def rank_one_cross(pair4, fourth, w):
    """One signed mode of Q_off; exact paired diagonal remains supplied separately."""
    off = fourth.copy()
    fnp.fill_diagonal(off,0)
    u = fnp.ones(w.shape[0],dtype=fnp.float32)
    u = u/fnp.sqrt(fnp.sum(u*u))
    for _ in range(8):
        u = off@u
        u = u/fnp.maximum(fnp.sqrt(fnp.sum(u*u)),1e-20)
    eigenvalue = fnp.sum(u*(off@u))
    mode = w.T@(u[:,None]*w)
    # Q_off has zero diagonal. Remove the approximate mode's diagonal contribution.
    square = w*w
    correction = 2*eigenvalue*(mode*mode-square.T@((u*u)[:,None]*square))
    return pair4+correction


'''
    cross = replace_once(baseline, 'class Estimator(BaseEstimator):', cross_helper+'class Estimator(BaseEstimator):')
    cross = replace_once(cross, '                    # Retain the actual projected Gram plane instead of a row-mean surrogate.',
                         '                    if layer in (1,4):\n                        pair4 = rank_one_cross(pair4,fourth,w)\n                    # Restore the exact diagonal for the retained paired tensor.')
    # Add the leading covariance contraction of the retained marginal fourth cumulant.
    response = replace_once(baseline, 'def nong_terms(cov,pair3,pair4,d3,f,need_higher=True,coefficients=None):',
                           'def nong_terms(cov,pair3,pair4,d3,f,need_higher=True,coefficients=None,d4=None):')
    response = replace_once(response, '    return moments, [None, f1, f2, f3, f4], sigma, r, phi',
                           '    f1.append(-(r*r*r-3*r)*phi/(variance*variance))\n    f2.append(2*f1[4])\n    return moments, [None, f1, f2, f3, f4], sigma, r, phi')
    response = replace_once(response, '    kernels = fnp.stack((pair4,pair3,pair3.T,pc,pc.T,pair3*pair3.T,dc,dc.T))',
                           '    d4cov = d4[:,None]*cov\n    kernels = fnp.stack((pair4,pair3,pair3.T,pc,pc.T,pair3*pair3.T,dc,dc.T,d4cov,d4cov.T))')
    response = replace_once(response, '    orders = ((2,2),(2,1),(1,2),(3,2),(2,3),(3,3),(4,1),(1,4))',
                           '''    coefficients = fnp.concatenate((coefficients,fnp.asarray((1/24,1/24),dtype=fnp.float32)[:,None]),axis=0)
    orders = ((2,2),(2,1),(1,2),(3,2),(2,3),(3,3),(4,1),(1,4),(5,1),(1,5))''')
    response = replace_once(response, 'e11,e21,e22 = nong_terms(pre,pair3,pair4,d3,f,need_higher=need_state,coefficients=coefficients)',
                           'e11,e21,e22 = nong_terms(pre,pair3,pair4,d3,f,need_higher=need_state,coefficients=coefficients,d4=d4)')
    candidates = {'third_memory16': memory, 'fourth_cross1': cross, 'fourth_response': response}
    hashes = {}
    for name, source in candidates.items():
        folder = OUT / name
        folder.mkdir()
        (folder / 'estimator.py').write_text(source, encoding='utf-8')
        hashes[name] = hashlib.sha256((folder / 'estimator.py').read_bytes()).hexdigest()
    protocol = {
        'baseline_sha256': hashlib.sha256((ROOT / 'joint_full_runtime/estimator.py').read_bytes()).hexdigest(),
        'versions': {d: importlib.metadata.version(d) for d in ('whestbench','flopscope','numpy','datasets')},
        'candidate_sha256': hashes,
        'screen': 'Identical exposed public networks 0,1,2,3; no fitting or parameter sweep.',
        'confirmation': 'Public IDs 8-11 excluded from this candidate selection, previously exposed in older studies; not a pristine holdout.',
        'fresh_validation': 'New synthetic seeds 20261051,20261052 if a candidate survives screening; fresh offline independent sampling only as needed.',
        'selection': 'Run all three frozen single-change ablations on 0-3. Advance only a candidate with >=2% mean nominal adjusted improvement and no numerical failure. Require confirmation improvement and unchanged official subprocess gate. No composition or retuning.',
        'parameters': {'third_memory16': 'Top 16 older CP columns by ||A_r||_2 ||B_r||_2^2, retaining the full new source; no learned coefficients.',
                       'fourth_cross1': 'One signed Q_off mode only at zero-based layers 1,4 (the earlier diagnostic favored early sources); eight deterministic power iterations from ones; subtract its diagonal contribution and restore the original exact paired diagonal.',
                       'fourth_response': 'Add kappa4_i*Cij*F5_i*F1_j/24 and its transpose to all three raw covariance planes.'},
        'limits': {'width':1024,'depth':16,'flop_budget':2**41,'lambda':0,'residual_cap_s':.4,'wall_cap_s':120,'threads':4},
        'score': 'MSE * max(0.1, FLOPs/2**41); eligibility requires official subprocess caps, failed predictions are zeroed.',
        'local_error': 'Reuse higher_moments_20261001_mlp0-3.npz, no new reference draws for initial diagnosis; no references in any estimator.'}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    print(json.dumps({'preserved_files':len(frozen),'candidates':hashes},indent=2))


if __name__ == '__main__':
    main()
