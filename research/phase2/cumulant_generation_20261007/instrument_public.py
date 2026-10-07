"""Diagnostics only; keep the public submission source unchanged."""
from pathlib import Path
R=Path(__file__).resolve().parent;s=(R/'public_v29/estimator.py').read_text()
mark='    def _predict_core(self, mlp: MLP, budget: int) -> fnp.ndarray:\n        _ = budget';assert s.count(mark)==1;s=s.replace(mark,mark+'\n        self.states = []')
mark='            pk1v, pk2v, pk3v, pk4v = PK1[0], PK1[1], PK1[2], PK1[3]';assert s.count(mark)==1
s=s.replace(mark,mark+'\n            state = dict(m=fnp.copy(mu),v=fnp.copy(var),d3=fnp.zeros(n,dtype=f32) if D3 is None else fnp.copy(D3),d4=fnp.zeros(n,dtype=f32) if g4row is None else fnp.copy(g4row),mu=fnp.copy(pk1v))\n            if not trim:\n                state["C"] = fnp.copy(C_pre)\n                state["P"] = fnp.zeros((n,n),dtype=f32) if D21 is None else fnp.copy(D21)\n            self.states.append(state)')
mark='            K2v, K3v, K4v = K[(2,)], K[(3,)], K[(4,)]';assert s.count(mark)==1;s=s.replace(mark,mark+'\n            self.states[-1].update(post_var=fnp.copy(K2v),post3=fnp.copy(K3v),post4=fnp.copy(K4v))')
d=R/'debug_public_v29';d.mkdir();(d/'estimator.py').write_text(s)
print('Public diagnostic copy built')
