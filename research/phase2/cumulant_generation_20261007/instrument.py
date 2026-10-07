"""Separate diagnostic copies. Instrumentation is never packaged or gated."""
from pathlib import Path
R=Path(__file__).resolve().parent
for name in ('baseline_snapshot','unclip34','centered','coherent'):
    s=(R/name/'estimator.py').read_text();s=s.replace('        self.trace = []','        self.trace = []\n        self.states = []')
    mark='            if layer==len(mlp.weights)-1:\n                break'
    assert s.count(mark)==1
    s=s.replace(mark,'            if layer==len(mlp.weights)-1:\n                self.states.append(dict(m=m,v=v,d3=d3,d4=d4,mu=mu))\n                break')
    mark='                gain = f[1][1]';assert s.count(mark)==1
    s=s.replace(mark,'                self.states.append(dict(m=m,v=v,d3=d3,d4=d4,mu=mu,post_var=fnp.diag(cov),post3=post3,post4=post4))\n'+mark)
    d=R/('debug_'+name);d.mkdir(exist_ok=True);assert not (d/'estimator.py').exists();(d/'estimator.py').write_text(s)
