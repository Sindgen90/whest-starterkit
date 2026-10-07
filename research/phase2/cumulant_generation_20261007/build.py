"""Asserted patches; incumbent remains byte-identical. Re-run only before measurement."""
from pathlib import Path
R=Path(__file__).resolve().parent
base=(R/'baseline_snapshot/estimator.py').read_text()
old='corrected_array = fnp.stack(moments[1:])+fnp.clip(delta,-bound,bound)'
new='corrected_array = fnp.stack(moments[1:])+fnp.concatenate((fnp.clip(delta[:2],-bound[:2],bound[:2]),delta[2:]),axis=0)'
assert base.count(old)==1
for name in ('unclip34',):
    d=R/name;d.mkdir(exist_ok=True);(d/'estimator.py').write_text(base.replace(old,new))
print('Built isolated unclip34')
