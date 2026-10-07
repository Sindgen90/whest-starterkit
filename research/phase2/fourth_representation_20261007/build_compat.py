"""Package-contract compatibility for the official width-four smoke network.

At width >=8 this is the same prescribed D,b8 schedule, without arithmetic
changes. Original source/gate/results stay frozen and separately identified.
"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
src=ROOT/'selective_block8/estimator.py';text=src.read_text()
text=text.replace('self.block_size','block_width')
needle='n=mlp.width;nb=n//block_width';assert needle in text
text=text.replace(needle,'n=mlp.width;block_width=min(self.block_size,n);nb=n//block_width',1)
folder=ROOT/'selective_block8_compat';folder.mkdir(exist_ok=True);out=folder/'estimator.py';assert not out.exists()
out.write_text(text,encoding='utf-8')
(ROOT/'compatibility.json').write_text(json.dumps(dict(original_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),change='Use block width min(8,n) to satisfy the official width-four/depth-two validation contract. Identical arithmetic at all study widths 16,32,1024; no coefficient, schedule or partition changes.'),indent=2),encoding='utf-8')
