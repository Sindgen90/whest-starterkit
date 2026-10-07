"""Phase-2 adaptation of published Kerdock cubature + Puffi A31--A37.

This is a partial-frame adaptation, not the complete spherical 5-design and
not a reproduction of a leader's current private Phase-2 submission.
Kerdock/FWHT construction adapted from Skye Nygaard (MIT; see LICENSE).
CV ridge=.25 and shrinkage=.75 follow Puffi's Phase-1 score appendix.
All inference numerical operations are metered through flopscope.
"""
from pathlib import Path
import flopscope.numpy as fnp
from whestbench import BaseEstimator, MLP
from reference import Estimator as Reference


def fwht(values):
    """Unnormalized Walsh transform on the penultimate axis."""
    frames, width, outputs = values.shape
    span = 1
    while span < width:
        blocks = values.reshape((frames,width//(2*span),2,span,outputs))
        left, right = blocks[:,:,0], blocks[:,:,1]
        values = fnp.stack((left+right,left-right),axis=2).reshape((frames,width,outputs))
        span *= 2
    return values


def ridge_correction(anchor, terminal, discrepancy, ridge):
    """Dual application of the ridge response; avoids building an n x n map."""
    x = anchor - fnp.mean(anchor,axis=0)
    y = terminal - fnp.mean(terminal,axis=0)
    gram = x.T @ x
    regularizer = fnp.maximum(ridge*fnp.trace(gram)/gram.shape[0], 1e-12)
    gram = gram + regularizer*fnp.eye(gram.shape[0],dtype=fnp.float32)
    v = fnp.linalg.solve(gram,discrepancy)
    return y.T @ (x @ v)


class Estimator(BaseEstimator):
    frames = 8
    anchor_layer = 5  # one-based; zero disables the reference and CV
    ridge = 0.25
    strength = 0.75

    def setup(self, context):
        root = Path(context.submission_dir) if context.submission_dir else Path(__file__).parent
        self.assets = fnp.load(str(root / "design.npz"))

    def predict(self, mlp, budget):
        width, depth = mlp.width, mlp.depth
        key = f"chirps_{width}"
        if key not in self.assets:
            # Generic genuine estimator for unsupported smoke-test dimensions.
            ref = Reference()
            ref.reference_layers = 0
            return ref.predict(mlp,budget)
        anchor_layer = min(self.anchor_layer,depth)
        if depth == 1:
            w = mlp.weights[0]
            return (fnp.sqrt(fnp.sum(w*w,axis=0))*0.3989422804014327)[None,:]
        # Conservatively reserve rich-reference and regression arithmetic.
        reserve = (12*anchor_layer*anchor_layer+20)*width**3 if anchor_layer else 3*width**3
        per_frame = 4*(depth+2)*width**3 + 50*width**2
        affordable = int(max(0,0.90*budget-reserve)/per_frame)
        count = min(self.frames,len(self.assets[key]),affordable)
        if count < 1:
            ref = Reference()
            ref.reference_layers = 0
            return ref.predict(mlp,budget)
        weight = fnp.asarray(mlp.weights[0],dtype=fnp.float32)
        rotated = self.assets[f"rotation_{width}"] @ weight
        pre = fwht(self.assets[key][:count,:,None]*rotated[None,:,:])
        pre = pre * (self.assets[f"radius_{width}"]/fnp.sqrt(fnp.asarray(width)))
        pre = pre.astype(fnp.float32).reshape((-1,width))
        y = fnp.maximum(fnp.concatenate((pre,-pre),axis=0),0.0)
        rows = [fnp.mean(y,axis=0)]
        anchor = y if anchor_layer == 1 else None
        for layer,w in enumerate(mlp.weights[1:],start=2):
            y = fnp.maximum(y @ fnp.asarray(w,dtype=fnp.float32),0.0)
            rows.append(fnp.mean(y,axis=0))
            if layer == anchor_layer:
                anchor = y
        if anchor_layer:
            ref = Reference()
            ref.reference_layers = anchor_layer
            prefix = MLP(width,anchor_layer,mlp.weights[:anchor_layer],seed=mlp.seed)
            analytic = ref.predict(prefix,budget)
            discrepancy = rows[anchor_layer-1]-analytic[-1]
            correction = ridge_correction(anchor,y,discrepancy,self.ridge)
            rows[-1] = fnp.maximum(rows[-1]-self.strength*correction,0.0)
        rows[0] = fnp.sqrt(fnp.sum(weight*weight,axis=0))*0.3989422804014327
        return fnp.stack(rows)
