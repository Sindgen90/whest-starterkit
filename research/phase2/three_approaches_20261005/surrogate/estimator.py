"""EXPERIMENTAL: fixed ReLU dictionary, independent fitted/residual samples.

All fitting, network evaluations and feature operations are inside predict.
No external targets or approximate control-variate centers are used.
"""
import flopscope.numpy as fnp
from whestbench import BaseEstimator


def dictionary(mlp):
    w = mlp.weights[0]
    norms2 = fnp.sum(w*w,axis=0)
    indices = fnp.argsort(norms2)[-min(16,mlp.width):]
    directions = w[:,indices]/fnp.sqrt(fnp.maximum(norms2[indices],1e-30))[None,:]
    norms = fnp.sqrt(fnp.sum(directions*directions,axis=0))
    exact = fnp.concatenate((fnp.ones(1,dtype=fnp.float32),fnp.zeros(len(indices),dtype=fnp.float32),norms/2.5066282746310002))
    return directions,exact


def features(x,directions):
    z = x@directions
    return fnp.concatenate((fnp.ones((len(x),1),dtype=fnp.float32),z,fnp.maximum(z,0)),axis=1)


def forward(x,weights):
    rows = []
    for w in weights:
        x = fnp.maximum(x@w,0)
        rows.append(x)
    return fnp.stack(rows,axis=1).reshape((len(x),-1))


def fit(mlp,rng,directions,n=512):
    x = rng.standard_normal((n,mlp.width)).astype(fnp.float32)
    design = features(x,directions)
    values = forward(x,mlp.weights)
    gram = design.T@design
    ridge = .001*fnp.sum(fnp.diag(gram))/gram.shape[0]
    return fnp.linalg.solve(gram+ridge*fnp.eye(gram.shape[0],dtype=fnp.float32),design.T@values)


class Estimator(BaseEstimator):
    random_seed = 20261211
    def predict(self,mlp,budget):
        rng = fnp.random.default_rng(self.random_seed+1009*int(mlp.seed or 0))
        directions,exact = dictionary(mlp)
        coefficients = fit(mlp,rng,directions)
        residual_sum = fnp.zeros(mlp.depth*mlp.width,dtype=fnp.float32)
        for _ in range(4):
            x = rng.standard_normal((1024,mlp.width)).astype(fnp.float32)
            residual = forward(x,mlp.weights)-features(x,directions)@coefficients
            residual_sum = residual_sum+fnp.sum(residual,axis=0)
        result = (exact@coefficients+residual_sum/4096).reshape((mlp.depth,mlp.width))
        w = mlp.weights[0]
        first = fnp.sqrt(fnp.sum(w*w,axis=0))/2.5066282746310002
        return fnp.concatenate((first[None,:],result[1:]),axis=0)
