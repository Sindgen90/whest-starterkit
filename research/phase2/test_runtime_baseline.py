"""Independent identities and contract checks for the runtime derivative."""
from types import SimpleNamespace
import numpy as np
import flopscope.numpy as fnp
from whestbench import MLP
from benchmark_runtime import load,ROOT


def test_runtime_first_covariance_matches_gaussian_arccosine(monkeypatch):
    mod = load("venture_runtime")
    w = np.array([[1.,.3,-.4],[.2,1.,.3],[.1,-.2,1.]],dtype=np.float32)
    calls = []
    original = mod.flops.as_symmetric
    def capture(array,**kwargs):
        calls.append(np.asarray(array).copy())
        return original(array,**kwargs)
    monkeypatch.setattr(mod.flops,"as_symmetric",capture)
    est = mod.Estimator()
    est.setup(SimpleNamespace(seed=0,submission_dir=ROOT/"venture_runtime"))
    pred = np.asarray(est.predict(MLP(3,2,[fnp.asarray(w),fnp.eye(3,dtype=fnp.float32)],seed=0),2**41))
    sigma = np.linalg.norm(w.astype(np.float64),axis=0)
    rho = np.clip((w.astype(np.float64).T@w)/np.outer(sigma,sigma),-1,1)
    expected = np.outer(sigma,sigma)*(np.sqrt(1-rho*rho)+(np.pi-np.arccos(rho))*rho-1)/(2*np.pi)
    assert len(calls)==2  # Identity plus first covariance; no unused final covariance.
    np.testing.assert_allclose(calls[-1],expected,rtol=3e-5,atol=2e-7)
    np.testing.assert_allclose(pred[0],sigma/np.sqrt(2*np.pi),rtol=2e-6)


def test_runtime_nonstandard_depth_matches_original():
    mlp = MLP(4,18,[fnp.eye(4,dtype=fnp.float32)]*18,seed=0)
    rows = []
    for name in ("venture_baseline","venture_runtime"):
        est = load(name).Estimator()
        est.setup(SimpleNamespace(seed=0,submission_dir=ROOT/name))
        rows.append(np.asarray(est.predict(mlp,2**41)))
    assert rows[1].shape==(18,4) and np.isfinite(rows[1]).all()
    np.testing.assert_allclose(rows[1],rows[0],atol=2e-6,rtol=2e-6)
