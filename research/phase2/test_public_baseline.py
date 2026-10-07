"""Independent first-layer Gaussian identities for the unchanged public source."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import flopscope.numpy as fnp
from whestbench import MLP

ROOT = Path(__file__).parent/"venture_baseline"
spec = importlib.util.spec_from_file_location("venture_test",ROOT/"estimator.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_first_layer_mean_and_covariance_match_arccosine_identity(monkeypatch):
    w = np.array([[1.,.3,-.4],[.2,1.,.3],[.1,-.2,1.]],dtype=np.float32)
    calls = []
    original = module.flops.as_symmetric
    def capture(array,**kwargs):
        calls.append(np.asarray(array).copy())
        return original(array,**kwargs)
    monkeypatch.setattr(module.flops,"as_symmetric",capture)
    estimator = module.Estimator()
    estimator.setup(SimpleNamespace(seed=0,submission_dir=ROOT))
    result = np.asarray(estimator.predict(MLP(3,1,[fnp.asarray(w)],seed=0),2**41))
    sigma = np.linalg.norm(w.astype(np.float64),axis=0)
    rho = (w.astype(np.float64).T@w)/np.outer(sigma,sigma)
    rho = np.clip(rho,-1,1)
    expected = np.outer(sigma,sigma)*(np.sqrt(1-rho*rho)+(np.pi-np.arccos(rho))*rho-1)/(2*np.pi)
    np.testing.assert_allclose(result[0],sigma/np.sqrt(2*np.pi),rtol=2e-6,atol=2e-7)
    np.testing.assert_allclose(calls[-1],expected,rtol=3e-5,atol=2e-7)


def test_nonstandard_depth_skips_calibration_safely():
    estimator = module.Estimator()
    estimator.setup(SimpleNamespace(seed=0,submission_dir=ROOT))
    mlp = MLP(4,18,[fnp.eye(4,dtype=fnp.float32)]*18,seed=0)
    result = np.asarray(estimator.predict(mlp,2**41))
    assert result.shape==(18,4)
    assert np.isfinite(result).all()
