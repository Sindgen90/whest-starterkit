"""Independent geometry, transform, and regression checks."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import flopscope.numpy as fnp

ROOT = Path(__file__).parent
sys.path.insert(0,str(ROOT / "kerdock_baseline"))
spec = importlib.util.spec_from_file_location("kerdock_impl",ROOT/"kerdock_baseline"/"estimator.py")
impl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(impl)
sys.path.pop(0)


def hadamard(n):
    h = np.ones((1,1))
    while len(h)<n:
        h = np.block([[h,h],[h,-h]])
    return h


def test_fwht_against_explicit_nonsymmetric_matrix():
    x = np.random.default_rng(4).normal(size=(3,16,7)).astype(np.float32)
    actual = impl.fwht(fnp.asarray(x))
    np.testing.assert_allclose(actual,hadamard(16)@x,rtol=1e-5,atol=2e-6)


def test_assets_are_mutually_unbiased_and_rotation_orthogonal():
    with np.load(ROOT/"kerdock_baseline"/"design.npz") as assets:
        for n in (4,16,64,256,1024):
            c = assets[f"chirps_{n}"]
            assert np.all(np.isin(c,[-1,1]))
            # Walsh spectrum of every pair with the first frame must be flat.
            products = c[1:]*c[0]
            transformed = np.asarray(impl.fwht(fnp.asarray(products[:,:,None])))[:,:,0]
            np.testing.assert_allclose(np.abs(transformed),np.sqrt(n),atol=0)
            # Independent pairs catch more than first-frame comparisons alone.
            rng = np.random.default_rng(9)
            for _ in range(8):
                i,j = rng.choice(len(c),2,replace=False)
                p = (c[i]*c[j])[None,:,None]
                np.testing.assert_allclose(np.abs(impl.fwht(fnp.asarray(p))),np.sqrt(n),atol=0)
            q = assets[f"rotation_{n}"].astype(np.float64)
            np.testing.assert_allclose(q.T@q,np.eye(n),atol=1e-7)


def test_dual_ridge_matches_explicit_response_with_correct_sign():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(70,9)).astype(np.float32)
    y = rng.normal(size=(70,5)).astype(np.float32)
    delta = rng.normal(size=9).astype(np.float32)
    xc,yc = x-x.mean(0),y-y.mean(0)
    gram = xc.T@xc
    matrix = np.linalg.solve(gram+0.25*np.trace(gram)/9*np.eye(9),xc.T@yc)
    got = impl.ridge_correction(fnp.asarray(x),fnp.asarray(y),fnp.asarray(delta),0.25)
    np.testing.assert_allclose(got,delta@matrix,rtol=2e-5,atol=1e-6)


def test_zero_discrepancy_produces_zero_correction():
    x = fnp.ones((12,4),dtype=fnp.float32)
    got = impl.ridge_correction(x,x,fnp.zeros(4,dtype=fnp.float32),0.25)
    np.testing.assert_array_equal(got,np.zeros(4))
