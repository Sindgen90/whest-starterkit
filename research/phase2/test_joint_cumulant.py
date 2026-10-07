import numpy as np
import flopscope.numpy as fnp
from whestbench import MLP
from joint_cumulant import gaussian_pairs,Estimator,quadrature_tables
from puffi_reference import responses,third_slices,new_third_source


def test_three_gaussian_covariances_against_independent_double_integration():
    nodes,weights = np.polynomial.legendre.leggauss(150)
    means = np.array([-.4,.6],dtype=np.float32)
    sigmas = np.array([1.2,.8],dtype=np.float32)
    for rho in (-.4,0.,.4):
        covariance = np.outer(sigmas,sigmas)*np.array([[1.,rho],[rho,1.]])
        moment,f,sigma,a,phi = responses(fnp.asarray(means),fnp.asarray(sigmas**2))
        c11,c21,c22 = [np.asarray(x) for x in gaussian_pairs(fnp.asarray(covariance,dtype=fnp.float32),moment,sigma,a,phi)]
        upper = np.maximum(means+12*sigmas,12*sigmas)
        x = (nodes+1)*upper[0]/2
        y = (nodes+1)*upper[1]/2
        xx = (x[:,None]-means[0])/sigmas[0]
        yy = (y[None,:]-means[1])/sigmas[1]
        density = np.exp(-(xx*xx-2*rho*xx*yy+yy*yy)/(2*(1-rho*rho)))/(2*np.pi*np.prod(sigmas)*np.sqrt(1-rho*rho))
        mass = density*np.outer(weights,weights)*np.prod(upper)/4
        for p,q,actual in ((1,1,c11[0,1]),(2,1,c21[0,1]),(2,2,c22[0,1])):
            expected = np.sum(mass*x[:,None]**p*y[None,:]**q)-np.asarray(moment[p])[0]*np.asarray(moment[q])[1]
            np.testing.assert_allclose(actual,expected,atol=5e-7,rtol=2e-5)


def test_third_projection_preserves_diagonal_and_repeated_slices():
    p = np.array([[.2,.03,-.04],[.01,.4,.08],[-.02,.05,.1]],dtype=np.float32)
    a,b = new_third_source(fnp.asarray(np.diag(p)),fnp.asarray(p))
    d,repeated = third_slices(a,b)
    np.testing.assert_allclose(np.asarray(repeated),p,atol=1e-7)
    np.testing.assert_allclose(np.asarray(d),np.diag(p),atol=1e-7)


def test_first_layer_has_exact_gaussian_mean():
    weights = np.array([[1.,.2],[-.3,.8]],dtype=np.float32)
    pred = np.asarray(Estimator().predict(MLP(2,1,[fnp.asarray(weights)],seed=0),2**41))
    np.testing.assert_allclose(pred[0],np.linalg.norm(weights,axis=0)/np.sqrt(2*np.pi),rtol=1e-6)


def test_batched_quadrature_matches_scalar_quadrature():
    mean = fnp.asarray(np.array([-.3,.5],dtype=np.float32))
    cov = fnp.asarray(np.array([[1.,-.2],[-.2,.6]],dtype=np.float32))
    moments,f,sigma,a,phi = responses(mean,fnp.diag(cov))
    for order in (2,4,8):
        scalar = gaussian_pairs(cov,moments,sigma,a,phi,order=order)
        batched = gaussian_pairs(cov,moments,sigma,a,phi,order=order,tables=quadrature_tables(order))
        for x,y in zip(scalar,batched):
            np.testing.assert_allclose(np.asarray(x),np.asarray(y),atol=5e-7,rtol=2e-6)


def test_optimized_joint_state_matches_frozen_v1_on_small_network():
    from joint_cumulant_v1 import Estimator as Original
    rng = np.random.default_rng(301)
    w = [fnp.asarray(rng.normal(size=(12,12)).astype(np.float32)*np.sqrt(2/12)) for _ in range(5)]
    mlp = MLP(12,5,w,seed=0)
    for history in (1,2):
        old,new = Original(),Estimator()
        old.history = new.history = history
        a = np.asarray(old.predict(mlp,2**41))
        b = np.asarray(new.predict(mlp,2**41))
        np.testing.assert_allclose(b,a,rtol=2e-5,atol=3e-6)


def test_standalone_calibration_changes_outputs_but_not_propagated_state():
    from joint_candidate.estimator import Estimator as Frozen,CALIBRATION
    rng = np.random.default_rng(302)
    w = [fnp.asarray(rng.normal(size=(12,12)).astype(np.float32)*np.sqrt(2/12)) for _ in range(16)]
    mlp = MLP(12,16,w,seed=0)
    raw,frozen = Estimator(),Frozen()
    raw.rich_layers,raw.quadrature_order = 12,4
    raw.collect_diagnostics = frozen.collect_diagnostics = True
    a = np.asarray(raw.predict(mlp,2**41))
    b = np.asarray(frozen.predict(mlp,2**41))
    trace = np.array([[np.asarray(x) for x in layer] for layer in raw.trace])
    frozen_trace = np.array([[np.asarray(x) for x in layer] for layer in frozen.trace])
    np.testing.assert_allclose(frozen_trace,trace,atol=1e-7,rtol=1e-6)
    design = np.stack((a,trace[:,1],trace[:,0]/np.sqrt(trace[:,1]),trace[:,5],np.ones_like(a)),axis=-1)
    expected = a+np.einsum('lnk,lk->ln',design,np.asarray(CALIBRATION,dtype=np.float32))
    np.testing.assert_allclose(b,expected,rtol=1e-6,atol=1e-7)
