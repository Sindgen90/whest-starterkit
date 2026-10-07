import numpy as np
from diagnose_distribution import gaussian_relu,closure,risk_summary


def test_gaussian_relu_zero_mean_identity():
    variance = np.array([.1,1.,4.])
    np.testing.assert_allclose(gaussian_relu(np.zeros(3),variance),np.sqrt(variance/(2*np.pi)),rtol=1e-12)


def test_restart_after_first_layer_matches_uninterrupted_identity_weights():
    w = np.array([np.eye(3,dtype=np.float32)]*3)
    mu = np.ones(3,dtype=np.float32)/np.sqrt(2*np.pi)
    cov = np.eye(3,dtype=np.float32)*(.5-1/(2*np.pi))
    np.testing.assert_allclose(closure(w,mu,cov,start=1)["mean"],closure(w)["mean"][1:],atol=5e-7)


def test_cross_stream_risk_is_average_pairwise_error_product():
    p = np.array([[1.,2.],[3.,-1.],[4.,2.],[-2.,5.]])
    target = np.array([.2,.3])
    e = p-target
    expected = np.mean([np.mean(e[i]*e[j]) for i in range(4) for j in range(i+1,4)])
    np.testing.assert_allclose(risk_summary(p,target)["cross_stream_risk"],expected)


def test_mean_feedback_changes_next_linear_mean():
    weights = np.array([np.eye(2,dtype=np.float32)]*2)
    base = closure(weights)
    adjusted = closure(weights,mean_correction=lambda m,v,l: .1 if l==0 else 0.)
    np.testing.assert_allclose(adjusted["premean"][1],base["mean"][0]+.1,atol=1e-7)
    np.testing.assert_array_equal(adjusted["prevar"][1],base["prevar"][1])
