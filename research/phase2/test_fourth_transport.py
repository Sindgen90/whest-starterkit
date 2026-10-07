"""Independent dense-tensor verification of the retained paired projection."""
from itertools import permutations
import numpy as np
import pytest
import flopscope.numpy as fnp
from whestbench import MLP
from fourth_transport import transport


def dense_paired_tensor(q):
    n = len(q)
    tensor = np.zeros((n,n,n,n),dtype=np.float64)
    for i in range(n):
        tensor[i,i,i,i] = q[i,i]
        for j in range(i+1,n):
            for indices in set(permutations((i,i,j,j))):
                tensor[indices] = q[i,j]
    return tensor


@pytest.mark.parametrize('seed',range(4))
def test_complete_plane_matches_independent_full_tensor_transformation(seed):
    rng = np.random.default_rng(800+seed)
    q = rng.normal(size=(4,4))
    q = ((q+q.T)/2).astype(np.float32)
    w = rng.normal(size=(4,3)).astype(np.float32)
    original = dense_paired_tensor(q)
    transformed = np.einsum('ijkl,ia,jb,kc,ld->abcd',original,w,w,w,w,optimize=True)
    expected = np.array([[transformed[a,a,b,b] for b in range(3)] for a in range(3)])
    diagonal,actual = transport(fnp.asarray(q),fnp.asarray(w),'complete')
    np.testing.assert_allclose(np.asarray(actual),expected,rtol=3e-6,atol=2e-5)
    np.testing.assert_allclose(np.asarray(diagonal),np.diag(expected),rtol=3e-6,atol=2e-5)


def test_pairing_term_changes_only_off_diagonal_relative_to_current_transport():
    q = fnp.asarray(np.array([[.7,.2,-.1],[.2,-.3,.4],[-.1,.4,.5]],dtype=np.float32))
    w = fnp.asarray(np.array([[.4,-.2],[.3,.6],[-.5,.8]],dtype=np.float32))
    old_d,old_q = transport(q,w,'gram')
    new_d,new_q = transport(q,w,'complete')
    np.testing.assert_array_equal(np.asarray(new_d),np.asarray(old_d))
    np.testing.assert_array_equal(np.diag(np.asarray(new_q)),np.diag(np.asarray(old_q)))
    assert abs(float(new_q[0,1]-old_q[0,1])) > .01


def test_identity_transport_and_diagonal_source_are_exact():
    q = np.array([[1.,.3],[.3,-.4]],dtype=np.float32)
    _,plane = transport(fnp.asarray(q),fnp.eye(2,dtype=fnp.float32),'complete')
    np.testing.assert_allclose(np.asarray(plane),q,atol=1e-7)
    w = fnp.asarray(np.array([[.5,-.3],[.2,.8]],dtype=np.float32))
    for mode in ('gram','complete'):
        _,plane = transport(fnp.asarray(np.diag(np.diag(q))),w,mode)
        expected = np.asarray(w*w).T@np.diag(np.diag(q))@np.asarray(w*w)
        np.testing.assert_allclose(np.asarray(plane),expected,rtol=2e-6,atol=1e-7)


def test_gram_research_replica_matches_frozen_runtime_state_and_means():
    from joint_full_runtime.estimator import Estimator as Frozen
    from fourth_transport_candidate.estimator import Estimator as Research
    rng = np.random.default_rng(816)
    w = [fnp.asarray(rng.normal(size=(12,12)).astype(np.float32)*np.sqrt(2/12)) for _ in range(16)]
    mlp = MLP(12,16,w,seed=0)
    a,b = Frozen(),Research()
    a.collect_diagnostics = b.collect_diagnostics = True
    old,new = np.asarray(a.predict(mlp,2**41)),np.asarray(b.predict(mlp,2**41))
    np.testing.assert_array_equal(new,old)
    for x,y in zip(a.trace,b.trace):
        for left,right in zip(x,y):
            np.testing.assert_array_equal(np.asarray(left),np.asarray(right))


def test_paired_kstat_translation_and_diagonal_formula():
    from diagnose_fourth_transport import paired_kstat,student_critical_95
    rng = np.random.default_rng(817)
    y = rng.normal(size=(300,4))

    def statistic(values):
        square = values*values
        return paired_kstat((values.mean(0),values.T@values/len(values),
                             square.T@values/len(values),square.T@square/len(values)),len(values))

    actual = statistic(y)
    np.testing.assert_allclose(statistic(y+np.array([2.,-.7,1.,.2])),actual,atol=2e-13,rtol=2e-12)
    centered = y-y.mean(0)
    v = (centered*centered).mean(0)
    m4 = (centered**4).mean(0)
    n = len(y)
    expected = n*n*((n+1)*m4-3*(n-1)*v*v)/((n-1)*(n-2)*(n-3))
    np.testing.assert_allclose(np.diag(actual),expected,atol=1e-14)
    assert abs(student_critical_95(3)-3.182446305284263)<1e-10
