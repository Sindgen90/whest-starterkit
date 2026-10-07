"""Equivalence checks for the standalone full joint runtime derivative."""
import numpy as np
import pytest
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP
from joint_full_candidate.estimator import Estimator as Original, responses
from joint_full_runtime.estimator import Estimator as Runtime, final_mean, nong_terms


def test_final_mean_matches_four_raw_moment_correction_including_clipping():
    mean = fnp.asarray(np.array([-3., -.5, 0., .7, 4.], dtype=np.float32))
    variance = fnp.asarray(np.array([.05, .6, 1., 2., .08], dtype=np.float32))
    third = fnp.asarray(np.array([30., -.1, 2., -40., .1], dtype=np.float32))
    fourth = fnp.asarray(np.array([-60., .4, -20., 50., -.1], dtype=np.float32))
    moments, f, sigma, _, _ = responses(mean, variance)
    expected = moments[1] + fnp.clip(f[1][3]*(third/6) + f[1][4]*(fourth/24),
                                   -.25*sigma, .25*sigma)
    np.testing.assert_array_equal(np.asarray(final_mean(mean, variance, third, fourth)),
                                  np.asarray(expected))


def test_response_coefficient_factoring_matches_original_mixed_contractions():
    from joint_full_candidate.estimator import nong_terms as reference
    rng = np.random.default_rng(142)
    arrays = [fnp.asarray(rng.normal(size=(7, 7)).astype(np.float32)) for _ in range(3)]
    d3 = fnp.asarray(rng.normal(size=7).astype(np.float32))
    _, f, _, _, _ = responses(fnp.asarray(rng.normal(size=7).astype(np.float32)),
                              fnp.asarray(rng.uniform(.1, 2., size=7).astype(np.float32)))
    for higher in (False, True):
        actual = nong_terms(*arrays, d3, f, need_higher=higher)
        expected = reference(*arrays, d3, f, need_higher=higher)
        for a, b in zip(actual, expected):
            if b is None:
                assert a is None
            else:
                np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=3e-6, atol=2e-6)


@pytest.mark.parametrize('depth,history,fourth,rich,batch', [
    (1, 1, True, 16, True), (2, 1, True, 16, True),
    (16, 1, True, 16, True), (6, 2, True, 16, True),
    (6, 1, False, 16, True), (17, 1, True, 12, True),
    (5, 1, True, 16, False), (3, 1, True, 0, True),
])
def test_all_layer_predictions_match_frozen_source(depth, history, fourth, rich, batch):
    rng = np.random.default_rng(903)
    weights = [fnp.asarray(rng.normal(size=(12, 12)).astype(np.float32)*np.sqrt(2/12))
               for _ in range(depth)]
    mlp = MLP(12, depth, weights, seed=0)
    old, new = Original(), Runtime()
    for estimator in (old, new):
        estimator.history = history
        estimator.use_fourth = fourth
        estimator.rich_layers = rich
        estimator.batch_quadrature = batch
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        a = np.asarray(old.predict(mlp, 2**41)).copy()
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        b = np.asarray(new.predict(mlp, 2**41)).copy()
    assert b.shape == (depth, 12) and np.isfinite(b).all()
    np.testing.assert_allclose(b, a, rtol=5e-6, atol=2e-6)
    if depth == 1:
        np.testing.assert_allclose(b[0], np.linalg.norm(np.asarray(weights[0]), axis=0)
                                   / np.sqrt(2*np.pi), rtol=1e-6)


def test_terminal_specialization_matches_diagnostic_full_matrix_path():
    rng = np.random.default_rng(905)
    weights = [fnp.asarray(rng.normal(size=(16, 16)).astype(np.float32)*np.sqrt(2/16))
               for _ in range(8)]
    mlp = MLP(16, 8, weights, seed=0)
    fast, diagnostic = Runtime(), Runtime()
    diagnostic.collect_diagnostics = True
    a = np.asarray(fast.predict(mlp, 2**41))
    b = np.asarray(diagnostic.predict(mlp, 2**41))
    np.testing.assert_allclose(a, b, rtol=5e-6, atol=2e-6)
    assert len(diagnostic.trace) == mlp.depth


def test_buffer_reuse_preserves_inputs_and_repeat_predictions():
    rng = np.random.default_rng(906)
    raw = [rng.normal(size=(12,12)).astype(np.float32)*np.sqrt(2/12) for _ in range(5)]
    mlp = MLP(12,5,[fnp.asarray(w.copy()) for w in raw],seed=0)
    estimator = Runtime()
    first = np.asarray(estimator.predict(mlp,2**41)).copy()
    second = np.asarray(estimator.predict(mlp,2**41)).copy()
    np.testing.assert_array_equal(second,first)
    for original,weight in zip(raw,mlp.weights):
        np.testing.assert_array_equal(np.asarray(weight),original)
