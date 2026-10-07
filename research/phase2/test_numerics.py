"""Independent numerical invariants for the reconstructed estimator."""
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

from estimator import Estimator, covariance_path, whitened_path


def test_first_layer_matches_exact_gaussian_integral_with_asymmetric_weights():
    w = np.array([[1., -2., 0.], [3., 4., 0.], [-2., 1., 0.]], dtype=np.float32)
    mlp = MLP(3, 1, [fnp.asarray(w)], seed=12)
    result, _ = covariance_path(mlp)
    expected = np.linalg.norm(w, axis=0) / np.sqrt(2 * np.pi)
    np.testing.assert_allclose(result[0], expected, atol=1e-6, rtol=1e-6)


def test_whitened_antipodal_sample_preserves_linear_response():
    # For paired x and -x, ReLU(x)-ReLU(-x) == x. Opposite output weight
    # columns therefore have the same mean, independent of Monte Carlo error.
    w = fnp.asarray([[1., -1.], [2., -2.]], dtype=fnp.float32)
    mlp = MLP(2, 1, [w], seed=18)
    result = whitened_path(mlp, 4096, 18)
    np.testing.assert_allclose(result[0, 0], result[0, 1], atol=1e-6)
    assert abs(float(result[0, 0]) - np.sqrt(5 / (2 * np.pi))) < 0.03


def test_small_budget_avoids_rank_deficient_whitening():
    mlp = MLP(4, 2, [fnp.eye(4, dtype=fnp.float32)] * 2, seed=1)
    estimator = Estimator()
    with flops.BudgetContext(flop_budget=100_000, quiet=True):
        result = estimator.predict(mlp, 1000)
    expected, _ = covariance_path(mlp)
    np.testing.assert_allclose(result, expected)
