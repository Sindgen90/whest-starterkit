"""Correctness tests for paired Box--Muller RQMC."""
from __future__ import annotations

from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
import pytest

from submission_variants.rqmc_box_muller import estimator


def test_box_muller_pair_has_expected_radius_and_angle() -> None:
    uniforms = fnp.array(
        [[0.2, 0.3, 0.4, 0.5], [0.7, 0.9, 0.8, 0.1]],
        dtype=fnp.float64,
    )
    transformed = np.asarray(estimator.box_muller_transform(uniforms))
    radial_uniforms = np.asarray(uniforms)[:, 0::2]
    radii_squared = transformed[:, 0::2] ** 2 + transformed[:, 1::2] ** 2
    np.testing.assert_allclose(
        radii_squared,
        -2.0 * np.log(radial_uniforms),
        rtol=1e-13,
        atol=1e-13,
    )


def test_box_muller_lattice_is_reproducible_finite_and_seeded() -> None:
    first = np.asarray(estimator.normal_lattice_samples(1_024, 16, seed=91))
    repeated = np.asarray(estimator.normal_lattice_samples(1_024, 16, seed=91))
    different = np.asarray(estimator.normal_lattice_samples(1_024, 16, seed=92))
    np.testing.assert_array_equal(first, repeated)
    assert first.shape == (1_024, 16)
    assert np.isfinite(first).all()
    assert not np.array_equal(first, different)


def test_box_muller_independent_uniform_reference_has_gaussian_moments() -> None:
    rng = np.random.default_rng(101)
    uniforms = fnp.array(rng.random((100_000, 8)), dtype=fnp.float64)
    samples = np.asarray(estimator.box_muller_transform(uniforms))
    np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=0.012)
    np.testing.assert_allclose(samples.var(axis=0), 1.0, atol=0.025)


def test_box_muller_shape_validation() -> None:
    with pytest.raises(ValueError, match="shape"):
        estimator.box_muller_transform(fnp.ones(8))
    with pytest.raises(ValueError, match="even"):
        estimator.box_muller_transform(fnp.ones((4, 3)))


def test_box_muller_generation_is_flop_counted() -> None:
    with flops.BudgetContext(flop_budget=1_000_000_000, quiet=True) as context:
        estimator.normal_lattice_samples(128, 16, seed=4)
    assert context.flops_used > 0


def test_submission_variant_has_no_unsupported_imports() -> None:
    source = Path(estimator.__file__).read_text(encoding="utf-8")
    assert "import numpy" not in source
    assert "import scipy" not in source
    assert "import torch" not in source
    assert "import jax" not in source
