"""Tests for non-learned RQMC partition attribution."""
from __future__ import annotations

import numpy as np
import pytest

from research.scripts.run_rqmc_partition_family_diagnostic import (
    build_partition_collection,
    partition_statistics,
)


def test_partition_collection_contains_fixed_and_random_families() -> None:
    names, partitions, members = build_partition_collection(3, random_seed=71)
    assert names[:5] == [
        "true_residue_cosets",
        "contiguous_blocks",
        "antithetic_pair_blocks",
        "affine_index_blocks",
        "bit_reversed_blocks",
    ]
    assert partitions.shape == (8, 6, 4096)
    assert members["random_balanced_ensemble"] == [5, 6, 7]


def test_partition_statistics_matches_six_way_mse_identity() -> None:
    rng = np.random.default_rng(19)
    errors = rng.normal(size=(7, 6, 9))
    summary = partition_statistics(errors)
    assert summary["mse_identity_prediction"] == pytest.approx(
        summary["full_six_group_mse"],
        rel=1e-14,
        abs=1e-14,
    )
    assert len(summary["group_mse"]) == 6
    assert len(summary["leave_one_out_mse"]) == 6


def test_removing_exact_group_identifies_dominant_error() -> None:
    errors = np.zeros((4, 6, 3), dtype=np.float64)
    errors[:, 2, :] = 6.0
    summary = partition_statistics(errors)
    assert summary["largest_removal_penalty_group"] != 2
    assert summary["removal_delta"][2] < 0
    assert summary["best_group"] == 0
