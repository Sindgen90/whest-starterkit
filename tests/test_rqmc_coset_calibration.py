"""Tests for disjoint fixed-coset calibration."""
from __future__ import annotations

from research.scripts.run_rqmc_coset_calibration_check import evaluate_frozen_choices


def test_frozen_choices_are_applied_without_reselection() -> None:
    calibration = {
        str(size): {
            "fixed_best": {
                "combo": list(range(size)),
                "mse": {"mean": float(size)},
            }
        }
        for size in (1, 2, 3)
    }
    selection_rows = [
        {
            "combination_mse": {
                "1": {"0": 1.0, "1": 0.0},
                "2": {"0+1": 2.0, "4+5": 0.0},
                "3": {"0+1+2": 3.0, "3+4+5": 0.0},
            }
        }
    ]
    result = evaluate_frozen_choices(selection_rows, calibration)
    assert result["1"]["combo"] == [0]
    assert result["1"]["selection_mse"]["mean"] == 1.0
    assert result["2"]["combo"] == [0, 1]
    assert result["2"]["selection_mse"]["mean"] == 2.0
    assert result["3"]["combo"] == [0, 1, 2]
    assert result["3"]["selection_mse"]["mean"] == 3.0
