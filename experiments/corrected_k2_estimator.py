"""Frozen global correction of the bundled covariance-propagation K2 estimator."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

COEFFICIENT = 0.9918858550287327
_BASELINE_PATH = Path(__file__).resolve().parents[1] / "examples" / "03_covariance_propagation.py"


def _baseline_class() -> type[BaseEstimator]:
    spec = importlib.util.spec_from_file_location("covariance_k2", _BASELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load covariance K2 from {_BASELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Estimator


class Estimator(BaseEstimator):
    def __init__(self) -> None:
        self._base = _baseline_class()()

    def setup(self, ctx: SetupContext) -> None:
        self._base.setup(ctx)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        return fnp.multiply(self._base.predict(mlp, budget), COEFFICIENT)
