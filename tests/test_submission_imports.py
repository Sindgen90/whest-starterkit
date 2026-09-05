from __future__ import annotations

import ast
import re
from pathlib import Path

BANNED_IMPORT_PATTERNS = [
    re.compile(r"^\s*import\s+(?:numpy|scipy|torch|jax|sklearn|pandas)\b", re.MULTILINE),
    re.compile(r"^\s*from\s+(?:numpy|scipy|torch|jax|sklearn|pandas)\b", re.MULTILINE),
]


def test_submission_estimators_do_not_import_unavailable_packages() -> None:
    estimator_paths = sorted(Path("submissions").glob("*/estimator.py"))
    estimator_paths.extend(sorted(Path("submission_variants").glob("*/estimator.py")))
    assert estimator_paths, "No submission estimator.py files found."

    violations: list[str] = []
    for path in estimator_paths:
        source = path.read_text(encoding="utf-8")
        if any(pattern.search(source) for pattern in BANNED_IMPORT_PATTERNS):
            violations.append(str(path))

    assert not violations, (
        "Submission estimators must import only packages available in the "
        f"grader smoke-test environment. Violations: {violations}"
    )


def test_structural_submission_avoids_remote_array_item_assignment() -> None:
    paths = (
        Path("submissions/structural_tail_sobol_32768/estimator.py"),
        Path("submissions/structural_tail6_sobol_32768/estimator.py"),
        Path("submissions/structural_tail8_sobol_32768/estimator.py"),
        Path("submissions/structural_tail10_sobol_32768/estimator.py"),
        Path("submissions/structural_tail12_sobol_32768/estimator.py"),
        Path("submissions/structural_tail14_sobol_32768/estimator.py"),
        Path("submissions/structural_tail16_sobol_32768/estimator.py"),
        Path(
            "submissions/structural_tail16_early_dead_sobol_32768/estimator.py"
        ),
        Path(
            "submissions/structural_tail16_dead_m050_sobol_32768/estimator.py"
        ),
        Path(
            "submissions/structural_tail16_tail8_linear_all061_sobol_32768/"
            "estimator.py"
        ),
        Path(
            "submissions/structural_tail16_hope_merge_static_sobol32768/"
            "estimator.py"
        ),
    )
    guarded_functions = {"_structural_prediction", "predict"}
    violations: list[str] = []

    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for function in (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in guarded_functions
        ):
            for node in ast.walk(function):
                targets = node.targets if isinstance(node, ast.Assign) else []
                if isinstance(node, ast.AugAssign):
                    targets = [node.target]
                if any(isinstance(target, ast.Subscript) for target in targets):
                    violations.append(f"{path}:{function.name}:{node.lineno}")

    assert not violations, (
        "Remote flopscope arrays are immutable; build indexed results "
        f"functionally. Violations: {violations}"
    )
