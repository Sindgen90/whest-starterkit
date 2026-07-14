from __future__ import annotations

import re
from pathlib import Path


BANNED_IMPORT_PATTERNS = [
    re.compile(r"^\s*import\s+(?:numpy|scipy|torch|jax)\b", re.MULTILINE),
    re.compile(r"^\s*from\s+(?:numpy|scipy|torch|jax)\b", re.MULTILINE),
]


def test_submission_estimators_do_not_import_unavailable_packages() -> None:
    estimator_paths = sorted(Path("submissions").glob("*/estimator.py"))
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
