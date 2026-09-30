"""Validate the Phase 4 feature pipeline on a small training subset.

Does not train a classifier and does not rewrite the shared split.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.validate import run_phase4_validation


if __name__ == "__main__":
    report = run_phase4_validation(ROOT)
    table = report["table"]
    print(table.loc[:, ["check", "feature_dimension", "all_finite", "reproducible", "passed"]].to_string(index=False))
    print("all checks passed", report["all_passed"])
