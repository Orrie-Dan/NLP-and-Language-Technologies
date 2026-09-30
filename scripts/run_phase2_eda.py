"""Run Phase 2 loading checks, shared split, and audio EDA.

Does not train models. Run from any working directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

from src.data.eda import run_phase2


if __name__ == "__main__":
    run_phase2(ROOT)
