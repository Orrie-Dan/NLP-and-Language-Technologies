"""Run the Phase 3 preprocessing investigation.

Does not train models and does not rewrite the shared split.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

from src.data.phase3 import run_phase3


if __name__ == "__main__":
    run_phase3(ROOT)
