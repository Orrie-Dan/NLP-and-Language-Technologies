"""Reproducibility helpers.

The shared split uses ``SHARED_SPLIT_SEED``. Neural-framework seeding is not
set here because no neural library has been chosen.
"""

from __future__ import annotations

import random
import sys

import numpy as np

SHARED_SPLIT_SEED = 42
SPLIT_FRACTIONS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}


def set_seed(seed: int = SHARED_SPLIT_SEED) -> None:
    """Seed Python and NumPy. Model libraries must be seeded separately later."""
    random.seed(seed)
    np.random.seed(seed)


def package_versions(packages: tuple[str, ...] = (
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "seaborn",
)) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            module = __import__(name)
        except ImportError:
            versions[name] = None
            continue
        versions[name] = getattr(module, "__version__", None)
    return versions


def environment_record(seed: int = SHARED_SPLIT_SEED) -> dict:
    return {
        "python_version": sys.version,
        "random_seed": seed,
        "split_fractions": dict(SPLIT_FRACTIONS),
        "packages": package_versions(),
    }
