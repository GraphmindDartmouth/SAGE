"""Keep signed knowledge local for the paper's Distance-Weighted baseline."""

from __future__ import annotations

import numpy as np


# Func: Use the identity operator instead of graph diffusion.
# Input: Current signed knowledge vector.
# Output: An unchanged copy of that vector.
def propagate(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).copy()
