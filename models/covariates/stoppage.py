"""Stoppage estimates from the first 100 EPL matches of 2024/25."""

import numpy as np

# Before the 2024-11-09 evaluation cutoff; reproduced by eval/refit.py.
H1_OBSERVED = np.repeat(np.arange(12, dtype=np.int8), [4, 20, 19, 25, 12, 6, 5, 3, 4, 1, 0, 1])
H2_OBSERVED = np.repeat(np.arange(3, 14, dtype=np.int8), [3, 14, 20, 20, 17, 12, 6, 4, 1, 0, 3])
MEAN_H1_STOPPAGE = float(H1_OBSERVED.mean())
MEAN_H2_STOPPAGE = float(H2_OBSERVED.mean())


def compute_remaining(minute: float, period: int) -> float:
    """Remaining match time under mean stoppage; `period` (1 or 2) is required."""
    if period not in (1, 2):
        raise ValueError(f'period must be 1 or 2, got {period!r}')
    in_h1 = period == 1
    return max(
        0.1,
        (90 + MEAN_H1_STOPPAGE + MEAN_H2_STOPPAGE - minute)
        if in_h1
        else (90 + MEAN_H2_STOPPAGE - minute),
    )
