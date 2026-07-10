from __future__ import annotations

import attrs
import numpy as np
from numpy.typing import NDArray

from oncosentinel.plan.hypervolume import (
    Point,
    dominated_mask,
    hypervolume,
    hypervolume_contribution,
)


@attrs.frozen
class ParetoSelection:
    argmax_index: int
    hv_total: float
    contributions: NDArray[np.float64]
    front_indices: NDArray[np.int_]

    @property
    def frontier_size(self) -> int:
        return int(self.front_indices.shape[0])

    @property
    def hv_argmax(self) -> float:
        return float(self.contributions[self.argmax_index])


def recommend(points: NDArray[np.float64] | list[Point], reference: Point) -> ParetoSelection:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
        raise ValueError("points must have shape (n, 2) with n >= 1")
    front_indices = np.flatnonzero(~dominated_mask(arr))
    total = hypervolume(arr, reference)
    contributions = hypervolume_contribution(arr, reference)
    masked = np.full(arr.shape[0], -np.inf, dtype=np.float64)
    masked[front_indices] = contributions[front_indices]
    argmax_index = int(np.argmax(masked))
    return ParetoSelection(
        argmax_index=argmax_index,
        hv_total=total,
        contributions=contributions,
        front_indices=front_indices,
    )
