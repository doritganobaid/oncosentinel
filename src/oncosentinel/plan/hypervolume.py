from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Point = tuple[float, float]


def _as_array(points: NDArray[np.float64] | list[Point]) -> NDArray[np.float64]:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    return arr


def dominated_mask(points: NDArray[np.float64]) -> NDArray[np.bool_]:
    n = points.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ge = points[j] >= points[i]
            gt = points[j] > points[i]
            if bool(ge.all()) and bool(gt.any()):
                dominated[i] = True
                break
    return dominated


def non_dominated_front(points: NDArray[np.float64] | list[Point]) -> NDArray[np.float64]:
    arr = _as_array(points)
    if arr.shape[0] == 0:
        return arr
    keep = ~dominated_mask(arr)
    return arr[keep]


def hypervolume(points: NDArray[np.float64] | list[Point], reference: Point) -> float:
    arr = non_dominated_front(points)
    if arr.shape[0] == 0:
        return 0.0
    ref_x, ref_y = reference
    order = np.argsort(-arr[:, 0], kind="mergesort")
    ordered = arr[order]
    area = 0.0
    prev_y = ref_y
    for x, y in ordered:
        if x <= ref_x or y <= prev_y:
            continue
        area += (x - ref_x) * (y - prev_y)
        prev_y = y
    return float(area)


def hypervolume_contribution(
    points: NDArray[np.float64] | list[Point], reference: Point
) -> NDArray[np.float64]:
    arr = _as_array(points)
    n = arr.shape[0]
    total = hypervolume(arr, reference)
    contributions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        without = np.delete(arr, i, axis=0)
        contributions[i] = total - hypervolume(without, reference)
    return contributions
