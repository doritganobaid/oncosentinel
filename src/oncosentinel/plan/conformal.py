from __future__ import annotations

import math

import attrs
import numpy as np
from numpy.typing import NDArray


@attrs.define
class SplitConformal:
    alpha: float
    quantile: float = attrs.field(default=math.nan, init=False)

    def calibrate(self, residuals: NDArray[np.float64]) -> None:
        scores = np.abs(np.asarray(residuals, dtype=np.float64))
        n = scores.shape[0]
        if n == 0:
            raise ValueError("calibration set must be non-empty")
        level = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.quantile = float(np.quantile(scores, level, method="higher"))

    @property
    def is_calibrated(self) -> bool:
        return not math.isnan(self.quantile)

    def interval(self, point_prediction: float) -> tuple[float, float]:
        if not self.is_calibrated:
            raise RuntimeError("calibrate before requesting intervals")
        return (point_prediction - self.quantile, point_prediction + self.quantile)

    def covers(self, observation: float, point_prediction: float) -> bool:
        lower, upper = self.interval(point_prediction)
        return lower <= observation <= upper

    def empirical_coverage(
        self, observations: NDArray[np.float64], point_predictions: NDArray[np.float64]
    ) -> float:
        lower = point_predictions - self.quantile
        upper = point_predictions + self.quantile
        inside = (observations >= lower) & (observations <= upper)
        return float(np.mean(inside))
