from __future__ import annotations

import numpy as np

from oncosentinel.plan.conformal import SplitConformal


def test_split_conformal_attains_marginal_coverage() -> None:
    rng = np.random.default_rng(0)
    residuals_calib = rng.normal(0.0, 1.0, size=4000)
    conformal = SplitConformal(alpha=0.1)
    conformal.calibrate(residuals_calib)

    point_predictions = np.zeros(4000, dtype=np.float64)
    observations = rng.normal(0.0, 1.0, size=4000)
    coverage = conformal.empirical_coverage(observations, point_predictions)
    assert coverage >= 0.86


def test_uncalibrated_interval_raises() -> None:
    conformal = SplitConformal(alpha=0.1)
    assert not conformal.is_calibrated
    try:
        conformal.interval(0.0)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError before calibration")
