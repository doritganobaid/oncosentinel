from __future__ import annotations

import math

from oncosentinel.knowledge.posteriors import JointPosterior, LvefPosterior, ResponsePosterior
from oncosentinel.monitor.triggers import beta_kl, gaussian_kl, joint_kl, triggered


def _joint(alpha: float, beta: float, mu: float, sigma2: float) -> JointPosterior:
    return JointPosterior(
        response=ResponsePosterior(alpha=alpha, beta=beta),
        lvef=LvefPosterior(mu=mu, sigma2=sigma2),
    )


def test_kl_is_zero_for_identical_distributions() -> None:
    response = ResponsePosterior(alpha=3.0, beta=4.0)
    lvef = LvefPosterior(mu=0.2, sigma2=0.03)
    assert math.isclose(beta_kl(response, response), 0.0, abs_tol=1e-9)
    assert math.isclose(gaussian_kl(lvef, lvef), 0.0, abs_tol=1e-9)


def test_kl_non_negative_and_positive_for_distinct() -> None:
    current = _joint(6.0, 3.0, 0.30, 0.02)
    previous = _joint(3.0, 4.0, 0.15, 0.02)
    assert joint_kl(current, previous) > 0.0
    assert beta_kl(current.response, previous.response) >= 0.0


def test_joint_kl_factorises_under_independence() -> None:
    current = _joint(6.0, 3.0, 0.30, 0.025)
    previous = _joint(3.0, 4.0, 0.15, 0.02)
    expected = beta_kl(current.response, previous.response) + gaussian_kl(
        current.lvef, previous.lvef
    )
    assert math.isclose(joint_kl(current, previous), expected, rel_tol=1e-12)


def test_trigger_fires_above_threshold() -> None:
    assert triggered(0.12, tau=0.10)
    assert not triggered(0.05, tau=0.10)
