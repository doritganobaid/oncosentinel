from __future__ import annotations

import math

from scipy.special import betaln, digamma

from oncosentinel.knowledge.posteriors import JointPosterior, LvefPosterior, ResponsePosterior


def beta_kl(current: ResponsePosterior, previous: ResponsePosterior) -> float:
    a1, b1 = current.alpha, current.beta
    a0, b0 = previous.alpha, previous.beta
    value = (
        betaln(a0, b0)
        - betaln(a1, b1)
        + (a1 - a0) * digamma(a1)
        + (b1 - b0) * digamma(b1)
        + (a0 - a1 + b0 - b1) * digamma(a1 + b1)
    )
    return float(max(value, 0.0))


def gaussian_kl(current: LvefPosterior, previous: LvefPosterior) -> float:
    s1, s0 = current.sigma2, previous.sigma2
    value = 0.5 * (math.log(s0 / s1) + (s1 + (current.mu - previous.mu) ** 2) / s0 - 1.0)
    return float(max(value, 0.0))


def joint_kl(current: JointPosterior, previous: JointPosterior) -> float:
    return beta_kl(current.response, previous.response) + gaussian_kl(current.lvef, previous.lvef)


def triggered(kl_value: float, tau: float) -> bool:
    return kl_value > tau
