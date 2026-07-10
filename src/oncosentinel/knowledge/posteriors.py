from __future__ import annotations

import math

import attrs


@attrs.frozen
class ResponsePosterior:
    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return float(self.alpha / (self.alpha + self.beta))

    @property
    def variance(self) -> float:
        s = self.alpha + self.beta
        return float(self.alpha * self.beta / (s * s * (s + 1.0)))

    @property
    def concentration(self) -> float:
        return float(self.alpha + self.beta)


@attrs.frozen
class LvefPosterior:
    mu: float
    sigma2: float

    @property
    def mean(self) -> float:
        return float(self.mu)

    @property
    def std(self) -> float:
        return float(math.sqrt(self.sigma2))

    @property
    def stable_probability(self) -> float:
        return float(min(max(1.0 - self.mu, 0.0), 1.0))


@attrs.frozen
class JointPosterior:
    response: ResponsePosterior
    lvef: LvefPosterior

    @property
    def efficacy(self) -> float:
        return self.response.mean

    @property
    def safety(self) -> float:
        return self.lvef.stable_probability

    @property
    def point(self) -> tuple[float, float]:
        return (self.efficacy, self.safety)
