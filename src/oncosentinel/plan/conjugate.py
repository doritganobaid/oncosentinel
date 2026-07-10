from __future__ import annotations

from oncosentinel.knowledge.posteriors import LvefPosterior, ResponsePosterior


def beta_from_moments(mean: float, variance: float) -> tuple[float, float]:
    if not 0.0 < mean < 1.0:
        raise ValueError("mean must lie in (0, 1)")
    max_var = mean * (1.0 - mean)
    if not 0.0 < variance < max_var:
        raise ValueError("variance must lie in (0, mean*(1-mean))")
    concentration = mean * (1.0 - mean) / variance - 1.0
    alpha = mean * concentration
    beta = (1.0 - mean) * concentration
    return float(alpha), float(beta)


def beta_binomial_update(
    prior: ResponsePosterior, successes: int, trials: int
) -> ResponsePosterior:
    if successes < 0 or trials < successes:
        raise ValueError("require 0 <= successes <= trials")
    return ResponsePosterior(
        alpha=prior.alpha + float(successes),
        beta=prior.beta + float(trials - successes),
    )


def gaussian_update(
    prior: LvefPosterior, observation: float, observation_variance: float
) -> LvefPosterior:
    if observation_variance <= 0.0 or prior.sigma2 <= 0.0:
        raise ValueError("variances must be positive")
    precision = 1.0 / prior.sigma2 + 1.0 / observation_variance
    posterior_variance = 1.0 / precision
    posterior_mean = posterior_variance * (
        prior.mu / prior.sigma2 + observation / observation_variance
    )
    return LvefPosterior(mu=float(posterior_mean), sigma2=float(posterior_variance))


def response_posterior_from_probability(
    probability: float, concentration: float
) -> ResponsePosterior:
    p = float(min(max(probability, 1e-6), 1.0 - 1e-6))
    return ResponsePosterior(alpha=p * concentration, beta=(1.0 - p) * concentration)
