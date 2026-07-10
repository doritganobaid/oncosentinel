from __future__ import annotations

import math

from oncosentinel.knowledge.posteriors import LvefPosterior, ResponsePosterior
from oncosentinel.plan.conjugate import (
    beta_binomial_update,
    beta_from_moments,
    gaussian_update,
    response_posterior_from_probability,
)


def test_beta_from_moments_recovers_mean_and_variance() -> None:
    alpha, beta = beta_from_moments(0.50, 0.04)
    posterior = ResponsePosterior(alpha=alpha, beta=beta)
    assert math.isclose(posterior.mean, 0.50, abs_tol=1e-9)
    assert math.isclose(posterior.variance, 0.04, abs_tol=1e-9)


def test_beta_binomial_update_increments_and_monotone() -> None:
    prior = ResponsePosterior(alpha=2.0, beta=2.0)
    low = beta_binomial_update(prior, successes=1, trials=4)
    high = beta_binomial_update(prior, successes=3, trials=4)
    assert math.isclose(low.alpha, 3.0) and math.isclose(low.beta, 5.0)
    assert high.mean > low.mean


def test_gaussian_update_shrinks_variance_and_interpolates_mean() -> None:
    prior = LvefPosterior(mu=0.15, sigma2=0.02)
    posterior = gaussian_update(prior, observation=0.35, observation_variance=0.02)
    assert posterior.sigma2 < prior.sigma2
    assert posterior.sigma2 < 0.02
    assert 0.15 < posterior.mean < 0.35


def test_response_posterior_from_probability_matches_probability() -> None:
    posterior = response_posterior_from_probability(0.72, concentration=10.0)
    assert math.isclose(posterior.mean, 0.72, abs_tol=1e-6)
    assert math.isclose(posterior.concentration, 10.0, abs_tol=1e-6)
