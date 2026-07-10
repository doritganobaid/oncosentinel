from __future__ import annotations

import attrs

from oncosentinel.knowledge.posteriors import JointPosterior
from oncosentinel.monitor.triggers import joint_kl, triggered
from oncosentinel.plan.conjugate import beta_binomial_update, gaussian_update

CARDIAC_BIOMARKERS = frozenset({"TROPONIN", "ECHO"})
RESPONSE_BIOMARKERS = frozenset({"RECIST", "IRECIST"})


@attrs.frozen
class Biomarker:
    kind: str
    value: float


@attrs.frozen
class MonitorResult:
    posterior: JointPosterior
    kl: float
    reinvoke: bool


@attrs.define
class AdaptiveMonitorAgent:
    tau: float = 0.10
    observation_variance: float = 0.02

    def update(self, prior: JointPosterior, biomarker: Biomarker) -> MonitorResult:
        response = prior.response
        lvef = prior.lvef
        if biomarker.kind in CARDIAC_BIOMARKERS:
            lvef = gaussian_update(lvef, biomarker.value, self.observation_variance)
        elif biomarker.kind in RESPONSE_BIOMARKERS:
            response = beta_binomial_update(response, int(round(biomarker.value)), 1)
        else:
            raise ValueError(f"unsupported biomarker kind: {biomarker.kind}")
        updated = JointPosterior(response=response, lvef=lvef)
        divergence = joint_kl(updated, prior)
        return MonitorResult(
            posterior=updated, kl=divergence, reinvoke=triggered(divergence, self.tau)
        )
