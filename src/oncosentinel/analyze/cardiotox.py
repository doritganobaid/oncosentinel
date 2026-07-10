from __future__ import annotations

import attrs
import numpy as np

from oncosentinel.analyze.backbone_ecg import EcgFmSurrogate
from oncosentinel.analyze.backbone_protocol import EcgBackbone
from oncosentinel.analyze.primitives import sigmoid
from oncosentinel.knowledge.clinical_records import CaseFeatures
from oncosentinel.knowledge.posteriors import LvefPosterior
from oncosentinel.plan.conjugate import gaussian_update


@attrs.define
class CardiacAssessment:
    posterior: LvefPosterior
    contraindicated: bool
    hfa_icos_stratum: int


@attrs.define
class CardiotoxicityRiskAgent:
    ecg_backbone: EcgBackbone
    prior_mu: float = 0.15
    prior_sigma2: float = 0.02
    observation_variance: float = 0.02
    ici_recalibration_pp: float = 0.05

    @classmethod
    def default(cls) -> CardiotoxicityRiskAgent:
        return cls(ecg_backbone=EcgFmSurrogate())

    def infer(self, case: CaseFeatures) -> CardiacAssessment:
        clinical_vector = np.array(
            [case.clinical.get(key, 0.0) for key in sorted(case.clinical)], dtype=np.float64
        )
        signal = np.concatenate([case.ecg.ravel(), case.echo.ravel(), clinical_vector])
        logit = self.ecg_backbone.risk_logit(signal)
        stratum = int(case.clinical.get("hfa_icos", 1.0))
        observed = sigmoid(logit + 0.4 * (stratum - 1)) + self.ici_recalibration_pp
        prior = LvefPosterior(mu=self.prior_mu, sigma2=self.prior_sigma2)
        posterior = gaussian_update(prior, observed, self.observation_variance)
        contraindicated = posterior.mean > 0.5 or stratum >= 2
        return CardiacAssessment(
            posterior=posterior, contraindicated=contraindicated, hfa_icos_stratum=stratum
        )
