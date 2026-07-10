from __future__ import annotations

import attrs

from oncosentinel.analyze.backbone_ensemble import OmicsEnsemble
from oncosentinel.analyze.primitives import sigmoid
from oncosentinel.knowledge.clinical_records import CaseFeatures
from oncosentinel.knowledge.posteriors import ResponsePosterior
from oncosentinel.plan.conjugate import response_posterior_from_probability


@attrs.define
class OmicsReasoningAgent:
    ensemble: OmicsEnsemble
    concentration: float = 5.25
    calibration_bias: float = 0.0

    @classmethod
    def default(cls) -> OmicsReasoningAgent:
        return cls(ensemble=OmicsEnsemble.default())

    def infer(self, case: CaseFeatures) -> ResponsePosterior:
        logit = self.ensemble.response_logit(case.omics) + self.calibration_bias
        probability = sigmoid(logit)
        return response_posterior_from_probability(probability, self.concentration)
