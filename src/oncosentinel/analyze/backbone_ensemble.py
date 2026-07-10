from __future__ import annotations

import attrs
import numpy as np
from numpy.typing import NDArray

from oncosentinel.analyze.backbone_bulkrnabert import BulkRnaBertSurrogate
from oncosentinel.analyze.backbone_geneformer import GeneformerV2Surrogate
from oncosentinel.analyze.backbone_protocol import OmicsBackbone
from oncosentinel.analyze.backbone_scgpt import ScGptV2Surrogate

DEFAULT_WEIGHTS = {"BulkRNABert": 0.46, "scGPT_v2": 0.42, "Geneformer_v2": 0.12}


@attrs.define
class OmicsEnsemble:
    members: list[OmicsBackbone]
    weights: dict[str, float]

    @classmethod
    def default(cls) -> OmicsEnsemble:
        members: list[OmicsBackbone] = [
            BulkRnaBertSurrogate(),
            ScGptV2Surrogate(),
            GeneformerV2Surrogate(),
        ]
        return cls(members=members, weights=dict(DEFAULT_WEIGHTS))

    def response_logit(self, features: NDArray[np.float64]) -> float:
        total_weight = 0.0
        accumulated = 0.0
        for member in self.members:
            weight = self.weights.get(member.name, 0.0)
            accumulated += weight * member.response_logit(features)
            total_weight += weight
        if total_weight == 0.0:
            raise ValueError("ensemble weights sum to zero")
        return float(accumulated / total_weight)
