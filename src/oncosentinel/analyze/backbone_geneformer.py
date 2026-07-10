from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from oncosentinel.analyze.backbone_protocol import projected_logit


class GeneformerV2Surrogate:
    name = "Geneformer_v2"

    def __init__(self, seed: int = 53) -> None:
        self.seed = seed
        self.scale = 0.95

    def response_logit(self, features: NDArray[np.float64]) -> float:
        return projected_logit(self.seed, features, self.scale)
