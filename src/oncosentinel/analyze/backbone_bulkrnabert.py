from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from oncosentinel.analyze.backbone_protocol import projected_logit

PANCANCER_F1_CEILING = 0.942


class BulkRnaBertSurrogate:
    name = "BulkRNABert"

    def __init__(self, seed: int = 91) -> None:
        self.seed = seed
        self.scale = 1.30

    def response_logit(self, features: NDArray[np.float64]) -> float:
        return projected_logit(self.seed, features, self.scale)
