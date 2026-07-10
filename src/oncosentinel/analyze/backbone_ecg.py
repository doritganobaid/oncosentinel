from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from oncosentinel.analyze.backbone_protocol import projected_logit


class EcgFmSurrogate:
    name = "ECG-FM/ECGFounder"

    def __init__(self, seed: int = 131) -> None:
        self.seed = seed
        self.scale = 1.05

    def risk_logit(self, signal: NDArray[np.float64]) -> float:
        return projected_logit(self.seed, signal, self.scale)
