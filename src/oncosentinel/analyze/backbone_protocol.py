from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class OmicsBackbone(Protocol):
    name: str

    def response_logit(self, features: NDArray[np.float64]) -> float: ...


class EcgBackbone(Protocol):
    name: str

    def risk_logit(self, signal: NDArray[np.float64]) -> float: ...


def seeded_projection(seed: int, dim: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed + 1_000_003 * dim)
    vector = rng.standard_normal(dim)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def projected_logit(seed: int, features: NDArray[np.float64], scale: float) -> float:
    flat = np.asarray(features, dtype=np.float64).ravel()
    projection = seeded_projection(seed, flat.shape[0])
    raw = float(np.dot(projection, flat)) / max(1.0, float(np.sqrt(flat.shape[0])))
    return float(scale * raw)
