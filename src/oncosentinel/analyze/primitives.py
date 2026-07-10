from __future__ import annotations

import hashlib
import math

import numpy as np
from numpy.typing import NDArray


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    z = math.exp(x)
    return float(z / (1.0 + z))


def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def softmax(values: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return np.asarray(exp / np.sum(exp), dtype=np.float64)


def projected_scalar(seed: int, features: NDArray[np.float64]) -> float:
    flat = np.asarray(features, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed + 7 * flat.shape[0])
    direction = rng.standard_normal(flat.shape[0])
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    return float(np.dot(direction, flat) / max(1.0, np.sqrt(flat.shape[0])))
