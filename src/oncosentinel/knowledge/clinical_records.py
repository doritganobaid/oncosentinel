from __future__ import annotations

import attrs
import numpy as np
from numpy.typing import NDArray


@attrs.define(eq=False)
class CaseFeatures:
    case_id: str
    cancer_type: str
    omics: NDArray[np.float64]
    ecg: NDArray[np.float64]
    echo: NDArray[np.float64]
    clinical: dict[str, float]

    @property
    def omics_dim(self) -> int:
        return int(self.omics.shape[-1])

    @property
    def cardiac_dim(self) -> int:
        return int(self.ecg.shape[-1] + self.echo.shape[-1])


@attrs.define(eq=False)
class ImmuneContexture:
    til_fractions: dict[str, float]
    ifn_gamma: float
    tmb: float
    pdl1_cps: float
    phenotype: str
    embedding: NDArray[np.float64]

    @property
    def inflamed(self) -> bool:
        return self.phenotype == "hot"


@attrs.frozen
class RegimenCandidate:
    name: str
    drugs: tuple[str, ...]
    genes: tuple[str, ...]

    @property
    def is_combination(self) -> bool:
        return len(self.drugs) > 1


@attrs.frozen
class RegimenRecommendation:
    regimen: str
    hv_argmax: float
    frontier_size: int
    mcid_crossing: bool
    efficacy: float
    safety: float
