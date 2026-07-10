from __future__ import annotations

import attrs
import numpy as np

from oncosentinel.analyze.primitives import projected_scalar, softmax, stable_seed
from oncosentinel.knowledge.clinical_records import CaseFeatures, ImmuneContexture

CELL_TYPES = ("CD8_T", "CD4_T", "Macrophage", "NK")


@attrs.define
class ImmuneContextureAgent:
    seed: int = 271

    def infer(self, case: CaseFeatures) -> ImmuneContexture:
        base = stable_seed(case.case_id) ^ self.seed
        raw = np.array(
            [projected_scalar(base + i, case.omics) for i in range(len(CELL_TYPES))],
            dtype=np.float64,
        )
        fractions = softmax(raw * 2.0)
        til_fractions = {
            name: float(value) for name, value in zip(CELL_TYPES, fractions, strict=True)
        }
        ifn_gamma = 5.0 + 4.0 * projected_scalar(base + 11, case.omics)
        tmb = max(0.0, 8.0 + 6.0 * projected_scalar(base + 23, case.omics))
        pdl1_cps = float(np.clip(50.0 + 45.0 * projected_scalar(base + 37, case.omics), 0.0, 100.0))
        inflamed_score = ifn_gamma - 5.0 + (til_fractions["CD8_T"] - 0.25) * 10.0
        phenotype = "hot" if inflamed_score > 0.0 else "cold"
        embedding = np.array(
            [til_fractions["CD8_T"], ifn_gamma / 10.0, tmb / 30.0, pdl1_cps / 100.0],
            dtype=np.float64,
        )
        return ImmuneContexture(
            til_fractions=til_fractions,
            ifn_gamma=float(ifn_gamma),
            tmb=float(tmb),
            pdl1_cps=pdl1_cps,
            phenotype=phenotype,
            embedding=embedding,
        )
