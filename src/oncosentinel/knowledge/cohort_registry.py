from __future__ import annotations

import numpy as np

from oncosentinel.knowledge.clinical_records import RegimenCandidate
from oncosentinel.knowledge.settings_schema import CohortConfig
from oncosentinel.monitor.adaptive import Biomarker

DEFAULT_ANCHORS: tuple[CohortConfig, ...] = (
    CohortConfig("TCGA", 4218, 33, "multi-omics", "open", "portal.gdc.cancer.gov", seed=1),
    CohortConfig("IMvigor210", 298, 1, "omics+clinical", "open", "ESMO-Open-supplement", seed=2),
    CohortConfig("OAK", 850, 1, "omics+clinical", "open", "published-supplement", seed=3),
    CohortConfig("UKB-CMR", 4612, 1, "cardiac-mri", "restricted", "ukbiobank.ac.uk", seed=4),
    CohortConfig(
        "CARDIOTOX_CARDIOCARE",
        421,
        1,
        "multi-vendor-echo",
        "CC-BY-NC",
        "zenodo.org/records/19727929",
        seed=5,
    ),
    CohortConfig("MIMIC-IV-ECG", 2741, 1, "ecg", "restricted", "physionet.org", seed=6),
)

CANDIDATE_REGIMENS: tuple[RegimenCandidate, ...] = (
    RegimenCandidate("pembrolizumab_mono", ("pembrolizumab",), ("PDCD1",)),
    RegimenCandidate("atezolizumab_mono", ("atezolizumab",), ("CD274",)),
    RegimenCandidate("nivolumab_ipilimumab", ("nivolumab", "ipilimumab"), ("PDCD1", "CTLA4")),
    RegimenCandidate(
        "atezolizumab_carboplatin", ("atezolizumab", "carboplatin"), ("CD274", "ERCC1")
    ),
    RegimenCandidate(
        "pembrolizumab_pemetrexed", ("pembrolizumab", "pemetrexed"), ("PDCD1", "TYMS")
    ),
)


def longitudinal_stream(case_id: str, length: int, seed: int) -> list[Biomarker]:
    rng = np.random.default_rng(seed)
    kinds = ("TROPONIN", "ECHO", "RECIST", "IRECIST")
    stream: list[Biomarker] = []
    for _ in range(length):
        kind = kinds[int(rng.integers(0, len(kinds)))]
        if kind in ("TROPONIN", "ECHO"):
            value = float(np.clip(rng.normal(0.18, 0.08), 0.0, 1.0))
        else:
            value = float(rng.integers(0, 2))
        stream.append(Biomarker(kind=kind, value=value))
    return stream
