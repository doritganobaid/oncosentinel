from __future__ import annotations

import numpy as np

from oncosentinel.analyze.primitives import sigmoid
from oncosentinel.knowledge.clinical_records import CaseFeatures
from oncosentinel.knowledge.cohort_spec import FeatureDims, LabeledCase
from oncosentinel.knowledge.settings_schema import CohortConfig

CANCER_TYPES = (
    "NSCLC",
    "BLCA",
    "SKCM",
    "PAAD",
    "GBM",
    "LGG",
    "OV",
    "PRAD",
    "BRCA",
    "STAD",
    "KIRC",
    "LIHC",
    "COAD",
    "HNSC",
    "LUSC",
    "ESCA",
    "CESC",
    "UCEC",
    "THCA",
    "READ",
    "SARC",
    "MESO",
    "ACC",
    "DLBC",
    "KICH",
    "KIRP",
    "PCPG",
    "TGCT",
    "THYM",
    "UVM",
    "CHOL",
    "UCS",
    "GBMLGG",
)
IMMUNE_COLD = frozenset({"PAAD", "GBM", "LGG", "OV", "PRAD"})


def generate_case(case_id: str, cancer_type: str, dims: FeatureDims, seed: int) -> LabeledCase:
    rng = np.random.default_rng(seed)
    omics = rng.standard_normal(dims.omics)
    ecg = rng.standard_normal(dims.ecg)
    echo = rng.standard_normal(dims.echo)
    hfa_icos = float(rng.integers(0, 3))
    clinical = {
        "hfa_icos": hfa_icos,
        "age": float(45.0 + 30.0 * rng.random()),
        "troponin": float(rng.lognormal(mean=0.0, sigma=0.5)),
        "nt_probnp": float(rng.lognormal(mean=1.0, sigma=0.6)),
    }
    response_drive = float(np.dot(omics[:8], rng.standard_normal(8)) / 8.0)
    if cancer_type in IMMUNE_COLD:
        response_drive -= 0.6
    response_label = int(rng.random() < sigmoid(response_drive))
    cardiac_drive = 0.5 * hfa_icos - 1.0 + float(np.mean(ecg[:6]))
    lvef_label = int(rng.random() < sigmoid(cardiac_drive))
    features = CaseFeatures(
        case_id=case_id,
        cancer_type=cancer_type,
        omics=omics,
        ecg=ecg,
        echo=echo,
        clinical=clinical,
    )
    return LabeledCase(features=features, response_label=response_label, lvef_label=lvef_label)


def generate_cohort(
    cohort: CohortConfig, dims: FeatureDims | None = None, limit: int | None = None
) -> list[LabeledCase]:
    feature_dims = dims or FeatureDims()
    count = cohort.n_cases if limit is None else min(limit, cohort.n_cases)
    type_count = max(1, min(cohort.n_cancer_types, len(CANCER_TYPES)))
    cases: list[LabeledCase] = []
    for index in range(count):
        cancer_type = CANCER_TYPES[index % type_count]
        case_seed = cohort.seed * 1_000_003 + index
        cases.append(
            generate_case(f"{cohort.name}-{index:05d}", cancer_type, feature_dims, case_seed)
        )
    return cases
