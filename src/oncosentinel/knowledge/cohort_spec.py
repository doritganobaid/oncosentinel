from __future__ import annotations

import attrs

from oncosentinel.knowledge.clinical_records import CaseFeatures


@attrs.frozen
class LabeledCase:
    features: CaseFeatures
    response_label: int
    lvef_label: int


@attrs.frozen
class FeatureDims:
    omics: int = 64
    ecg: int = 60
    echo: int = 16
