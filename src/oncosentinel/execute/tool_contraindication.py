from __future__ import annotations

from typing import Any

from oncosentinel.knowledge.audit_records import ToolReturn

VERSION = "NCCN-rule-table"


def check_contraindication(inputs: dict[str, Any]) -> ToolReturn:
    hfa_icos = int(inputs.get("hfa_icos", 1))
    is_combination = bool(inputs.get("is_combination", False))
    severity = str(inputs.get("interaction_severity", "none"))
    baseline_contra = bool(inputs.get("baseline_contraindicated", False))
    reasons: list[str] = []
    if severity == "major":
        reasons.append("major_drug_interaction")
    if (hfa_icos >= 2 or baseline_contra) and is_combination:
        reasons.append("high_baseline_cardiac_risk_with_combination")
    payload = {
        "regimen": inputs.get("regimen", ""),
        "contraindicated": len(reasons) > 0,
        "reasons": reasons,
    }
    return ToolReturn(payload=payload, version=VERSION)
