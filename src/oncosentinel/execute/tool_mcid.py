from __future__ import annotations

from typing import Any

from oncosentinel.knowledge.audit_records import ToolReturn

VERSION = "L1-guideline-anchor"

_THRESHOLDS: dict[str, tuple[float, int, str]] = {
    "ORR": (0.10, 1, "RECIST_1.1"),
    "PFS_HR": (0.75, 1, "iRECIST"),
    "LVEF": (0.10, 1, "ASCO_ESC_2022"),
    "TROPONIN": (0.99, 1, "ESC_2022_URL"),
}


def lookup_mcid(inputs: dict[str, Any]) -> ToolReturn:
    metric = str(inputs.get("metric", "ORR")).upper()
    if metric not in _THRESHOLDS:
        raise KeyError(f"no MCID anchor for metric: {metric}")
    mcid, tier, source = _THRESHOLDS[metric]
    payload = {
        "metric": metric,
        "regimen": inputs.get("regimen", ""),
        "mcid": mcid,
        "tier": tier,
        "source": source,
    }
    return ToolReturn(payload=payload, version=VERSION)
