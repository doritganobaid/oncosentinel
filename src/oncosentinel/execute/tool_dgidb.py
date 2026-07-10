from __future__ import annotations

import hashlib
from typing import Any

from oncosentinel.knowledge.audit_records import ToolReturn

VERSION = "DGIdb 5.0@2024-10"


def _severity(count: int) -> str:
    if count == 0:
        return "none"
    if count <= 2:
        return "minor"
    if count <= 4:
        return "moderate"
    return "major"


def query_dgidb(inputs: dict[str, Any]) -> ToolReturn:
    genes = tuple(inputs.get("genes", ()))
    drugs = tuple(inputs.get("drugs", ()))
    key = "|".join(sorted(genes)) + "#" + "|".join(sorted(drugs))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    count = digest[0] % 7
    payload = {
        "interaction_count": int(count),
        "severity": _severity(int(count)),
        "genes": list(genes),
        "drugs": list(drugs),
    }
    return ToolReturn(payload=payload, version=VERSION)
