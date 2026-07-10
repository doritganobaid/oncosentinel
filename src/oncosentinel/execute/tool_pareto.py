from __future__ import annotations

from typing import Any

import numpy as np

from oncosentinel.knowledge.audit_records import ToolReturn
from oncosentinel.plan.pareto import recommend

VERSION = "hypervolume-sorted-scan"


def compute_pareto_frontier(inputs: dict[str, Any]) -> ToolReturn:
    points = np.asarray(inputs["points"], dtype=np.float64)
    reference = tuple(float(v) for v in inputs.get("reference", (0.0, 0.0)))
    selection = recommend(points, (reference[0], reference[1]))
    front = selection.front_indices.tolist()
    payload = {
        "frontier_size": selection.frontier_size,
        "argmax_index": selection.argmax_index,
        "hv_total": selection.hv_total,
        "hv_argmax": selection.hv_argmax,
        "front_indices": front,
        "hv_contributions": [float(selection.contributions[i]) for i in front],
    }
    return ToolReturn(payload=payload, version=VERSION)
