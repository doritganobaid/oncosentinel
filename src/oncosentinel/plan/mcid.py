from __future__ import annotations

import attrs


@attrs.frozen
class McidAnchors:
    orr_pp: float = 10.0
    pfs_hr: float = 0.75
    lvef_drop_pp: float = 10.0
    troponin_percentile: float = 99.0

    def feasible(self, delta_orr_pp: float, delta_lvef_drop_pp: float) -> bool:
        return delta_orr_pp >= self.orr_pp and delta_lvef_drop_pp <= self.lvef_drop_pp

    def crossing(self, delta_orr_pp: float, delta_lvef_drop_pp: float) -> bool:
        return self.feasible(delta_orr_pp, delta_lvef_drop_pp)
