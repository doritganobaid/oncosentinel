from __future__ import annotations

import math

import attrs
import numpy as np

from oncosentinel.analyze.cardiotox import CardiacAssessment
from oncosentinel.execute.tool_dispatch import CallOutcome, Dispatcher
from oncosentinel.knowledge.clinical_records import (
    ImmuneContexture,
    RegimenCandidate,
    RegimenRecommendation,
)
from oncosentinel.knowledge.posteriors import JointPosterior, ResponsePosterior
from oncosentinel.plan.constrained import RegimenDeltas, constrained_recommend
from oncosentinel.plan.mcid import McidAnchors

AGENT = "TreatmentPlanner"


@attrs.frozen
class ToolEvent:
    tool: str
    inputs: dict[str, object]
    output: dict[str, object]
    version: str
    provenance: str
    attempts: int


@attrs.frozen
class PlanResult:
    recommendation: RegimenRecommendation
    joint: JointPosterior
    tool_events: list[ToolEvent]
    kept_regimens: list[str]


@attrs.define
class TreatmentPlannerAgent:
    dispatcher: Dispatcher
    reference: tuple[float, float] = (0.0, 0.0)
    baseline_orr: float = 0.20
    coverage_delta: float = 0.10
    anchors: McidAnchors = attrs.field(factory=McidAnchors)

    def _efficacy(self, base: float, immune: ImmuneContexture, severity: str, combo: bool) -> float:
        value = base
        if combo and immune.inflamed:
            value *= 1.15
        elif combo:
            value *= 1.05
        if severity == "moderate":
            value *= 0.93
        return float(min(max(value, 0.01), 0.99))

    def _safety(self, base: float, severity: str, combo: bool) -> float:
        value = base
        if combo:
            value *= 0.92
        if severity == "moderate":
            value *= 0.95
        return float(min(max(value, 0.01), 0.99))

    def _lvef_drop_pp(self, lvef_mean: float, severity: str, combo: bool) -> float:
        factor = 1.0
        if combo:
            factor *= 1.4
        if severity == "moderate":
            factor += 0.1
        return float(25.0 * lvef_mean * factor)

    def plan(
        self,
        response: ResponsePosterior,
        cardiac: CardiacAssessment,
        immune: ImmuneContexture,
        candidates: list[RegimenCandidate],
    ) -> PlanResult:
        events: list[ToolEvent] = []
        points: list[list[float]] = []
        orr_mean: list[float] = []
        orr_std: list[float] = []
        lvef_mean: list[float] = []
        lvef_std: list[float] = []
        kept: list[RegimenCandidate] = []

        for regimen in candidates:
            dgi = self.dispatcher.call(
                AGENT, "query_DGIdb", {"genes": list(regimen.genes), "drugs": list(regimen.drugs)}
            )
            events.append(_event("query_DGIdb", dgi))
            severity = str(dgi.result.payload["severity"])

            mcid = self.dispatcher.call(
                AGENT, "lookup_MCID", {"metric": "ORR", "regimen": regimen.name}
            )
            events.append(_event("lookup_MCID", mcid))

            contra = self.dispatcher.call(
                AGENT,
                "check_contraindication",
                {
                    "regimen": regimen.name,
                    "hfa_icos": cardiac.hfa_icos_stratum,
                    "is_combination": regimen.is_combination,
                    "interaction_severity": severity,
                    "baseline_contraindicated": cardiac.contraindicated,
                },
            )
            events.append(_event("check_contraindication", contra))
            if bool(contra.result.payload["contraindicated"]):
                continue

            efficacy = self._efficacy(response.mean, immune, severity, regimen.is_combination)
            safety = self._safety(
                cardiac.posterior.stable_probability, severity, regimen.is_combination
            )
            points.append([efficacy, safety])
            orr_mean.append((efficacy - self.baseline_orr) * 100.0)
            orr_std.append(max(100.0 * math.sqrt(response.variance), 3.0))
            lvef_mean.append(
                self._lvef_drop_pp(cardiac.posterior.mean, severity, regimen.is_combination)
            )
            lvef_std.append(2.0)
            kept.append(regimen)

        joint = JointPosterior(response=response, lvef=cardiac.posterior)
        if not kept:
            recommendation = RegimenRecommendation(
                regimen="escalate_to_tumor_board",
                hv_argmax=0.0,
                frontier_size=0,
                mcid_crossing=False,
                efficacy=response.mean,
                safety=cardiac.posterior.stable_probability,
            )
            return PlanResult(recommendation, joint, events, [])

        frontier = self.dispatcher.call(
            AGENT, "compute_pareto_frontier", {"points": points, "reference": list(self.reference)}
        )
        events.append(_event("compute_pareto_frontier", frontier))

        deltas = RegimenDeltas(
            orr_mean_pp=np.asarray(orr_mean, dtype=np.float64),
            orr_std_pp=np.asarray(orr_std, dtype=np.float64),
            lvef_mean_pp=np.asarray(lvef_mean, dtype=np.float64),
            lvef_std_pp=np.asarray(lvef_std, dtype=np.float64),
        )
        selection = constrained_recommend(
            np.asarray(points, dtype=np.float64),
            self.reference,
            deltas,
            self.anchors,
            self.coverage_delta,
        )
        best = kept[selection.argmax_index]
        recommendation = RegimenRecommendation(
            regimen=best.name,
            hv_argmax=selection.hv_argmax,
            frontier_size=selection.frontier_size,
            mcid_crossing=selection.mcid_crossing,
            efficacy=points[selection.argmax_index][0],
            safety=points[selection.argmax_index][1],
        )
        return PlanResult(recommendation, joint, events, [regimen.name for regimen in kept])


def _event(tool: str, outcome: CallOutcome) -> ToolEvent:
    return ToolEvent(
        tool=tool,
        inputs=dict(outcome.inputs),
        output=dict(outcome.result.payload),
        version=outcome.result.version,
        provenance=outcome.provenance,
        attempts=outcome.attempts,
    )
