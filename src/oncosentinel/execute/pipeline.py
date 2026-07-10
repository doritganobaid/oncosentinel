from __future__ import annotations

from collections.abc import Sequence

import attrs
import numpy as np

from oncosentinel.analyze.backbone_bulkrnabert import BulkRnaBertSurrogate
from oncosentinel.analyze.backbone_ecg import EcgFmSurrogate
from oncosentinel.analyze.backbone_ensemble import OmicsEnsemble
from oncosentinel.analyze.backbone_geneformer import GeneformerV2Surrogate
from oncosentinel.analyze.backbone_scgpt import ScGptV2Surrogate
from oncosentinel.analyze.cardiotox import CardiacAssessment, CardiotoxicityRiskAgent
from oncosentinel.analyze.immune import CELL_TYPES, ImmuneContextureAgent
from oncosentinel.analyze.omics import OmicsReasoningAgent
from oncosentinel.execute.controller import build_dispatcher
from oncosentinel.knowledge.clinical_records import (
    CaseFeatures,
    ImmuneContexture,
    RegimenCandidate,
    RegimenRecommendation,
)
from oncosentinel.knowledge.ledger import EvidenceChain
from oncosentinel.knowledge.posteriors import JointPosterior, LvefPosterior, ResponsePosterior
from oncosentinel.knowledge.settings_schema import RunConfig
from oncosentinel.monitor.adaptive import AdaptiveMonitorAgent, Biomarker
from oncosentinel.plan.conjugate import response_posterior_from_probability
from oncosentinel.plan.mcid import McidAnchors
from oncosentinel.plan.planner import PlanResult, TreatmentPlannerAgent


@attrs.define
class RunResult:
    recommendation: RegimenRecommendation
    joint: JointPosterior
    evidence: EvidenceChain
    reinvocations: int
    kl_values: list[float]


def _neutral_contexture() -> ImmuneContexture:
    uniform = {name: 1.0 / len(CELL_TYPES) for name in CELL_TYPES}
    return ImmuneContexture(
        til_fractions=uniform,
        ifn_gamma=4.0,
        tmb=0.0,
        pdl1_cps=0.0,
        phenotype="cold",
        embedding=np.zeros(4, dtype=np.float64),
    )


@attrs.define
class Orchestrator:
    config: RunConfig
    omics: OmicsReasoningAgent
    immune: ImmuneContextureAgent
    cardio: CardiotoxicityRiskAgent
    planner: TreatmentPlannerAgent
    monitor: AdaptiveMonitorAgent

    @classmethod
    def from_config(cls, config: RunConfig) -> Orchestrator:
        ensemble = OmicsEnsemble(
            members=[BulkRnaBertSurrogate(), ScGptV2Surrogate(), GeneformerV2Surrogate()],
            weights=dict(config.backbone.weights),
        )
        omics = OmicsReasoningAgent(ensemble=ensemble, concentration=config.backbone.concentration)
        immune = ImmuneContextureAgent()
        cardio = CardiotoxicityRiskAgent(
            ecg_backbone=EcgFmSurrogate(),
            prior_mu=config.priors.lvef_mu,
            prior_sigma2=config.priors.lvef_var,
            observation_variance=config.priors.lvef_var,
        )
        anchors = McidAnchors(
            orr_pp=config.mcid.orr_pp,
            pfs_hr=config.mcid.pfs_hr,
            lvef_drop_pp=config.mcid.lvef_drop_pp,
            troponin_percentile=config.mcid.troponin_percentile,
        )
        planner = TreatmentPlannerAgent(
            dispatcher=build_dispatcher(),
            reference=config.kernel.reference,
            baseline_orr=config.baseline_orr,
            coverage_delta=config.kernel.coverage_delta,
            anchors=anchors,
        )
        monitor = AdaptiveMonitorAgent(
            tau=config.kernel.tau, observation_variance=config.priors.lvef_var
        )
        return cls(
            config=config,
            omics=omics,
            immune=immune,
            cardio=cardio,
            planner=planner,
            monitor=monitor,
        )

    def _naive_select(
        self, response_mean: float, cardiac: CardiacAssessment, candidates: list[RegimenCandidate]
    ) -> RegimenRecommendation:
        best = max(candidates, key=lambda r: response_mean - 0.05 * float(r.is_combination))
        delta_orr = (response_mean - self.config.baseline_orr) * 100.0
        crossing = delta_orr >= self.config.mcid.orr_pp
        return RegimenRecommendation(
            regimen=best.name,
            hv_argmax=0.0,
            frontier_size=0,
            mcid_crossing=crossing,
            efficacy=response_mean,
            safety=cardiac.posterior.stable_probability,
        )

    def run(
        self,
        case: CaseFeatures,
        candidates: list[RegimenCandidate],
        longitudinal: Sequence[Biomarker] = (),
    ) -> RunResult:
        enabled = set(self.config.enabled_agents)
        evidence = EvidenceChain()

        if "omics" in enabled:
            response = self.omics.infer(case)
        else:
            response = response_posterior_from_probability(
                self.config.priors.resp_mean, self.config.backbone.concentration
            )
        evidence.append(
            "OmicsReasoning",
            "compute_posterior",
            {"case": case.case_id},
            {"p_resp_mean": response.mean},
        )

        immune = self.immune.infer(case) if "immune" in enabled else _neutral_contexture()
        evidence.append(
            "ImmuneContexture",
            "compute_contexture",
            {"case": case.case_id},
            {
                "phenotype": immune.phenotype,
                "ifn_gamma": immune.ifn_gamma,
                "pdl1_cps": immune.pdl1_cps,
            },
        )

        if "cardio" in enabled:
            cardiac = self.cardio.infer(case)
        else:
            cardiac = CardiacAssessment(
                posterior=LvefPosterior(
                    mu=self.config.priors.lvef_mu, sigma2=self.config.priors.lvef_var
                ),
                contraindicated=False,
                hfa_icos_stratum=1,
            )
        evidence.append(
            "CardiotoxicityRisk",
            "compute_LVEF_posterior",
            {"case": case.case_id, "hfa_icos": cardiac.hfa_icos_stratum},
            {"p_lvef_mean": cardiac.posterior.mean, "contraindicated": cardiac.contraindicated},
        )

        recommendation, joint = self._plan(response, cardiac, immune, candidates, evidence, enabled)
        evidence.append(
            "TreatmentPlanner",
            "select_recommendation",
            {"case": case.case_id},
            {
                "regimen": recommendation.regimen,
                "hv_argmax": recommendation.hv_argmax,
                "mcid_crossing": recommendation.mcid_crossing,
                "frontier_size": recommendation.frontier_size,
            },
        )

        kl_values: list[float] = []
        reinvocations = 0
        if "monitor" in enabled:
            current = joint
            for biomarker in longitudinal:
                result = self.monitor.update(current, biomarker)
                kl_values.append(result.kl)
                evidence.append(
                    "AdaptiveMonitor",
                    "compute_KL",
                    {"kind": biomarker.kind, "value": biomarker.value},
                    {"KL": result.kl, "trigger": result.reinvoke},
                )
                current = result.posterior
                if result.reinvoke:
                    reinvocations += 1
                    updated_cardiac = CardiacAssessment(
                        posterior=current.lvef,
                        contraindicated=cardiac.contraindicated,
                        hfa_icos_stratum=cardiac.hfa_icos_stratum,
                    )
                    recommendation, current = self._plan(
                        current.response, updated_cardiac, immune, candidates, evidence, enabled
                    )
                joint = current

        return RunResult(
            recommendation=recommendation,
            joint=joint,
            evidence=evidence,
            reinvocations=reinvocations,
            kl_values=kl_values,
        )

    def _plan(
        self,
        response: ResponsePosterior,
        cardiac: CardiacAssessment,
        immune: ImmuneContexture,
        candidates: list[RegimenCandidate],
        evidence: EvidenceChain,
        enabled: set[str],
    ) -> tuple[RegimenRecommendation, JointPosterior]:
        if "planner" in enabled and self.config.tool_layer_enabled:
            plan: PlanResult = self.planner.plan(response, cardiac, immune, candidates)
            for event in plan.tool_events:
                evidence.append("TreatmentPlanner", event.tool, event.inputs, event.output)
            return plan.recommendation, plan.joint
        recommendation = self._naive_select(response.mean, cardiac, candidates)
        joint = JointPosterior(response=response, lvef=cardiac.posterior)
        return recommendation, joint
