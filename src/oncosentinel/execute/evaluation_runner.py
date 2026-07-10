from __future__ import annotations

from typing import Any

import numpy as np

from oncosentinel.analyze.findings import complexity_index, tool_call_scaling
from oncosentinel.analyze.metrics import auroc, bootstrap_auroc_ci, mcid_crossing_rate
from oncosentinel.execute.pipeline import Orchestrator
from oncosentinel.knowledge.cohort_registry import CANDIDATE_REGIMENS
from oncosentinel.knowledge.cohort_synthesis import generate_cohort
from oncosentinel.knowledge.settings_schema import RunConfig

TOOL_ACTIONS = frozenset(
    {"query_DGIdb", "lookup_MCID", "check_contraindication", "compute_pareto_frontier"}
)


def evaluate_experiment(config: RunConfig, limit: int | None = None) -> dict[str, Any]:
    orchestrator = Orchestrator.from_config(config)
    candidates = list(CANDIDATE_REGIMENS)
    per_cohort: dict[str, dict[str, float]] = {}
    all_efficacy: list[float] = []
    all_response_label: list[int] = []
    all_dysfunction: list[float] = []
    all_lvef_label: list[int] = []
    all_crossing: list[bool] = []
    complexity: list[float] = []
    tool_counts: list[float] = []

    for cohort in config.cohorts:
        cases = generate_cohort(cohort, limit=limit)
        efficacy: list[float] = []
        response_label: list[int] = []
        dysfunction: list[float] = []
        lvef_label: list[int] = []
        for labeled in cases:
            result = orchestrator.run(labeled.features, candidates)
            efficacy.append(result.recommendation.efficacy)
            response_label.append(labeled.response_label)
            dysfunction.append(1.0 - result.recommendation.safety)
            lvef_label.append(labeled.lvef_label)
            all_crossing.append(result.recommendation.mcid_crossing)
            active_omics = int(np.sum(np.abs(labeled.features.omics) > 0.5))
            active_cardiac = int(
                np.sum(np.abs(labeled.features.ecg) > 0.5)
                + np.sum(np.abs(labeled.features.echo) > 0.5)
            )
            complexity.append(complexity_index(active_omics, active_cardiac, cohort.n_cancer_types))
            tool_counts.append(
                float(sum(1 for record in result.evidence.records if record.action in TOOL_ACTIONS))
            )
        scores = np.asarray(efficacy, dtype=np.float64)
        labels = np.asarray(response_label, dtype=np.int_)
        cardiac_scores = np.asarray(dysfunction, dtype=np.float64)
        cardiac_labels = np.asarray(lvef_label, dtype=np.int_)
        per_cohort[cohort.name] = {
            "response_auc": auroc(scores, labels),
            "lvef_auc": auroc(cardiac_scores, cardiac_labels),
            "n": float(len(cases)),
        }
        all_efficacy.extend(efficacy)
        all_response_label.extend(response_label)
        all_dysfunction.extend(dysfunction)
        all_lvef_label.extend(lvef_label)

    response_point, response_lo, response_hi = bootstrap_auroc_ci(
        np.asarray(all_efficacy, dtype=np.float64),
        np.asarray(all_response_label, dtype=np.int_),
        iters=min(config.bootstrap_iters, 2000),
        seed=config.seeds[0],
    )
    rho, slope = tool_call_scaling(
        np.asarray(complexity, dtype=np.float64), np.asarray(tool_counts, dtype=np.float64)
    )
    return {
        "experiment": config.experiment,
        "per_cohort": per_cohort,
        "response_auc": response_point,
        "response_auc_ci": [response_lo, response_hi],
        "lvef_auc": auroc(
            np.asarray(all_dysfunction, dtype=np.float64),
            np.asarray(all_lvef_label, dtype=np.int_),
        ),
        "mcid_crossing_rate": mcid_crossing_rate(np.asarray(all_crossing, dtype=np.bool_)),
        "tool_call_spearman": rho,
        "tool_call_slope": slope,
        "n_cases": len(all_efficacy),
    }
