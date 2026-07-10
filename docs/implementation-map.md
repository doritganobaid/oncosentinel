# Implementation Map

This document maps the package layout to the responsibilities of each module. The source code itself
carries no inline comments or docstrings; this map is the single place where module-level provenance and
intent are recorded.

## `src/oncosentinel`

Top-level package. `__init__.py` exports `Orchestrator` and `RunResult`; `__main__.py` defines the
`fm-mapb` Click CLI (`orchestrate`, `calibrate`, `evaluate`, `frontier`, `audit`). The package is laid
out as a MAPE-K autonomic loop — **monitor → analyze → plan → execute**, all reading and writing a shared
**knowledge** base.

### `knowledge` — the shared K
- `clinical_records.py` — case features, candidate regimens, and clinical data contracts.
- `posteriors.py` — posterior / uncertainty containers exchanged between phases.
- `audit_records.py` — record types for the evidence ledger.
- `settings_schema.py` — attrs-based configuration schema (run, calibration, backbone, kernel, compute,
  cohorts).
- `settings_loader.py` — loads and merges the layered TOML configs from `configs/` (`tomllib`) and
  resolves the default configs directory.
- `cohort_spec.py` — cohort specification types.
- `cohort_registry.py` — cohort registry, the longitudinal data stream, and the candidate regimens.
- `cohort_synthesis.py` — deterministic synthetic cohort generation matched to reported cohort sizes.
- `ledger.py` — the SHA-256-chained JSON-Lines evidence ledger and its verifier.

### `monitor` — sense and trigger (M)
- `adaptive.py` — longitudinal monitor that decides when to re-invoke the team.
- `triggers.py` — KL-divergence thresholds that fire re-invocation.

### `analyze` — interpret signals into beliefs (A)
- `primitives.py` — shared numeric plumbing for the reasoning units.
- `omics.py`, `immune.py`, `cardiotox.py` — role-specialised reasoning units (omics response, immune
  contexture, cardiotoxicity risk).
- `backbone_protocol.py` — the `typing.Protocol` every frozen foundation model implements.
- `backbone_ensemble.py` — combines per-modality backbones into a single embedding interface.
- `backbone_bulkrnabert.py`, `backbone_scgpt.py`, `backbone_geneformer.py`, `backbone_ecg.py` —
  deterministic seeded surrogates standing in for the named foundation models; no external weights or
  network access required.
- `metrics.py` — efficacy / safety and decision metrics.
- `stats.py` — bootstrap and statistical helpers.
- `findings.py` — structured findings emitted per experiment.

### `plan` — decide a regimen (P)
- `pareto.py` — Pareto frontier extraction and regimen recommendation.
- `hypervolume.py` — hypervolume indicator used to score the frontier.
- `conformal.py` — conformal prediction intervals.
- `conjugate.py` — conjugate posterior updates.
- `constrained.py` — safety-constrained selection.
- `guarantees.py` — coverage / safety guarantee checks.
- `mcid.py` — minimal clinically important difference gating.
- `planner.py` — the treatment-planning unit that assembles the candidate regimens.

### `execute` — act, dispatch tools, train, evaluate (E)
- `pipeline.py` — the `Orchestrator` entry point and `RunResult`.
- `controller.py` — coordinates the analyze / plan units with the decision kernel.
- `tool_dispatch.py` — tool-call dispatcher with retry and provenance.
- `tool_contraindication.py`, `tool_dgidb.py`, `tool_mcid.py`, `tool_pareto.py` — individual callable
  tools.
- `calibration_head.py` — the low-rank calibration head trained on top of frozen embeddings.
- `calibration_trainer.py` — calibration training loop and checkpoint I/O.
- `evaluation_runner.py` — drives an experiment end to end and aggregates results.
