# Project Context

## Problem

Immune-checkpoint inhibitors can produce durable responses but also carry a real risk of immune-related
cardiotoxicity. Choosing a regimen therefore means trading off expected efficacy against cardiac safety
for an individual patient, under genuine uncertainty. `fm-mapb` frames this as a constrained,
multi-objective decision problem and produces a recommendation on the efficacy–safety Pareto frontier,
together with an auditable record of how that recommendation was reached.

## Scope

The repository implements the full decision pipeline as a MAPE-K autonomic loop — monitoring,
analysis (frozen-backbone interface and reasoning units), planning (the Bayesian decision kernel),
and execution (tool layer, orchestration, calibration, evaluation) over a shared knowledge base — and
runs end to end on deterministic synthetic cohorts that mirror the sizes of the public reference
cohorts. It does **not** ship trained weights or restricted patient data. The intent is reproducibility
of the method and its behaviour, not distribution of clinical models.

## Design choices

- **MAPE-K autonomic layout:** the package is organised as a monitor → analyze → plan → execute loop
  over a shared `knowledge` base, rather than by software role. Each phase is a thin subpackage, which
  keeps the feedback loop — sense, interpret, decide, act — legible in the directory tree itself.
- **Configuration / CLI stack:** layered TOML configs read with `tomllib`, attrs schemas, and a Click
  CLI. This keeps configuration declarative and the entry points thin.
- **Frozen foundation models behind a Protocol:** backbones are accessed only through a single
  `typing.Protocol` in the analyze phase. The shipped implementations are deterministic seeded
  surrogates, so the pipeline is fully runnable offline and every run is reproducible; real weights can
  be dropped in behind the same interface.
- **Decision math isolated in the plan phase:** the kernel (Pareto, hypervolume, conformal, conjugate
  updates, constrained optimisation, MCID) lives in `plan/`, independent of how signals are interpreted
  (`analyze/`) or how recommendations are enacted (`execute/`).
- **Adaptive re-invocation:** a KL-divergence trigger in the monitor phase re-runs the team when
  longitudinal signals shift enough to matter, rather than on a fixed schedule.
- **Verifiable evidence ledger:** every decision step is appended to a SHA-256-chained JSON-Lines log
  (`knowledge/ledger.py`) that can be verified after the fact with `fm-mapb audit`.

## Reproducibility

- Pinned dependencies in `pyproject.toml` / `requirements.txt` / `environment.yml`; Python ≥ 3.11.
- All randomness is seeded; experiments declare their seeds in `configs/experiment/*.toml`.
- Quality gates (ruff, isort, black, mypy --strict, pytest) run in CI on every push and pull request.
- The `_smoke` experiment and the CI configuration run CPU-only for fast verification.
