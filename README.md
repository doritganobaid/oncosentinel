# oncosentinel

*Reference implementation accompanying a manuscript submitted to* **npj Digital Medicine**.
This repository (`oncosentinel`) ships the `oncosentinel` package and the `fm-mapb` command-line tool.

---

## Abstract

Immune-checkpoint inhibitors (ICIs) can deliver durable responses yet carry a genuine risk of
immune-related cardiotoxicity, so selecting a regimen for an individual patient is a multi-objective
decision under uncertainty. **oncosentinel** couples frozen biomedical foundation models with a team of
specialised reasoning units and a Bayesian decision kernel that recommends regimens on the
efficacy–safety Pareto frontier, while keeping a verifiable, hash-chained audit trail of every decision.
The system is implemented as a MAPE-K autonomic loop and runs end to end offline on deterministic
synthetic cohorts that mirror the sizes of the public reference cohorts; no external weights or network
access are required to reproduce its behaviour.

## Background

Choosing an ICI regimen means trading expected efficacy against cardiac safety for a specific patient.
We frame this as a constrained, multi-objective decision problem and produce a recommendation on the
efficacy–safety Pareto frontier, together with an auditable record of how the recommendation was reached.
Rather than organising the code by software role, the implementation mirrors the control loop of an
autonomic system — **M**onitor, **A**nalyze, **P**lan, **E**xecute, over a shared **K**nowledge base.

## Methods

**Foundation-model backbones.** Frozen models (BulkRNABert, scGPT-v2, Geneformer-v2, ECG-FM) are reached
only through a single `typing.Protocol` in the analyze phase. The shipped implementations are
deterministic seeded surrogates, so the full pipeline is reproducible offline; real weights drop in
behind the same interface.

**Reasoning units (analyze).** Role-specialised units interpret omics response, immune contexture, and
cardiotoxicity risk, emitting calibrated posteriors that are combined into joint efficacy/safety beliefs.

**Pareto–Bayesian decision kernel (plan).** The decision math is isolated in `plan/`: Pareto frontier
extraction, hypervolume scoring, conformal prediction intervals, conjugate posterior updates,
safety-constrained selection, coverage/safety guarantees, and minimal-clinically-important-difference
(MCID) gating.

**Adaptive re-invocation (monitor).** A KL-divergence trigger re-runs the team when longitudinal signals
shift enough to matter, instead of on a fixed schedule.

**Tooling and audit (execute, knowledge).** A BFCL-style tool dispatcher (contraindication checks, DGIdb
lookups, MCID, Pareto frontier) adds retry and provenance; every decision step is appended to a
SHA-256-chained JSON-Lines evidence ledger that can be re-verified after the fact.

## Implementation

The `oncosentinel` package is organised as a MAPE-K loop (see `docs/implementation-map.md` for the
module-by-module map):

| Phase | Subpackage | Responsibility |
| --- | --- | --- |
| Knowledge (K) | `knowledge/` | clinical/posterior/audit records, attrs settings + TOML loader, cohorts, evidence ledger |
| Monitor (M) | `monitor/` | longitudinal sensing and KL-divergence re-invocation triggers |
| Analyze (A) | `analyze/` | backbone protocol + surrogates, reasoning units, metrics/stats/findings |
| Plan (P) | `plan/` | Pareto / hypervolume / conformal / conjugate / constrained / guarantees / MCID / planner |
| Execute (E) | `execute/` | orchestrator + controller, tool dispatch, calibration training, evaluation runner |

Configuration is declarative: layered TOML configs read with `tomllib`, validated by attrs schemas, and
driven through a thin Click CLI. Python ≥ 3.11 is required.

```bash
pip install -e ".[dev]"                       # development install
# or
conda env create -f environment.yml && conda activate fm-mapb && pip install -e . --no-deps
# or
docker build -t fm-mapb . && docker run --rm fm-mapb --help
```

## Reproducibility

Only the lightweight calibration head is trained; the foundation-model backbones stay frozen. All
randomness is seeded, experiments declare their seeds in `configs/experiment/*.toml`, and the quality
gates (`ruff`, `isort`, `black`, `mypy --strict`, `pytest`) run in CI on every push. The `_smoke`
experiment and the CI configuration run CPU-only.

```bash
fm-mapb calibrate --experiment main --seed 0 --output runs/main_seed0_head.pt   # train calibration head
fm-mapb evaluate  --experiment main --limit 256                                 # aggregate metrics
fm-mapb orchestrate --experiment main --case 0                                  # one recommendation + evidence
fm-mapb frontier  --points "0.6,0.2;0.5,0.4;0.7,0.1"                            # Pareto / hypervolume selection
fm-mapb audit     --path runs/evidence.jsonl                                    # verify a hash-chained ledger
```

Calibration settings (`configs/train/calibration.toml`): AdamW, learning rate 2e-4, cosine schedule with
20-step linear warmup, weight decay 0.01, LoRA rank 8, batch size 32, 200 steps. Experiment configs in
`configs/experiment/` cover the full system (`main`), ablations
(`ablation_drop_{omics,immune,cardio,planner,monitor}`, `ablation_no_tool`, `ablation_tau_sweep`), and
supplementary analyses (`supplementary_loco_cv`, `supplementary_mimic_excluded`,
`supplementary_reference_sweep`).

**Compute budget** (`configs/model/compute.toml`): the reported full-scale setting uses 8× A100-80GB,
bf16 backbones with an fp32 decision kernel, world size 8, **187 GPU-hours** total, and ~22 s end-to-end
inference per case.

## Data availability

Evaluation uses public retrospective cohorts. Restricted cohorts require their own access agreements and
are **not** redistributed here; deterministic synthetic fixtures matched to the reported cohort sizes are
generated on demand by `oncosentinel.knowledge.cohort_synthesis`, so no download is needed to run the
pipeline (`bash scripts/prepare_data.sh` prints access points only).

| Cohort | Cases | Modality | License | Access |
| --- | --- | --- | --- | --- |
| TCGA Pan-Cancer | 4218 | multi-omics | open | portal.gdc.cancer.gov |
| IMvigor210 | 298 | omics + clinical | open | ESMO Open supplement |
| OAK | 850 | omics + clinical | open | published supplement |
| UK Biobank Cardiac MRI | 4612 | cardiac MRI | restricted | ukbiobank.ac.uk |
| CARDIOTOX / CARDIOCARE | 421 | multi-vendor echo | CC-BY-NC | zenodo.org/records/19727929 |
| MIMIC-IV-ECG | 2741 | ECG | restricted | physionet.org |

## Code availability

The complete decision pipeline is contained in this repository and is fully runnable via the
deterministic synthetic cohorts and surrogate backbones. Calibration checkpoints will be released upon
acceptance.

## Ethics

`fm-mapb` is a decision-support tool, not an autonomous treatment system; every recommendation is
accompanied by an auditable evidence chain intended for clinician review. All development and reported
evaluation use retrospective, de-identified public cohorts under their respective licenses.
Restricted-access cohorts (UK Biobank, MIMIC-IV-ECG) are governed by separate data-use agreements and
are not redistributed in this repository. No protected health information is included here.

## Citation

The manuscript is under review at *npj Digital Medicine*. Until publication, please cite the repository:

```bibtex
@software{oncosentinel,
  title  = {oncosentinel: a foundation-model multi-agent Pareto--Bayesian orchestrator for joint ICI efficacy and cardiotoxicity decision support},
  year   = {2026},
  note   = {npj Digital Medicine submission}
}
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
