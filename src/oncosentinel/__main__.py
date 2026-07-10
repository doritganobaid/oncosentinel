from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import torch

from oncosentinel.execute.calibration_head import CalibrationHead
from oncosentinel.execute.calibration_trainer import save_checkpoint, train_calibration
from oncosentinel.execute.evaluation_runner import evaluate_experiment
from oncosentinel.execute.pipeline import Orchestrator
from oncosentinel.knowledge.cohort_registry import CANDIDATE_REGIMENS, longitudinal_stream
from oncosentinel.knowledge.cohort_synthesis import generate_cohort
from oncosentinel.knowledge.ledger import EvidenceChain
from oncosentinel.knowledge.settings_loader import default_configs_dir, load_config
from oncosentinel.plan.pareto import recommend


def _configs(path: Path | None) -> Path:
    return path if path is not None else default_configs_dir()


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--experiment", default="main", show_default=True)
@click.option("--case", "case_index", default=0, show_default=True)
@click.option("--configs", "configs_dir", type=Path, default=None)
@click.option("--output", type=Path, default=None)
@click.option("--monitor-steps", default=4, show_default=True)
def orchestrate(
    experiment: str,
    case_index: int,
    configs_dir: Path | None,
    output: Path | None,
    monitor_steps: int,
) -> None:
    config = load_config(experiment, _configs(configs_dir))
    orchestrator = Orchestrator.from_config(config)
    cohort = config.cohorts[0]
    cases = generate_cohort(cohort, limit=case_index + 1)
    labeled = cases[case_index]
    stream = longitudinal_stream(labeled.features.case_id, monitor_steps, seed=case_index)
    result = orchestrator.run(labeled.features, list(CANDIDATE_REGIMENS), stream)
    recommendation = result.recommendation
    click.echo(
        json.dumps(
            {
                "experiment": experiment,
                "case": labeled.features.case_id,
                "regimen": recommendation.regimen,
                "efficacy": round(recommendation.efficacy, 4),
                "safety": round(recommendation.safety, 4),
                "hv_argmax": round(recommendation.hv_argmax, 6),
                "frontier_size": recommendation.frontier_size,
                "mcid_crossing": recommendation.mcid_crossing,
                "reinvocations": result.reinvocations,
                "evidence_records": len(result.evidence.records),
                "evidence_verified": result.evidence.verify(),
            },
            indent=2,
        )
    )
    if output is not None:
        result.evidence.write(output)
        click.echo(f"evidence written to {output}")


@cli.command()
@click.option("--experiment", default="main", show_default=True)
@click.option("--configs", "configs_dir", type=Path, default=None)
@click.option("--limit", default=128, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--output", type=Path, default=None)
def calibrate(
    experiment: str, configs_dir: Path | None, limit: int, seed: int, output: Path | None
) -> None:
    config = load_config(experiment, _configs(configs_dir))
    cohort = config.cohorts[0]
    cases = generate_cohort(cohort, limit=limit)
    features = torch.tensor(np.stack([case.features.omics for case in cases]), dtype=torch.float32)
    labels = torch.tensor(
        np.asarray([case.response_label for case in cases], dtype=np.float32), dtype=torch.float32
    )
    head = CalibrationHead(in_features=features.shape[1], rank=config.calibration.lora_rank)
    history = train_calibration(head, features, labels, config.calibration, seed=seed)
    click.echo(
        json.dumps(
            {
                "experiment": experiment,
                "cohort": cohort.name,
                "n": len(cases),
                "first_loss": round(history[0], 5),
                "last_loss": round(history[-1], 5),
                "steps": len(history),
            },
            indent=2,
        )
    )
    if output is not None:
        save_checkpoint(head, seed, output)
        click.echo(f"checkpoint written to {output}")


@cli.command()
@click.option("--experiment", default="main", show_default=True)
@click.option("--configs", "configs_dir", type=Path, default=None)
@click.option("--limit", default=64, show_default=True)
def evaluate(experiment: str, configs_dir: Path | None, limit: int) -> None:
    config = load_config(experiment, _configs(configs_dir))
    results = evaluate_experiment(config, limit=limit)
    click.echo(json.dumps(results, indent=2))


@cli.command()
@click.option("--points", required=True, help="semicolon-separated efficacy,safety pairs")
@click.option("--reference", default="0,0", show_default=True)
def frontier(points: str, reference: str) -> None:
    parsed = [[float(v) for v in pair.split(",")] for pair in points.split(";") if pair]
    ref_x, ref_y = (float(v) for v in reference.split(","))
    selection = recommend(np.asarray(parsed, dtype=np.float64), (ref_x, ref_y))
    click.echo(
        json.dumps(
            {
                "argmax_index": selection.argmax_index,
                "frontier_size": selection.frontier_size,
                "hv_total": round(selection.hv_total, 6),
                "hv_argmax": round(selection.hv_argmax, 6),
                "front_indices": selection.front_indices.tolist(),
            },
            indent=2,
        )
    )


@cli.command()
@click.option("--path", type=Path, required=True)
def audit(path: Path) -> None:
    chain = EvidenceChain.load(path)
    verified = chain.verify()
    click.echo(
        json.dumps(
            {"path": str(path), "records": len(chain.records), "verified": verified}, indent=2
        )
    )


if __name__ == "__main__":
    cli()
