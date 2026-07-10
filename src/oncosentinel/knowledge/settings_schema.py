from __future__ import annotations

from typing import Any

import attrs
from attrs import validators

ALL_AGENTS = ("omics", "immune", "cardio", "planner", "monitor")


@attrs.frozen
class PriorConfig:
    resp_mean: float = attrs.field(
        default=0.50, validator=validators.and_(validators.gt(0.0), validators.lt(1.0))
    )
    resp_var: float = attrs.field(default=0.04, validator=validators.gt(0.0))
    lvef_mu: float = attrs.field(default=0.15, validator=validators.gt(0.0))
    lvef_var: float = attrs.field(default=0.02, validator=validators.gt(0.0))


@attrs.frozen
class KernelConfig:
    tau: float = attrs.field(default=0.10, validator=validators.gt(0.0))
    tau_star: float = attrs.field(default=0.082, validator=validators.gt(0.0))
    alpha: float = attrs.field(
        default=0.10, validator=validators.and_(validators.gt(0.0), validators.lt(1.0))
    )
    coverage_delta: float = attrs.field(
        default=0.10, validator=validators.and_(validators.gt(0.0), validators.lt(1.0))
    )
    reference: tuple[float, float] = (0.0, 0.0)


@attrs.frozen
class McidConfig:
    orr_pp: float = 10.0
    pfs_hr: float = 0.75
    lvef_drop_pp: float = 10.0
    troponin_percentile: float = 99.0


@attrs.frozen
class BackboneConfig:
    weights: dict[str, float] = attrs.field(
        factory=lambda: {"BulkRNABert": 0.46, "scGPT_v2": 0.42, "Geneformer_v2": 0.12}
    )
    concentration: float = attrs.field(default=5.25, validator=validators.gt(0.0))


@attrs.frozen
class CalibrationConfig:
    optimizer: str = "adamw"
    lr: float = attrs.field(default=2e-4, validator=validators.gt(0.0))
    scheduler: str = "cosine"
    warmup: str = "linear"
    weight_decay: float = 0.01
    lora_rank: int = attrs.field(
        default=8, validator=validators.and_(validators.ge(4), validators.le(16))
    )
    batch_size: int = attrs.field(default=32, validator=validators.gt(0))
    steps: int = attrs.field(default=200, validator=validators.gt(0))
    warmup_steps: int = 20


@attrs.frozen
class ComputeConfig:
    world_size: int = 8
    device: str = "A100-80GB"
    backbone_precision: str = "bf16"
    kernel_precision: str = "fp32"
    gpu_hours: float = 187.0
    inference_seconds: float = 22.0


@attrs.frozen
class CohortConfig:
    name: str
    n_cases: int
    n_cancer_types: int
    modality: str
    license: str
    access_url: str
    seed: int = 0


@attrs.frozen
class RunConfig:
    experiment: str
    seeds: tuple[int, ...]
    bootstrap_iters: int
    baseline_orr: float
    priors: PriorConfig
    kernel: KernelConfig
    mcid: McidConfig
    backbone: BackboneConfig
    calibration: CalibrationConfig
    compute: ComputeConfig
    cohorts: tuple[CohortConfig, ...]
    enabled_agents: tuple[str, ...]
    tool_layer_enabled: bool

    @classmethod
    def from_merged(cls, data: dict[str, Any], cohorts: list[CohortConfig]) -> RunConfig:
        run = data.get("run", {})
        enabled = tuple(run.get("enabled_agents", ALL_AGENTS))
        unknown = set(enabled) - set(ALL_AGENTS)
        if unknown:
            raise ValueError(f"unknown agents in config: {sorted(unknown)}")
        kernel_raw = dict(data.get("kernel", {}))
        if "reference" in kernel_raw:
            kernel_raw["reference"] = tuple(float(v) for v in kernel_raw["reference"])
        return cls(
            experiment=str(run["experiment"]),
            seeds=tuple(int(s) for s in run.get("seeds", (0,))),
            bootstrap_iters=int(run.get("bootstrap_iters", 10000)),
            baseline_orr=float(run.get("baseline_orr", 0.20)),
            priors=PriorConfig(**data.get("priors", {})),
            kernel=KernelConfig(**kernel_raw),
            mcid=McidConfig(**data.get("mcid", {})),
            backbone=BackboneConfig(**data.get("backbone", {})),
            calibration=CalibrationConfig(**data.get("calibration", {})),
            compute=ComputeConfig(**data.get("compute", {})),
            cohorts=tuple(cohorts),
            enabled_agents=enabled,
            tool_layer_enabled=bool(run.get("tool_layer_enabled", True)),
        )
