from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import attrs

from oncosentinel.knowledge.audit_records import ToolReturn

ToolFn = Callable[[dict[str, Any]], ToolReturn]


class ToolFailure(RuntimeError):
    pass


def provenance_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_coerce)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _coerce(value: Any) -> Any:
    if isinstance(value, set | frozenset):
        return sorted(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


@attrs.define
class CallOutcome:
    agent: str
    tool: str
    inputs: dict[str, Any]
    result: ToolReturn
    attempts: int
    provenance: str


@attrs.define
class Dispatcher:
    registry: dict[str, ToolFn] = attrs.field(factory=dict)
    max_retries: int = 1

    def register(self, name: str, fn: ToolFn) -> None:
        self.registry[name] = fn

    def call(self, agent: str, tool: str, inputs: dict[str, Any]) -> CallOutcome:
        if tool not in self.registry:
            raise KeyError(f"unregistered tool: {tool}")
        attempts = 0
        last_error: ToolFailure | None = None
        while attempts <= self.max_retries:
            attempts += 1
            try:
                result = self.registry[tool](inputs)
                provenance = provenance_hash(
                    {"agent": agent, "tool": tool, "inputs": inputs, "output": result.payload}
                )
                return CallOutcome(
                    agent=agent,
                    tool=tool,
                    inputs=inputs,
                    result=result,
                    attempts=attempts,
                    provenance=provenance,
                )
            except ToolFailure as error:
                last_error = error
        raise ToolFailure(f"{tool} failed after {attempts} attempts: {last_error}")
