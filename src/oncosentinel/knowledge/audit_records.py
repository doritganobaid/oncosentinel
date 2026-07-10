from __future__ import annotations

from typing import Any

import attrs


@attrs.frozen
class ToolCall:
    agent: str
    tool: str
    inputs: dict[str, Any]


@attrs.frozen
class ToolReturn:
    payload: dict[str, Any]
    version: str


@attrs.frozen
class EvidenceRecord:
    t: int
    agent: str
    action: str
    inputs: dict[str, Any]
    output: dict[str, Any]
    provenance_hash: str
    prev_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "agent": self.agent,
            "action": self.action,
            "inputs": self.inputs,
            "output": self.output,
            "provenance_hash": self.provenance_hash,
            "prev_hash": self.prev_hash,
        }
