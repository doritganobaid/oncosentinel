from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import attrs

from oncosentinel.execute.tool_dispatch import provenance_hash
from oncosentinel.knowledge.audit_records import EvidenceRecord

GENESIS = "sha256:genesis"


@attrs.define
class EvidenceChain:
    records: list[EvidenceRecord] = attrs.field(factory=list)

    def append(
        self, agent: str, action: str, inputs: dict[str, Any], output: dict[str, Any]
    ) -> EvidenceRecord:
        prev = self.records[-1].provenance_hash if self.records else GENESIS
        content = {
            "t": len(self.records),
            "agent": agent,
            "action": action,
            "inputs": inputs,
            "output": output,
            "prev_hash": prev,
        }
        record = EvidenceRecord(
            t=len(self.records),
            agent=agent,
            action=action,
            inputs=inputs,
            output=output,
            provenance_hash=provenance_hash(content),
            prev_hash=prev,
        )
        self.records.append(record)
        return record

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(record.as_dict(), sort_keys=True) for record in self.records)

    def verify(self) -> bool:
        prev = GENESIS
        for index, record in enumerate(self.records):
            if record.t != index or record.prev_hash != prev:
                return False
            content = {
                "t": record.t,
                "agent": record.agent,
                "action": record.action,
                "inputs": record.inputs,
                "output": record.output,
                "prev_hash": record.prev_hash,
            }
            if provenance_hash(content) != record.provenance_hash:
                return False
            prev = record.provenance_hash
        return True

    def write(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_jsonl() + "\n", encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> EvidenceChain:
        records: list[EvidenceRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            records.append(
                EvidenceRecord(
                    t=int(data["t"]),
                    agent=str(data["agent"]),
                    action=str(data["action"]),
                    inputs=dict(data["inputs"]),
                    output=dict(data["output"]),
                    provenance_hash=str(data["provenance_hash"]),
                    prev_hash=str(data["prev_hash"]),
                )
            )
        return cls(records=records)
