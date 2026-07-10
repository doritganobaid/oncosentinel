from __future__ import annotations

import pytest

from oncosentinel.execute.tool_contraindication import check_contraindication
from oncosentinel.execute.tool_dgidb import query_dgidb
from oncosentinel.execute.tool_dispatch import Dispatcher, ToolFailure, provenance_hash
from oncosentinel.execute.tool_mcid import lookup_mcid
from oncosentinel.execute.tool_pareto import compute_pareto_frontier
from oncosentinel.knowledge.audit_records import ToolReturn


def test_dgidb_payload_schema() -> None:
    out = query_dgidb({"genes": ["PDCD1"], "drugs": ["pembrolizumab"]})
    assert isinstance(out.payload["interaction_count"], int)
    assert out.payload["severity"] in {"none", "minor", "moderate", "major"}
    assert out.version.startswith("DGIdb")


def test_mcid_unknown_metric_raises() -> None:
    assert lookup_mcid({"metric": "ORR", "regimen": "x"}).payload["mcid"] == 0.10
    with pytest.raises(KeyError):
        lookup_mcid({"metric": "UNKNOWN", "regimen": "x"})


def test_contraindication_rules() -> None:
    blocked = check_contraindication({"interaction_severity": "major", "is_combination": False})
    assert blocked.payload["contraindicated"]
    allowed = check_contraindication(
        {"interaction_severity": "minor", "is_combination": False, "hfa_icos": 0}
    )
    assert not allowed.payload["contraindicated"]


def test_provenance_hash_is_stable_and_sensitive() -> None:
    a = provenance_hash({"tool": "x", "inputs": {"k": 1}})
    b = provenance_hash({"tool": "x", "inputs": {"k": 1}})
    c = provenance_hash({"tool": "x", "inputs": {"k": 2}})
    assert a == b
    assert a != c


def test_dispatcher_retry_then_success() -> None:
    state = {"calls": 0}

    def flaky(_: dict[str, object]) -> ToolReturn:
        state["calls"] += 1
        if state["calls"] < 2:
            raise ToolFailure("transient")
        return ToolReturn(payload={"ok": True}, version="test")

    dispatcher = Dispatcher(max_retries=1)
    dispatcher.register("flaky", flaky)
    outcome = dispatcher.call("agent", "flaky", {})
    assert outcome.attempts == 2
    assert outcome.result.payload["ok"] is True


def test_dispatcher_exhausts_retries() -> None:
    def always_fail(_: dict[str, object]) -> ToolReturn:
        raise ToolFailure("down")

    dispatcher = Dispatcher(max_retries=1)
    dispatcher.register("bad", always_fail)
    with pytest.raises(ToolFailure):
        dispatcher.call("agent", "bad", {})


def test_pareto_frontier_tool_payload() -> None:
    out = compute_pareto_frontier(
        {"points": [[0.3, 0.6], [0.5, 0.4], [0.05, 0.05]], "reference": [0.0, 0.0]}
    )
    assert out.payload["frontier_size"] == 2
    assert isinstance(out.payload["argmax_index"], int)
