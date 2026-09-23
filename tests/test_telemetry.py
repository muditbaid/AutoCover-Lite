import json

import pytest

from autocover.telemetry import Telemetry


def test_events_share_run_id_and_append_jsonl(tmp_path):
    path = tmp_path / "t.jsonl"
    tel = Telemetry(run_id="run1", jsonl_path=path)
    tel.event("preparer", "scenarios", count=3)
    with tel.span("executor", "run", candidate="c1") as extra:
        extra["passed"] = True
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["event"] for r in lines] == ["scenarios", "run"]
    assert all(r["run_id"] == "run1" for r in lines)
    assert lines[1]["passed"] is True and lines[1]["status"] == "ok"
    assert lines[1]["duration_ms"] >= 0


def test_span_records_errors_and_reraises():
    tel = Telemetry()
    with pytest.raises(RuntimeError), tel.span("fixer", "repair"):
        raise RuntimeError("boom")
    assert tel.events[-1]["status"] == "error:RuntimeError"


def test_no_path_keeps_events_in_memory_only():
    tel = Telemetry()
    tel.event("x", "y")
    assert tel.path is None and len(tel.events) == 1
