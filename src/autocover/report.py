"""Human-readable run report built from telemetry events (and the run summary, if saved).

Answers the operational questions: where did the time go, which models answered (and
how often the router fell back), how much sandbox work ran, and what happened to every
candidate on the way from generated to accepted.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def load_events(path: str | Path, run_id: str | None = None) -> tuple[str, list[dict]]:
    """Events of one run from a telemetry JSONL file (default: the last run)."""
    events = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if run_id is None:
        runs = [e["run_id"] for e in events if e.get("stage") == "run"]
        if not runs:
            raise ValueError(f"no completed run in {path}")
        run_id = runs[-1]
    return run_id, [e for e in events if e.get("run_id") == run_id]


def build_report(events: list[dict], summary: dict | None = None) -> str:
    out: list[str] = []
    run = next((e for e in reversed(events) if e.get("stage") == "run"), {})
    out.append(f"run {events[0]['run_id'] if events else '?'}: {run.get('target', '?')}, "
               f"{run.get('duration_ms', 0) / 1000:.1f}s, status {run.get('status', '?')}")
    if summary:
        base, final = summary.get("baseline", {}), summary.get("final", {})
        out.append(f"  lines {base.get('line_pct', 0)}% -> {final.get('line_pct', 0)}%, "
                   f"branches {base.get('branch_pct', 0)}% -> {final.get('branch_pct', 0)}%")
        if summary.get("mutation"):
            m = summary["mutation"]
            out.append(f"  mutation score {m['score_pct']}% ({m['killed']}/{m['total']})")
        out.append(f"  stopped by: {summary.get('stopped_by', '?')}; "
                   f"tests in suite: {summary.get('tests_in_suite', '?')}")

    # -- time by stage ----------------------------------------------------------------
    stage_time: dict[str, float] = defaultdict(float)
    for e in events:
        if "duration_ms" in e and e.get("stage") not in ("sandbox", "run"):
            stage_time[f"{e['stage']}.{e['event']}"] += e["duration_ms"] / 1000
    out.append("\ntime by stage")
    for key, secs in sorted(stage_time.items(), key=lambda kv: -kv[1]):
        out.append(f"  {key:<28} {secs:7.1f}s")

    # -- sandbox -------------------------------------------------------------------------
    runs = [e for e in events if e.get("stage") == "sandbox" and e.get("event") == "run"]
    if runs:
        batched = [e for e in runs if e.get("files", 1) > 1]
        mutated = [e for e in runs if e.get("mutated")]
        total = sum(e["duration_ms"] for e in runs) / 1000
        fallbacks = sum(1 for e in events if e.get("event") == "batch_fallback")
        out.append("\nsandbox")
        out.append(f"  {len(runs)} runs ({len(batched)} batched, {len(mutated)} with a mutant), "
                   f"{total:.0f}s container time, avg {total / len(runs):.1f}s; "
                   f"{fallbacks} batch fallbacks")

    # -- LLM ---------------------------------------------------------------------------
    calls = [e for e in events if e.get("stage") == "llm" and e.get("event") == "completion"]
    if calls:
        out.append("\nllm (role / model: calls, cached, avg fallback depth, avg latency, tokens)")
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for e in calls:
            groups[(e.get("role", "?"), e["model"])].append(e)
        for (role, model), items in sorted(groups.items()):
            live = [e for e in items if not e.get("cached")]
            latency = sum(e["latency_s"] for e in live) / len(live) if live else 0.0
            tokens = sum(e["prompt_tokens"] + e["completion_tokens"] for e in live)
            depth = sum(e.get("fallbacks", 0) for e in items) / len(items)
            out.append(f"  {role:<9} {model:<46} {len(items):>3}, {len(items) - len(live)} cached, "
                       f"{depth:.1f}, {latency:5.1f}s, {tokens}")
        skips = Counter(e["event"] for e in events if e.get("stage") == "llm"
                        and e.get("event") in ("skip_throttled", "skip_daily_cap",
                                               "daily_quota_exhausted", "model_failed", "retry"))
        if skips:
            out.append("  router: " + ", ".join(f"{k} x{v}" for k, v in sorted(skips.items())))

    # -- candidate funnel ----------------------------------------------------------------
    verdicts = [e for e in events
                if e.get("stage") == "validator" and e.get("event") == "candidate"]
    if verdicts:
        final_by_id: dict[str, dict] = {e["id"]: e for e in verdicts}
        generated = [e for e in final_by_id.values() if e.get("attempt", 0) == 0]
        repaired = [e for e in final_by_id.values() if e.get("attempt", 0) > 0]
        statuses = Counter(e["status"] for e in final_by_id.values())
        accepted_by = Counter(e.get("accepted_by") for e in final_by_id.values()
                              if e["status"] == "accepted")
        reasons = Counter(e["reason"].split(":")[0].split(" (")[0]
                          for e in final_by_id.values() if e.get("reason"))
        out.append("\ncandidate funnel")
        out.append(f"  generated {len(generated)}, repaired versions {len(repaired)}")
        out.append("  outcomes: " + ", ".join(f"{k} {v}" for k, v in statuses.most_common()))
        if accepted_by:
            out.append("  accepted by: " + ", ".join(f"{k} {v}" for k, v in accepted_by.items()))
        if reasons:
            out.append("  reasons seen: " +
                       ", ".join(f"{k} x{v}" for k, v in reasons.most_common()))
        models = Counter(e.get("model") for e in final_by_id.values() if e["status"] == "accepted")
        if models:
            out.append("  accepted tests by model: " +
                       ", ".join(f"{k} {v}" for k, v in models.most_common()))
    return "\n".join(out)
