"""Time-budget policy: how much of the run's deadline each stage may use.

The run ends with `finalize` (whole-suite check, flaky rerun, final mutation score), which
is sandbox work whose cost grows with the mutant pool. Everything before it must leave
that much time, so:

* a generation round starts only if one more round (as long as the last) still fits;
* a fix cycle starts only if one more fix cycle (as long as the last) still fits;
* every LLM call gets a deadline (`llm_deadline`) that the router never waits past, so a
  queue on a slow, last-resort model cannot eat the time the sandbox work needs.
"""

from __future__ import annotations

import math

from autocover.state import RunContext

FINALIZE_FLOOR_S = 45    # minimum finalize reserve (cap: 10% of the budget)
FINALIZE_SAFETY = 1.25   # past runs: estimate x 1.25 covered the real finalize time
DEFAULT_RUN_S = 5.0      # assumed sandbox run time before any run was measured
CHECK_PRIOR_SHARE = 0.1  # assumed execute + mutation check before one was measured


def typical_run_s(ctx: RunContext) -> float:
    """75th percentile of this run's mutant sandbox runs (the closest match to the final
    score's whole-suite mutant runs), or of all its sandbox runs before any mutant ran."""
    runs = [e for e in ctx.telemetry.events[ctx.telemetry_start:]
            if e.get("stage") == "sandbox" and e.get("event") == "run" and "duration_ms" in e]
    durations = sorted(e["duration_ms"] / 1000
                       for e in ([e for e in runs if e.get("mutated")] or runs))
    return durations[int(len(durations) * 0.75)] if durations else DEFAULT_RUN_S


def finalize_reserve_s(ctx: RunContext) -> float:
    """Expected time of the finalize step: suite run + flaky reruns + one whole-suite run
    per mutant not yet killed by an accepted test, `max_parallel` at a time."""
    cfg = ctx.config
    unknown = 0
    if cfg.mutation.enabled and cfg.mutation.final_score:
        unknown = sum(m.id not in ctx.killed_mutants for m in ctx.mutants())
    waves = 1 + cfg.run.flaky_reruns + math.ceil(unknown / max(1, cfg.sandbox.max_parallel))
    floor = min(FINALIZE_FLOOR_S, cfg.run.budget_min * 60 * 0.1)
    return max(floor, waves * typical_run_s(ctx) * FINALIZE_SAFETY)


def llm_deadline(ctx: RunContext, role: str) -> float:
    """Monotonic time after which the router must not start or wait for a call.

    Judge calls end just before finalize. Generator and Fixer output still has to be
    executed and mutation-checked, so they also leave the longest such check so far
    (before the first one is measured: 10% of the budget)."""
    deadline = ctx.deadline - finalize_reserve_s(ctx)
    if role != "judge":
        deadline -= ctx.check_s or ctx.config.run.budget_min * 60 * CHECK_PRIOR_SHARE
    return deadline
