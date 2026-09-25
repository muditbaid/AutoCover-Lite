"""Time-budget policy: finalize reserve, per-role LLM deadlines, fix-cycle gating."""

import time

import pytest

from autocover.budget import finalize_reserve_s, llm_deadline, typical_run_s
from autocover.config import Config, MutationConfig, RunConfig, SandboxConfig, TelemetryConfig
from autocover.graph import stop_reason
from autocover.state import RunContext
from autocover.telemetry import Telemetry
from autocover.tools.context import build_module_context
from tests.test_graph_e2e import EXAMPLE


def make_ctx(*, mutation=True, budget_min=15.0, max_parallel=6) -> RunContext:
    config = Config(mutation=MutationConfig(enabled=mutation),
                    run=RunConfig(budget_min=budget_min, flaky_reruns=1),
                    sandbox=SandboxConfig(max_parallel=max_parallel),
                    telemetry=TelemetryConfig(jsonl_path=None))
    return RunContext(config=config, repo=EXAMPLE, target="ticket_price.py", test_path="t.py",
                      router=None, sandbox=None, telemetry=Telemetry(),
                      module=build_module_context(EXAMPLE, "ticket_price.py"),
                      deadline=time.monotonic() + budget_min * 60)


def sandbox_run(ctx: RunContext, seconds: float, mutated: bool) -> None:
    ctx.telemetry.event("sandbox", "run", mutated=mutated, duration_ms=seconds * 1000)


def test_typical_run_prefers_mutant_runs_and_falls_back_to_any_run():
    ctx = make_ctx()
    assert typical_run_s(ctx) == 5.0  # nothing measured yet
    sandbox_run(ctx, 2.0, mutated=False)
    assert typical_run_s(ctx) == 2.0
    for s in (4.0, 4.0, 4.0, 8.0):
        sandbox_run(ctx, s, mutated=True)
    assert typical_run_s(ctx) == 8.0  # 75th percentile of the mutant runs only


def test_finalize_reserve_grows_with_mutants_not_yet_killed():
    ctx = make_ctx(budget_min=60, max_parallel=2)
    sandbox_run(ctx, 10.0, mutated=True)
    pool = ctx.mutants()
    assert len(pool) >= 4
    # suite run + 1 flaky rerun + ceil(unkilled / 2) waves, 10s each, x1.25 safety
    waves = 2 + -(-len(pool) // 2)
    assert finalize_reserve_s(ctx) == pytest.approx(waves * 10 * 1.25)
    ctx.killed_mutants |= {m.id for m in pool}
    assert finalize_reserve_s(ctx) == pytest.approx(max(45, 2 * 10 * 1.25))


def test_small_runs_keep_the_floor_capped_at_ten_percent_of_the_budget():
    ctx = make_ctx(mutation=False, budget_min=1)
    sandbox_run(ctx, 0.5, mutated=False)
    assert finalize_reserve_s(ctx) == pytest.approx(6.0)  # min(45s, 10% of 60s)


def test_generator_deadline_also_leaves_time_to_check_its_tests():
    ctx = make_ctx(mutation=False, budget_min=15)
    before = llm_deadline(ctx, "judge") - llm_deadline(ctx, "generator")
    assert before == pytest.approx(90.0, abs=0.01)  # nothing measured: 10% of 15 min
    ctx.check_s = 140.0  # measured: the longest execute + mutation check so far
    judge, generator = llm_deadline(ctx, "judge"), llm_deadline(ctx, "generator")
    assert judge == pytest.approx(ctx.deadline - finalize_reserve_s(ctx), abs=0.01)
    assert judge - generator == pytest.approx(140.0, abs=0.01)


def test_no_new_round_when_it_would_eat_the_finalize_reserve():
    ctx = make_ctx(budget_min=10)
    sandbox_run(ctx, 10.0, mutated=True)
    state = {"targets": ["ticket_price"], "round": 1}
    ctx.deadline = time.monotonic() + finalize_reserve_s(ctx) + 100
    ctx.last_round_s = 60
    assert stop_reason(ctx, state) is None
    ctx.last_round_s = 120  # one more round like the last would cut into finalize
    assert stop_reason(ctx, state) == "time budget"
