import asyncio

import pytest

from autocover.config import CircuitBreakerConfig, LLMConfig, ModelLimits, ProviderLimits
from autocover.llm.cache import ResponseCache, UsageLedger
from autocover.llm.router import AllModelsFailed, LLMRouter, TokenBucket, is_retryable
from autocover.telemetry import Telemetry

MESSAGES = [{"role": "user", "content": "hi"}]


class ProviderError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeCompletion:
    """Scripted completion fn: per-model list of outcomes (str reply or Exception)."""

    def __init__(self, script: dict[str, list]):
        self.script = {m: list(v) for m, v in script.items()}
        self.calls: list[str] = []

    async def __call__(self, *, model, messages, **kwargs):
        self.calls.append(model)
        outcome = self.script[model].pop(0) if self.script[model] else "default"
        if isinstance(outcome, Exception):
            raise outcome
        return {"choices": [{"message": {"content": outcome}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5}}


def make_router(script, *, chain=("a/one", "b/two"), cache=None, retries=2, threshold=3):
    clock = FakeClock()
    cfg = LLMConfig(
        roles={"gen": list(chain)},
        providers={"a": ProviderLimits(rpm=600, max_concurrency=2),
                   "b": ProviderLimits(rpm=600, max_concurrency=2)},
        retries=retries,
        circuit_breaker=CircuitBreakerConfig(failure_threshold=threshold, cooldown_s=60),
    )
    fake = FakeCompletion(script)
    router = LLMRouter(cfg, cache=cache, telemetry=Telemetry(), completion_fn=fake,
                       require_keys=False, clock=clock, sleep=clock.sleep)
    return router, fake, clock


def run(coro):
    return asyncio.run(coro)


def test_first_model_success():
    router, fake, _ = make_router({"a/one": ["hello"], "b/two": []})
    resp = run(router.complete("gen", MESSAGES))
    assert (resp.text, resp.model, resp.fallbacks) == ("hello", "a/one", 0)
    assert (resp.prompt_tokens, resp.completion_tokens) == (3, 5)
    assert fake.calls == ["a/one"]


def test_retries_rate_limit_then_succeeds_with_backoff():
    router, fake, clock = make_router({"a/one": [ProviderError(429), "ok"], "b/two": []})
    resp = run(router.complete("gen", MESSAGES))
    assert resp.text == "ok" and resp.model == "a/one"
    assert fake.calls == ["a/one", "a/one"]
    assert clock.now > 0  # slept for backoff


def test_exhausted_retries_fall_back_to_next_model():
    errors = [ProviderError(503)] * 3
    router, fake, _ = make_router({"a/one": errors, "b/two": ["from b"]})
    resp = run(router.complete("gen", MESSAGES))
    assert resp.model == "b/two" and resp.fallbacks == 1
    assert fake.calls == ["a/one"] * 3 + ["b/two"]


def test_non_retryable_error_skips_retries():
    router, fake, _ = make_router({"a/one": [ProviderError(401)], "b/two": ["b"]})
    resp = run(router.complete("gen", MESSAGES))
    assert resp.model == "b/two"
    assert fake.calls == ["a/one", "b/two"]


def test_circuit_breaker_opens_then_half_opens_after_cooldown():
    script = {"a/one": [ProviderError(400)] * 2 + ["recovered"], "b/two": ["b"] * 5}
    router, fake, clock = make_router(script, threshold=2, retries=0)
    run(router.complete("gen", MESSAGES))
    run(router.complete("gen", MESSAGES))
    assert router.breaker.is_open("a/one")
    fake.calls.clear()
    run(router.complete("gen", MESSAGES))
    assert fake.calls == ["b/two"]  # open breaker: a/one not even tried
    clock.now += 61
    resp = run(router.complete("gen", MESSAGES))
    assert resp.model == "a/one" and resp.text == "recovered"


def test_all_models_failed_lists_every_reason():
    router, _, _ = make_router({"a/one": [ProviderError(400)], "b/two": [ProviderError(400)]})
    with pytest.raises(AllModelsFailed) as info:
        run(router.complete("gen", MESSAGES))
    assert "a/one" in str(info.value) and "b/two" in str(info.value)


def test_cache_hit_avoids_provider_call(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    router, fake, _ = make_router({"a/one": ["first", "second"], "b/two": []}, cache=cache)
    first = run(router.complete("gen", MESSAGES))
    again = run(router.complete("gen", MESSAGES))
    assert again.cached and again.text == first.text == "first"
    assert fake.calls == ["a/one"]
    fresh = run(router.complete("gen", MESSAGES, use_cache=False))
    assert fresh.text == "second" and not fresh.cached


def test_missing_api_key_skips_model(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "x")
    cfg = LLMConfig(roles={"gen": ["gemini/flash", "groq/fast"]})
    fake = FakeCompletion({"gemini/flash": ["no"], "groq/fast": ["yes"]})
    router = LLMRouter(cfg, completion_fn=fake)
    resp = run(router.complete("gen", MESSAGES))
    assert resp.model == "groq/fast" and fake.calls == ["groq/fast"]


def test_unknown_role_raises():
    router, _, _ = make_router({"a/one": [], "b/two": []})
    with pytest.raises(KeyError):
        run(router.complete("nope", MESSAGES))


def test_token_bucket_spaces_requests():
    clock = FakeClock()
    bucket = TokenBucket(per_minute=60, capacity=1, clock=clock, sleep=clock.sleep)

    async def take(n):
        for _ in range(n):
            await bucket.acquire()

    run(take(3))
    assert clock.now == pytest.approx(2.0)  # 1 immediate + 2 more at 1 req/s


def test_is_retryable():
    assert is_retryable(ProviderError(429))
    assert is_retryable(TimeoutError())
    assert not is_retryable(ProviderError(400))
    assert not is_retryable(ValueError("bad"))


# -- per-model limits -------------------------------------------------------------

def make_limited_router(models, *, usage=None, max_wait=15, day="2026-09-23"):
    clock = FakeClock()
    cfg = LLMConfig(
        roles={"gen": ["a/one", "b/two"]},
        providers={"a": ProviderLimits(rpm=None, max_concurrency=4),
                   "b": ProviderLimits(rpm=None, max_concurrency=4)},
        models=models, max_queue_wait_s=max_wait, retries=0,
    )
    fake = FakeCompletion({"a/one": ["a"] * 20, "b/two": ["b"] * 20})
    router = LLMRouter(cfg, usage=usage, completion_fn=fake, require_keys=False,
                       clock=clock, sleep=clock.sleep, today=lambda _tz: day)
    return router, fake, clock


def test_daily_cap_skips_model_and_persists(tmp_path):
    ledger = UsageLedger(tmp_path / "u.sqlite")
    router, fake, _ = make_limited_router({"a/one": ModelLimits(rpd=2)}, usage=ledger)
    models = [run(router.complete("gen", MESSAGES, use_cache=False)).model for _ in range(3)]
    assert models == ["a/one", "a/one", "b/two"]
    # A new router (new process) sharing the ledger still sees today's usage...
    router2, _, _ = make_limited_router({"a/one": ModelLimits(rpd=2)},
                                        usage=UsageLedger(tmp_path / "u.sqlite"))
    assert run(router2.complete("gen", MESSAGES)).model == "b/two"
    # ...but a new UTC day resets the cap.
    router3, _, _ = make_limited_router({"a/one": ModelLimits(rpd=2)},
                                        usage=UsageLedger(tmp_path / "u.sqlite"),
                                        day="2026-09-24")
    assert run(router3.complete("gen", MESSAGES)).model == "a/one"


def test_tpm_throttle_skips_to_next_model_instead_of_waiting():
    # 1,200 tokens/min; each call estimates ~1,024+ tokens, so the 2nd call would wait ~50s.
    router, fake, clock = make_limited_router({"a/one": ModelLimits(tpm=1200)})
    assert run(router.complete("gen", MESSAGES)).model == "a/one"
    assert run(router.complete("gen", MESSAGES, use_cache=False)).model == "b/two"
    assert clock.now == 0  # skipped, never slept
    assert router.expected_wait("a/one", 1024) > 15


def test_last_model_in_chain_waits_rather_than_failing():
    router, fake, clock = make_limited_router(
        {"a/one": ModelLimits(rpm=1), "b/two": ModelLimits(rpm=1)}, max_wait=5)
    for _ in range(3):
        run(router.complete("gen", MESSAGES, use_cache=False))
    assert fake.calls == ["a/one", "b/two", "b/two"]  # a/one skipped once; b/two waited
    assert clock.now == pytest.approx(60, rel=0.01)


def test_usage_ledger_records_requests_and_tokens():
    ledger = UsageLedger()
    router, _, _ = make_limited_router({}, usage=ledger)
    run(router.complete("gen", MESSAGES))
    assert ledger.day_summary("2026-09-23") == {"a/one": (1, 8)}


def test_token_bucket_amounts_and_wait_time():
    clock = FakeClock()
    bucket = TokenBucket(per_minute=600, capacity=600, clock=clock, sleep=clock.sleep)
    assert bucket.wait_time(600) == 0
    run(bucket.acquire(600))
    assert bucket.wait_time(300) == pytest.approx(30)  # 10 tokens/s
    run(bucket.acquire(5000))  # oversized: capped at capacity, waits for a full bucket
    assert clock.now == pytest.approx(60)


class QuotaError(Exception):
    """Shaped like Gemini's 429 for an exhausted daily quota."""

    status_code = 429

    def __init__(self):
        super().__init__("RESOURCE_EXHAUSTED: Quota exceeded for metric "
                         "generate_content_free_tier_requests, quotaId: "
                         "GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20")


def test_daily_quota_429_is_not_retried_and_skips_model_for_the_day():
    clock = FakeClock()
    cfg = LLMConfig(roles={"gen": ["a/one", "b/two"]}, retries=3,
                    providers={"a": ProviderLimits(rpm=None), "b": ProviderLimits(rpm=None)})
    fake = FakeCompletion({"a/one": [QuotaError(), "a"], "b/two": ["b1", "b2"]})
    router = LLMRouter(cfg, completion_fn=fake, require_keys=False, clock=clock,
                       sleep=clock.sleep, today=lambda _tz: "2026-09-23")
    assert run(router.complete("gen", MESSAGES)).model == "b/two"
    assert fake.calls == ["a/one", "b/two"] and clock.now == 0  # no retries, no backoff
    assert run(router.complete("gen", MESSAGES, use_cache=False)).model == "b/two"
    assert fake.calls.count("a/one") == 1  # skipped for the rest of the quota day


def test_ordinary_rate_limit_is_still_retried():
    from autocover.llm.router import is_daily_quota_error

    assert is_daily_quota_error(QuotaError())
    assert not is_daily_quota_error(ProviderError(429))  # per-minute limit: retry later


def test_quota_day_uses_provider_time_zone():
    seen = []
    cfg = LLMConfig(roles={"gen": ["gemini/x"]},
                    providers={"gemini": ProviderLimits(quota_timezone="America/Los_Angeles")},
                    models={"gemini/x": ModelLimits(rpd=5)})
    router = LLMRouter(cfg, completion_fn=FakeCompletion({"gemini/x": ["ok"]}),
                       require_keys=False, today=lambda tz: seen.append(tz) or "2026-09-23")
    run(router.complete("gen", MESSAGES))
    assert set(seen) == {"America/Los_Angeles"}


def test_provider_api_base_and_key_env_are_passed_through(monkeypatch):
    monkeypatch.setenv("MY_OLLAMA_KEY", "secret")
    seen = {}

    async def completion(**kwargs):
        seen.update(kwargs)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    cfg = LLMConfig(roles={"gen": ["ollama_chat/gpt-oss:120b"]},
                    providers={"ollama_chat": ProviderLimits(
                        api_base="https://ollama.com", api_key_env="MY_OLLAMA_KEY")})
    router = LLMRouter(cfg, completion_fn=completion)
    assert router.key_env("ollama_chat/gpt-oss:120b") == "MY_OLLAMA_KEY"
    run(router.complete("gen", MESSAGES))
    assert seen["api_base"] == "https://ollama.com" and seen["api_key"] == "secret"
    monkeypatch.delenv("MY_OLLAMA_KEY")
    assert not router.available("ollama_chat/gpt-oss:120b")
