import asyncio

import pytest

from autocover.config import CircuitBreakerConfig, LLMConfig, ProviderLimits
from autocover.llm.cache import ResponseCache
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
    bucket = TokenBucket(rpm=60, capacity=1, clock=clock, sleep=clock.sleep)

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
