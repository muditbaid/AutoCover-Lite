"""Role-based LLM router over free-tier providers.

Each agent role maps to an ordered fallback chain of LiteLLM model ids. For every call
the router walks the chain and, per model:

* skips it if its provider has no API key, its circuit breaker is open, or its
  requests-per-day cap is used up (counted in a persistent usage ledger);
* serves an exact-match reply from the SQLite cache when possible;
* skips it if its per-model RPM / TPM buckets (or the provider-wide RPM bucket) would
  make the call wait longer than `max_queue_wait_s`, unless it is the last option;
* otherwise waits on those buckets and the provider's concurrency semaphore, and
  retries retryable errors (429 / 5xx / timeouts) with jittered exponential backoff,
  then moves to the next model.

This mirrors AutoCover's multi-level fallbacks, adaptive concurrency and circuit breakers
(paper section 5.1), scaled down to free-tier quotas.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from autocover.config import LLMConfig
from autocover.llm.cache import ResponseCache, UsageLedger, cache_key
from autocover.telemetry import Telemetry

CompletionFn = Callable[..., Awaitable[Any]]
DEFAULT_LATENCY_S = 10.0  # assumed call latency before a model has been observed

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
RETRYABLE_NAMES = ("RateLimit", "Timeout", "ServiceUnavailable", "APIConnection", "InternalServer")

# Environment variable LiteLLM reads for each provider prefix.
PROVIDER_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "nvidia_nim": "NVIDIA_NIM_API_KEY",
    "zai": "ZAI_API_KEY",
    "cloudflare": "CLOUDFLARE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class AllModelsFailed(RuntimeError):
    """Every model in a role's chain was skipped or failed."""


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached: bool = False
    latency_s: float = 0.0
    fallbacks: int = 0  # how many earlier models in the chain were skipped or failed


def provider_of(model: str) -> str:
    return model.split("/", 1)[0]


_DAILY_QUOTA = re.compile(
    r"per[ _-]?day|PerDay|daily (quota|limit|free allocation)|requests per day", re.I
)


def is_daily_quota_error(exc: BaseException) -> bool:
    """A 429 caused by an exhausted *daily* quota: retrying today is pointless."""
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    text = str(exc)
    looks_429 = (status == 429 or "RESOURCE_EXHAUSTED" in text
                 or "RateLimit" in type(exc).__name__ or "free allocation" in text)
    return looks_429 and bool(_DAILY_QUOTA.search(text))


def is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and status in RETRYABLE_STATUS:
        return True
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    return any(name in type(exc).__name__ for name in RETRYABLE_NAMES)


class TokenBucket:
    """Continuous-refill limiter. Units are requests (RPM) or tokens (TPM).

    `per_minute` units refill evenly; `capacity` bounds the burst (defaults to a small
    burst for request buckets; pass `capacity=tpm` for token buckets).
    """

    def __init__(
        self,
        per_minute: float,
        capacity: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.rate = per_minute / 60.0
        self.capacity = capacity if capacity is not None else max(1.0, min(per_minute, 5.0))
        self.tokens = self.capacity
        self.reserved = 0.0  # committed by callers that have not acquired yet
        self._clock, self._sleep = clock, sleep
        self._last = clock()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
        self._last = now

    def wait_time(self, amount: float = 1.0) -> float:
        """Seconds until `amount` units would be available after everything already
        reserved by queued callers (0 if available now)."""
        self._refill()
        amount = min(amount, self.capacity)
        return max(0.0, (amount + self.reserved - self.tokens) / self.rate)

    def refund(self, amount: float) -> None:
        """Give back units that were estimated but not used (e.g. unused completion
        tokens)."""
        if amount > 0:
            self._refill()
            self.tokens = min(self.capacity, self.tokens + amount)

    async def acquire(self, amount: float = 1.0) -> None:
        amount = min(amount, self.capacity)  # an oversized request waits for a full bucket
        async with self._lock:
            while True:
                self._refill()
                if self.tokens >= amount:
                    self.tokens -= amount
                    return
                await self._sleep((amount - self.tokens) / self.rate)


class CircuitBreaker:
    """Opens after `threshold` consecutive failures; half-opens after `cooldown_s`."""

    def __init__(self, threshold: int, cooldown_s: float, clock: Callable[[], float]):
        self.threshold, self.cooldown_s, self._clock = threshold, cooldown_s, clock
        self._failures: dict[str, int] = {}
        self._opened_at: dict[str, float] = {}

    def is_open(self, model: str) -> bool:
        opened = self._opened_at.get(model)
        if opened is None:
            return False
        if self._clock() - opened >= self.cooldown_s:
            # Half-open: allow one probe; a failure re-opens immediately.
            del self._opened_at[model]
            self._failures[model] = self.threshold - 1
            return False
        return True

    def record_success(self, model: str) -> None:
        self._failures.pop(model, None)
        self._opened_at.pop(model, None)

    def record_failure(self, model: str) -> None:
        self._failures[model] = self._failures.get(model, 0) + 1
        if self._failures[model] >= self.threshold:
            self._opened_at[model] = self._clock()


class LLMRouter:
    def __init__(
        self,
        config: LLMConfig,
        *,
        cache: ResponseCache | None = None,
        usage: UsageLedger | None = None,
        telemetry: Telemetry | None = None,
        completion_fn: CompletionFn | None = None,
        require_keys: bool = True,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
        today: Callable[[str], str] | None = None,  # time zone -> "YYYY-MM-DD"
    ):
        self.config = config
        self.cache = cache
        self.usage = usage or UsageLedger(":memory:")
        self._today = today or (lambda tz: datetime.now(ZoneInfo(tz)).date().isoformat())
        self._exhausted: set[tuple[str, str]] = set()  # (quota day, model) told "daily quota"
        self.telemetry = telemetry or Telemetry()
        self._completion_fn = completion_fn
        self._require_keys = require_keys
        self._clock, self._sleep = clock, sleep
        self._rng = rng or random.Random()
        self.breaker = CircuitBreaker(
            config.circuit_breaker.failure_threshold, config.circuit_breaker.cooldown_s, clock
        )
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._queued: dict[str, int] = {}     # provider -> committed callers not yet running
        self._running: dict[str, int] = {}    # provider -> calls in flight
        self._latency: dict[str, float] = {}  # model -> moving average latency (s)
        self._buckets: dict[tuple[str, str], TokenBucket] = {}  # (scope, name) -> bucket

    # -- public API ---------------------------------------------------------------

    def chain(self, role: str) -> list[str]:
        try:
            return self.config.roles[role]
        except KeyError:
            raise KeyError(f"no models configured for role {role!r}") from None

    def key_env(self, model: str) -> str | None:
        """Environment variable holding the API key for `model`'s provider."""
        provider = provider_of(model)
        return self.config.limits_for(provider).api_key_env or PROVIDER_KEY_ENV.get(provider)

    def available(self, model: str) -> bool:
        if not self._require_keys:
            return True
        env = self.key_env(model)
        return env is None or bool(os.environ.get(env))

    async def complete(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        params = {"temperature": self.config.temperature, "json_mode": json_mode,
                  "max_tokens": max_tokens}
        errors: list[str] = []
        chain = self.chain(role)
        for index, model in enumerate(chain):
            is_last = index == len(chain) - 1
            if not self.available(model):
                errors.append(f"{model}: no API key")
                continue
            if self.breaker.is_open(model):
                errors.append(f"{model}: circuit open")
                continue
            if self._daily_cap_reached(model):
                errors.append(f"{model}: daily request cap reached")
                self.telemetry.event("llm", "skip_daily_cap", role=role, model=model)
                continue
            key = cache_key(model, messages, params)
            if use_cache and self.cache and (hit := self.cache.get(key)):
                resp = LLMResponse(**hit, cached=True, fallbacks=index)
                self._emit(role, resp)
                return resp
            est_tokens = estimate_tokens(messages, max_tokens)
            wait = self.expected_wait(model, est_tokens)
            if wait > self.config.max_queue_wait_s and not is_last:
                errors.append(f"{model}: throttled (~{wait:.0f}s wait)")
                self.telemetry.event("llm", "skip_throttled", role=role, model=model,
                                     wait_s=round(wait, 1))
                continue
            # Commit to this model: reserve its bucket units and join its queue, so
            # concurrent callers see the real backlog (and spill over to fallbacks)
            # instead of all seeing an empty bucket at the same instant.
            reservation = self._limiters(model, est_tokens)
            for bucket, amount in reservation:
                bucket.reserved += amount
            provider = provider_of(model)
            self._queued[provider] = self._queued.get(provider, 0) + 1
            try:
                resp = await self._call_with_retries(model, messages, params, est_tokens,
                                                     reservation)
            except Exception as exc:  # noqa: BLE001 - any provider error moves down the chain
                errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:200]}")
                self.telemetry.event("llm", "model_failed", role=role, model=model,
                                     error=type(exc).__name__)
                continue
            resp.fallbacks = index
            if self.cache:
                self.cache.put(key, model, {
                    "text": resp.text, "model": resp.model,
                    "prompt_tokens": resp.prompt_tokens,
                    "completion_tokens": resp.completion_tokens,
                    "latency_s": resp.latency_s,
                })
            self._emit(role, resp)
            return resp
        raise AllModelsFailed(f"role {role!r}: " + "; ".join(errors))

    # -- internals ------------------------------------------------------------------

    async def _call_with_retries(
        self, model: str, messages: list[dict[str, Any]], params: dict[str, Any],
        est_tokens: int, reservation: list[tuple[TokenBucket, float]] | None = None,
    ) -> LLMResponse:
        provider = provider_of(model)
        attempts = self.config.retries + 1
        pending = list(reservation or [])
        queued = reservation is not None

        def settle() -> None:  # drop our reservation / queue slot once, when we start
            nonlocal pending, queued
            for bucket, amount in pending:
                bucket.reserved = max(0.0, bucket.reserved - amount)
            pending = []
            if queued:
                self._queued[provider] = max(0, self._queued.get(provider, 0) - 1)
                queued = False

        try:
            for attempt in range(attempts):
                try:
                    async with self._semaphore(provider):
                        limiters = self._limiters(model, est_tokens)
                        for bucket, amount in limiters:
                            await bucket.acquire(amount)
                        settle()
                        self._running[provider] = self._running.get(provider, 0) + 1
                        self.usage.record(self._day(model), model)  # every attempt counts
                        start = self._clock()
                        try:
                            raw = await self._invoke(model, messages, params)
                        finally:
                            self._running[provider] -= 1
                        latency = self._clock() - start
                except Exception as exc:
                    if is_daily_quota_error(exc):
                        self._exhausted.add((self._day(model), model))
                        self.telemetry.event("llm", "daily_quota_exhausted", model=model)
                        raise
                    if is_retryable(exc) and attempt < attempts - 1:
                        self.telemetry.event("llm", "retry", model=model, attempt=attempt + 1,
                                             error=type(exc).__name__)
                        await self._sleep(self._backoff(attempt))
                        continue
                    self.breaker.record_failure(model)
                    raise
                self.breaker.record_success(model)
                text, usage = _unpack(raw)
                used = usage["prompt_tokens"] + usage["completion_tokens"]
                for bucket, amount in limiters:
                    if amount > 1 and used:  # token bucket: refund the unused estimate
                        bucket.refund(amount - used)
                previous = self._latency.get(model)
                self._latency[model] = (latency if previous is None
                                        else 0.7 * previous + 0.3 * latency)
                self.usage.record(self._day(model), model, requests=0, tokens=used)
                return LLMResponse(
                    text=text, model=model, latency_s=latency,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
            raise AssertionError("unreachable")
        finally:
            settle()

    async def _invoke(self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": params["temperature"],
            "timeout": self.config.timeout_s,
        }
        if params["max_tokens"]:
            kwargs["max_tokens"] = params["max_tokens"]
        if params["json_mode"]:
            kwargs["response_format"] = {"type": "json_object"}
        provider = self.config.limits_for(provider_of(model))
        if provider.api_base:
            kwargs["api_base"] = provider.api_base
        if provider.api_key_env and os.environ.get(provider.api_key_env):
            kwargs["api_key"] = os.environ[provider.api_key_env]
        if self._completion_fn is not None:
            return await self._completion_fn(**kwargs)
        import litellm  # heavy import; deferred so tests and the CLI start fast

        litellm.drop_params = True  # silently drop params a provider doesn't support
        return await litellm.acompletion(**kwargs)

    def _backoff(self, attempt: int) -> float:
        base = min(self.config.backoff_max_s, self.config.backoff_base_s * 2**attempt)
        return base * self._rng.uniform(0.5, 1.5)

    def _semaphore(self, provider: str) -> asyncio.Semaphore:
        if provider not in self._semaphores:
            limit = self.config.limits_for(provider).max_concurrency
            self._semaphores[provider] = asyncio.Semaphore(limit)
        return self._semaphores[provider]

    def _limiters(self, model: str, est_tokens: int) -> list[tuple[TokenBucket, float]]:
        """(bucket, amount) pairs a call to `model` must acquire."""
        limiters: list[tuple[TokenBucket, float]] = []
        provider = provider_of(model)
        provider_rpm = self.config.limits_for(provider).rpm
        if provider_rpm:
            limiters.append((self._bucket("provider", provider, provider_rpm), 1))
        limits = self.config.model_limits(model)
        if limits.rpm:
            limiters.append((self._bucket("rpm", model, limits.rpm), 1))
        if limits.tpm:
            limiters.append((self._bucket("tpm", model, limits.tpm, capacity=limits.tpm),
                             est_tokens))
        return limiters

    def expected_wait(self, model: str, est_tokens: int) -> float:
        """Rate-limit wait (including units reserved by queued callers) plus the wait for
        a concurrency slot behind the calls already queued or running."""
        buckets = max((b.wait_time(a) for b, a in self._limiters(model, est_tokens)),
                      default=0.0)
        provider = provider_of(model)
        limit = max(1, self.config.limits_for(provider).max_concurrency)
        ahead = self._queued.get(provider, 0) + self._running.get(provider, 0)
        slots = max(0, ahead - limit + 1) / limit * self._latency.get(model, DEFAULT_LATENCY_S)
        return max(buckets, slots)

    def _day(self, model: str) -> str:
        """Current quota day for `model`, in its provider's reset time zone."""
        return self._today(self.config.limits_for(provider_of(model)).quota_timezone)

    def _daily_cap_reached(self, model: str) -> bool:
        day = self._day(model)
        if (day, model) in self._exhausted:
            return True
        rpd = self.config.model_limits(model).rpd
        return rpd is not None and self.usage.requests(day, model) >= rpd

    def _bucket(self, scope: str, name: str, per_minute: float,
                capacity: float | None = None) -> TokenBucket:
        key = (scope, name)
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(per_minute, capacity=capacity,
                                             clock=self._clock, sleep=self._sleep)
        return self._buckets[key]

    def _emit(self, role: str, resp: LLMResponse) -> None:
        self.telemetry.event(
            "llm", "completion", role=role, model=resp.model, cached=resp.cached,
            fallbacks=resp.fallbacks, prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens, latency_s=round(resp.latency_s, 3),
        )


def estimate_tokens(messages: list[dict[str, Any]], max_tokens: int | None) -> int:
    """Rough prompt size (~4 chars/token) plus the completion budget, for TPM limiting."""
    chars = sum(len(str(m.get("content") or "")) for m in messages)
    return chars // 4 + (max_tokens or 1024)


def _unpack(raw: Any) -> tuple[str, dict[str, int]]:
    """Read text and token usage from a LiteLLM ModelResponse or an equivalent dict."""
    get = (lambda o, k: o.get(k)) if isinstance(raw, dict) else getattr
    choice = get(raw, "choices")[0]
    message = choice.get("message") if isinstance(choice, dict) else choice.message
    text = (message.get("content") if isinstance(message, dict) else message.content) or ""
    usage_obj = get(raw, "usage") or {}
    read = usage_obj.get if isinstance(usage_obj, dict) else (lambda f: getattr(usage_obj, f, 0))
    usage = {field: int(read(field) or 0) for field in ("prompt_tokens", "completion_tokens")}
    return text, usage
