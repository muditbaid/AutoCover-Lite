"""Role-based LLM router over free-tier providers.

Each agent role maps to an ordered fallback chain of LiteLLM model ids. For every call
the router walks the chain and, per model:

* skips it if its circuit breaker is open, or its provider has no API key;
* serves an exact-match reply from the SQLite cache when possible;
* waits on the provider's concurrency semaphore and requests-per-minute token bucket;
* retries retryable errors (429 / 5xx / timeouts) with jittered exponential backoff,
  then moves to the next model.

This mirrors AutoCover's multi-level fallbacks, adaptive concurrency and circuit breakers
(paper section 5.1), scaled down to free-tier quotas.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from autocover.config import LLMConfig
from autocover.llm.cache import ResponseCache, cache_key
from autocover.telemetry import Telemetry

CompletionFn = Callable[..., Awaitable[Any]]

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


def is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and status in RETRYABLE_STATUS:
        return True
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True
    return any(name in type(exc).__name__ for name in RETRYABLE_NAMES)


class TokenBucket:
    """Requests-per-minute limiter; `capacity` allows a short burst."""

    def __init__(
        self,
        rpm: float,
        capacity: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.rate = rpm / 60.0
        self.capacity = capacity if capacity is not None else max(1.0, min(rpm, 5.0))
        self.tokens = self.capacity
        self._clock, self._sleep = clock, sleep
        self._last = clock()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = self._clock()
                self.tokens = min(self.capacity, self.tokens + (now - self._last) * self.rate)
                self._last = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                await self._sleep((1 - self.tokens) / self.rate)


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
        telemetry: Telemetry | None = None,
        completion_fn: CompletionFn | None = None,
        require_keys: bool = True,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
    ):
        self.config = config
        self.cache = cache
        self.telemetry = telemetry or Telemetry()
        self._completion_fn = completion_fn
        self._require_keys = require_keys
        self._clock, self._sleep = clock, sleep
        self._rng = rng or random.Random()
        self.breaker = CircuitBreaker(
            config.circuit_breaker.failure_threshold, config.circuit_breaker.cooldown_s, clock
        )
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._buckets: dict[str, TokenBucket] = {}

    # -- public API ---------------------------------------------------------------

    def chain(self, role: str) -> list[str]:
        try:
            return self.config.roles[role]
        except KeyError:
            raise KeyError(f"no models configured for role {role!r}") from None

    def available(self, model: str) -> bool:
        if not self._require_keys:
            return True
        env = PROVIDER_KEY_ENV.get(provider_of(model))
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
        for index, model in enumerate(self.chain(role)):
            if not self.available(model):
                errors.append(f"{model}: no API key")
                continue
            if self.breaker.is_open(model):
                errors.append(f"{model}: circuit open")
                continue
            key = cache_key(model, messages, params)
            if use_cache and self.cache and (hit := self.cache.get(key)):
                resp = LLMResponse(**hit, cached=True, fallbacks=index)
                self._emit(role, resp)
                return resp
            try:
                resp = await self._call_with_retries(model, messages, params)
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
        self, model: str, messages: list[dict[str, Any]], params: dict[str, Any]
    ) -> LLMResponse:
        provider = provider_of(model)
        attempts = self.config.retries + 1
        for attempt in range(attempts):
            try:
                async with self._semaphore(provider):
                    await self._bucket(provider).acquire()
                    start = self._clock()
                    raw = await self._invoke(model, messages, params)
                    latency = self._clock() - start
            except Exception as exc:
                if is_retryable(exc) and attempt < attempts - 1:
                    self.telemetry.event("llm", "retry", model=model, attempt=attempt + 1,
                                         error=type(exc).__name__)
                    await self._sleep(self._backoff(attempt))
                    continue
                self.breaker.record_failure(model)
                raise
            self.breaker.record_success(model)
            text, usage = _unpack(raw)
            return LLMResponse(
                text=text, model=model, latency_s=latency,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )
        raise AssertionError("unreachable")

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

    def _bucket(self, provider: str) -> TokenBucket:
        if provider not in self._buckets:
            rpm = self.config.limits_for(provider).rpm
            self._buckets[provider] = TokenBucket(rpm, clock=self._clock, sleep=self._sleep)
        return self._buckets[provider]

    def _emit(self, role: str, resp: LLMResponse) -> None:
        self.telemetry.event(
            "llm", "completion", role=role, model=resp.model, cached=resp.cached,
            fallbacks=resp.fallbacks, prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens, latency_s=round(resp.latency_s, 3),
        )


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
