"""Typed configuration loaded from config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ProviderLimits(BaseModel):
    """Limits shared by every model of a provider (e.g. NVIDIA NIM's account-wide RPM)."""

    rpm: float | None = 10
    max_concurrency: int = 1
    # Time zone in which the provider's daily quotas reset (Gemini: midnight Pacific).
    quota_timezone: str = "UTC"
    # Connection overrides, e.g. Ollama Cloud: api_base https://ollama.com + OLLAMA_API_KEY.
    api_base: str | None = None
    api_key_env: str | None = None


class ModelLimits(BaseModel):
    """Per-model free-tier quotas. Free tiers meter most limits per model version."""

    rpm: float | None = None  # requests per minute
    tpm: float | None = None  # tokens per minute (prompt + completion)
    rpd: int | None = None    # requests per day (provider's quota day)


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 3
    cooldown_s: float = 120


class CacheConfig(BaseModel):
    enabled: bool = True
    path: str = ".autocover/llm_cache.sqlite"


class LLMConfig(BaseModel):
    roles: dict[str, list[str]] = Field(default_factory=dict)
    providers: dict[str, ProviderLimits] = Field(default_factory=dict)
    models: dict[str, ModelLimits] = Field(default_factory=dict)
    # If a model's limits would make a call wait longer than this, try the next model in
    # the chain instead (the last model in a chain always waits).
    max_queue_wait_s: float = 15
    temperature: float = 0.2
    timeout_s: float = 120
    retries: int = 3
    backoff_base_s: float = 1.0
    backoff_max_s: float = 30.0
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    usage_path: str = ".autocover/usage.sqlite"  # per-day request counts for RPD caps

    def limits_for(self, provider: str) -> ProviderLimits:
        return self.providers.get(provider, ProviderLimits())

    def model_limits(self, model: str) -> ModelLimits:
        return self.models.get(model, ModelLimits())


class SandboxConfig(BaseModel):
    backend: Literal["docker", "local"] = "docker"
    python_version: str = "3.12"
    timeout_s: float = 120
    memory: str = "1g"
    cpus: float = 1.0
    pids_limit: int = 256
    max_parallel: int = 4
    workdir: str | None = None


class MutationConfig(BaseModel):
    enabled: bool = True
    max_mutants_per_function: int = 8    # size of the per-function mutant pool
    max_mutants_per_candidate: int = 12  # mutants run against one candidate test
    final_score: bool = True             # mutation score of the final suite
    timeout_s: float = 30
    seed: int = 0


class RunConfig(BaseModel):
    max_rounds: int = 3              # generate -> execute rounds per run
    budget_min: float = 10           # wall-clock budget for a run
    max_scenarios_per_function: int = 6
    preparer_max_tokens: int = 4096
    generator_max_tokens: int = 8192  # reasoning models spend part of this thinking
    include_private: bool = False    # also test _private functions
    max_fix_attempts: int = 2        # Fixer attempts per test before it is frozen
    judge_scenarios: bool = True     # LLM judge may accept a test for a new scenario


class TelemetryConfig(BaseModel):
    jsonl_path: str | None = ".autocover/telemetry.jsonl"
    otlp_endpoint: str | None = None


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    mutation: MutationConfig = Field(default_factory=MutationConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)


def load_config(path: str | Path | None = "config.yaml") -> Config:
    """Load config from YAML; a missing file yields all defaults."""
    if path is None or not Path(path).exists():
        return Config()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config.model_validate(data)
