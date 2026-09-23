"""Typed configuration loaded from config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ProviderLimits(BaseModel):
    rpm: float = 10
    max_concurrency: int = 1


class CircuitBreakerConfig(BaseModel):
    failure_threshold: int = 3
    cooldown_s: float = 120


class CacheConfig(BaseModel):
    enabled: bool = True
    path: str = ".autocover/llm_cache.sqlite"


class LLMConfig(BaseModel):
    roles: dict[str, list[str]] = Field(default_factory=dict)
    providers: dict[str, ProviderLimits] = Field(default_factory=dict)
    temperature: float = 0.2
    timeout_s: float = 120
    retries: int = 3
    backoff_base_s: float = 1.0
    backoff_max_s: float = 30.0
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    def limits_for(self, provider: str) -> ProviderLimits:
        return self.providers.get(provider, ProviderLimits())


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
    max_mutants_per_function: int = 8
    seed: int = 0


class TelemetryConfig(BaseModel):
    jsonl_path: str | None = ".autocover/telemetry.jsonl"
    otlp_endpoint: str | None = None


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    mutation: MutationConfig = Field(default_factory=MutationConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)


def load_config(path: str | Path | None = "config.yaml") -> Config:
    """Load config from YAML; a missing file yields all defaults."""
    if path is None or not Path(path).exists():
        return Config()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config.model_validate(data)
