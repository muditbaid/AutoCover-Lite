"""Wiring shared by the CLI, the agent graph and the benchmark scripts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from autocover.config import Config
from autocover.llm.cache import ResponseCache, UsageLedger
from autocover.llm.router import LLMRouter
from autocover.telemetry import Telemetry


@dataclass
class Runtime:
    config: Config
    telemetry: Telemetry
    router: LLMRouter
    cache: ResponseCache | None
    usage: UsageLedger

    def close(self) -> None:
        if self.cache:
            self.cache.close()
        self.usage.close()
        self.telemetry.close()


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader: KEY=VALUE lines; existing environment variables win."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def build_runtime(config: Config, *, run_id: str | None = None) -> Runtime:
    endpoint = config.telemetry.otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    telemetry = Telemetry(run_id=run_id, jsonl_path=config.telemetry.jsonl_path,
                          otlp_endpoint=endpoint)
    cache = ResponseCache(config.llm.cache.path) if config.llm.cache.enabled else None
    usage = UsageLedger(config.llm.usage_path)
    router = LLMRouter(config.llm, cache=cache, usage=usage, telemetry=telemetry)
    return Runtime(config=config, telemetry=telemetry, router=router, cache=cache, usage=usage)
