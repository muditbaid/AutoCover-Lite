"""Structured telemetry: JSONL events plus optional OpenTelemetry spans.

Every event carries the run's correlation id so a whole Preparer -> ... -> Fixer run can be
reconstructed from the JSONL file, and every `span` is also exported to an OTLP endpoint
(Jaeger in docker-compose) when one is configured.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class Telemetry:
    def __init__(
        self,
        run_id: str | None = None,
        jsonl_path: str | Path | None = None,
        otlp_endpoint: str | None = None,
    ):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.path = Path(jsonl_path) if jsonl_path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._tracer = _make_tracer(otlp_endpoint) if otlp_endpoint else None
        self.events: list[dict[str, Any]] = []  # in-memory copy for run summaries

    def event(self, stage: str, name: str, **fields: Any) -> dict[str, Any]:
        record = {"ts": round(time.time(), 3), "run_id": self.run_id, "stage": stage,
                  "event": name, **fields}
        with self._lock:
            self.events.append(record)
            if self.path:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, default=str) + "\n")
        return record

    @contextmanager
    def span(self, stage: str, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        """Time a block; callers may add result attributes to the yielded dict."""
        extra: dict[str, Any] = dict(attrs)
        start = time.perf_counter()
        otel_cm = self._tracer.start_as_current_span(f"{stage}.{name}") if self._tracer else None
        otel_span = otel_cm.__enter__() if otel_cm else None
        status = "ok"
        try:
            yield extra
        except BaseException as exc:
            status = f"error:{type(exc).__name__}"
            raise
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            if otel_span is not None:
                otel_span.set_attribute("autocover.run_id", self.run_id)
                for key, value in extra.items():
                    if isinstance(value, (str, int, float, bool)):
                        otel_span.set_attribute(f"autocover.{key}", value)
                otel_span.set_attribute("autocover.status", status)
                otel_cm.__exit__(None, None, None)
            self.event(stage, name, duration_ms=duration_ms, status=status, **extra)


def _make_tracer(endpoint: str):
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:  # optional dependency
        return None
    provider = TracerProvider(resource=Resource.create({"service.name": "autocover-lite"}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces"))
    )
    trace.set_tracer_provider(provider)
    return trace.get_tracer("autocover")
