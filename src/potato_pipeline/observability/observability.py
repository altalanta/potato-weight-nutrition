"""Observability helpers: logging, Prometheus counters, OpenTelemetry spans."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass

from .utils import LOGGER_NAME


@dataclass
class ObservabilityState:
    enabled: bool
    metrics_enabled: bool
    tracing_enabled: bool
    port: int


class ObservabilityManager:
    """Optional observability plumbing used by the CLI."""

    def __init__(self, enabled: bool = False, port: int = 9157) -> None:
        self.state = ObservabilityState(enabled=enabled, metrics_enabled=False, tracing_enabled=False, port=port)
        self._registry = None
        self._counter = None
        self._histogram = None
        self._tracer = None
        if not enabled:
            return
        self._configure_metrics()
        self._configure_tracing()

    def _configure_metrics(self) -> None:
        try:
            from prometheus_client import CollectorRegistry, Counter, Histogram, start_http_server
        except ImportError:  # pragma: no cover - optional dependency
            return
        self._registry = CollectorRegistry()
        self._counter = Counter(
            "pwn_pipeline_steps_total",
            "Count of completed pipeline steps",
            labelnames=("step",),
            registry=self._registry,
        )
        self._histogram = Histogram(
            "pwn_pipeline_step_seconds",
            "Duration of pipeline steps in seconds",
            labelnames=("step",),
            registry=self._registry,
            buckets=(0.5, 1, 2, 5, 10, 20, 40, 80, float("inf")),
        )
        start_http_server(self.state.port, registry=self._registry)
        self.state.metrics_enabled = True

    def _configure_tracing(self) -> None:
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
        except ImportError:  # pragma: no cover - optional dependency
            return
        resource = Resource(attributes={"service.name": "potato-weight-nutrition"})
        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)
        self._tracer = trace.get_tracer(LOGGER_NAME)
        self.state.tracing_enabled = True

    @contextlib.contextmanager
    def record_step(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        span_cm = self._span(name)
        span = span_cm if self.state.tracing_enabled else contextlib.nullcontext()
        with span:
            try:
                yield
            finally:
                duration = time.perf_counter() - start
                if self.state.metrics_enabled and self._counter and self._histogram:
                    self._counter.labels(step=name).inc()
                    self._histogram.labels(step=name).observe(duration)

    def _span(self, name: str) -> contextlib.AbstractContextManager[None]:
        if not self.state.tracing_enabled or self._tracer is None:  # pragma: no cover - optional dependency
            return contextlib.nullcontext()
        return self._tracer.start_as_current_span(name)
