"""Langfuse traces, when there is a Langfuse to send them to.

Opt-in and silent by default: with no `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` in the environment, `tracer()` returns something whose
every method does nothing. The gate CI runs, the tests, and a replay on a
laptop therefore never need a server, a network, or the dependency — which is
why `langfuse` is an extra (`uv sync --extra tracing`) rather than a dependency
of the harness.

What a run looks like there: one trace per item and condition, carrying the
request as its input; one generation inside it per turn, with the prompt, the
answer and the token counts the provider reported; and three scores on the
trace — did it parse, did it validate, was it the right chart. The error codes
and mismatch codes ride along as metadata, so a condition can be compared
against another by which failures came back, not only by how many.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from .dataset import Item
from .score import Outcome
from .store import Record

#: Both must be set for tracing to switch on. `LANGFUSE_HOST` picks the
#: instance; the SDK defaults it to Langfuse Cloud, so a self-hosted run sets
#: it to its own URL.
REQUIRED = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def configured() -> bool:
    return all(os.environ.get(name) for name in REQUIRED)


def host() -> str:
    return os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")


class NullTracer:
    """What you get when Langfuse is not configured: nothing, quietly."""

    enabled = False

    @contextmanager
    def item(self, *args: Any, **kwargs: Any) -> Iterator["NullTracer"]:
        yield self

    def turn(self, *args: Any, **kwargs: Any) -> None:
        pass

    def score(self, *args: Any, **kwargs: Any) -> None:
        pass

    def flush(self) -> None:
        pass


class Tracer:
    """One trace per item and condition, one generation per turn."""

    enabled = True

    def __init__(self, run: str, dataset: str):
        from langfuse import Langfuse

        self.run = run
        self.dataset = dataset
        self._client = Langfuse()
        self._span: Any = None

    @contextmanager
    def item(self, item: Item, condition: str, model: str) -> Iterator["Tracer"]:
        from langfuse import propagate_attributes

        tags = [f"run:{self.run}", f"dataset:{self.dataset}",
                f"condition:{condition}", f"model:{model}", *item.tags]
        with propagate_attributes(tags=tags, metadata={"item": item.id,
                                                       "frame": item.frame,
                                                       "expect": item.expect}):
            with self._client.start_as_current_observation(
                name=f"{item.id} · {condition}",
                input=item.request,
            ) as span:
                self._span = span
                try:
                    yield self
                finally:
                    self._span = None

    def turn(self, record: Record, model: str, params: dict, user: str) -> None:
        if self._span is None:
            return
        usage = {k: v for k, v in (("input", record.input_tokens),
                                   ("output", record.output_tokens)) if v is not None}
        with self._span.start_as_current_generation(
            name=f"turn {record.turn}",
            model=model,
            model_parameters=params,
            input=user,
        ) as generation:
            generation.update(output=record.raw, usage_details=usage or None)

    def score(self, outcome: Outcome) -> None:
        """Three questions, in the order they stop mattering."""
        if self._span is None:
            return
        for name, value in (("parsed", outcome.parsed), ("valid", outcome.valid),
                            ("right", outcome.scored)):
            self._span.score(name=name, value=int(value), data_type="NUMERIC")
        self._span.update(output=outcome.spec, metadata={
            "codes": outcome.codes,
            "errors": outcome.errors,
            "mismatches": outcome.mismatches,
        })

    def flush(self) -> None:
        self._client.flush()


def tracer(run: str, dataset: str, enabled: bool | None = None):
    """A tracer, or a no-op when Langfuse is not configured.

    `enabled=True` asks for it explicitly, and then a missing key is an error
    rather than a silent afternoon of untraced runs.
    """
    if enabled is None:
        enabled = configured()
    if not enabled:
        return NullTracer()
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"tracing was asked for, but {', '.join(missing)} "
                           f"is not set")
    return Tracer(run, dataset)
