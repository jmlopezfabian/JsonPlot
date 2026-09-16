"""Put items through a model under a condition, recording before scoring.

Three modes decide when the model is actually called:

- `auto` — use the recording when it answers this exact prompt; call otherwise.
- `replay` — never call. A recorded answer to an older prompt is still used
  (and counted as stale); a missing one is an error. This is what CI runs.
- `fresh` — always call, and record the new answer.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal

import pandas as pd

from . import conditions as cond
from .conditions import Condition
from .dataset import Item
from .providers import Provider
from .score import Outcome, score
from .store import Record, Store, now, prompt_sha

Mode = Literal["auto", "replay", "fresh"]


class ReplayMiss(LookupError):
    """Replay asked for an answer that was never recorded."""


@dataclass
class Result:
    item: Item
    condition: str
    model: str
    outcome: Outcome
    turns: int
    seconds: float
    #: answers recorded against a prompt that no longer matches
    stale: int
    #: answers that came from the model during this run, not the recording
    called: int


class Runner:
    def __init__(self, provider: Provider, store: Store, mode: Mode = "auto",
                 frame_loader: Callable[[str], pd.DataFrame] | None = None,
                 tracer: Any = None):
        from . import frames, tracing

        self.provider = provider
        self.store = store
        self.mode = mode
        self.tracer = tracer or tracing.NullTracer()
        self._load = frame_loader or frames.load
        self._frames: dict[str, pd.DataFrame] = {}
        self._preambles: dict[tuple[str, str], str] = {}

    def frame(self, name: str) -> pd.DataFrame:
        if name not in self._frames:
            self._frames[name] = self._load(name)
        return self._frames[name]

    def preamble(self, condition: Condition, frame: str) -> str:
        key = (condition.name, frame)
        if key not in self._preambles:
            self._preambles[key] = condition.preamble(self.frame(frame))
        return self._preambles[key]

    def run(self, items: list[Item], condition: Condition,
            workers: int = 1) -> Iterator[Result]:
        """Results in item order, yielded as soon as each one is ready."""
        for item in items:                   # build shared state before threading
            self.preamble(condition, item.frame)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(lambda item: self.one(item, condition), items)

    def one(self, item: Item, condition: Condition) -> Result:
        df = self.frame(item.frame)
        preamble = self.preamble(condition, item.frame)
        stale = called = 0
        seconds = 0.0

        with self.tracer.item(item, condition.name, self.provider.model) as trace:

            def ask(turn: int, user: str) -> Record:
                nonlocal stale, called, seconds
                rec, was_stale, was_called = self._answer(item, condition, turn, user)
                stale += was_stale
                called += was_called
                seconds += rec.seconds
                trace.turn(rec, self.provider.model, self.provider.params, user)
                return rec

            rec = ask(0, cond.first_turn(preamble, item.request))
            outcome = score(item, rec.raw, df)
            turns = 1
            if condition.repair and outcome.parsed and not outcome.valid:
                follow = cond.repair_turn(preamble, item.request, outcome.spec,
                                          outcome.errors)
                repaired = score(item, ask(1, follow).raw, df)
                turns = 2
                if repaired.parsed:
                    outcome = repaired
            trace.score(outcome)

        return Result(item, condition.name, self.provider.model, outcome, turns,
                      seconds, stale, called)

    def _answer(self, item: Item, condition: Condition, turn: int,
                user: str) -> tuple[Record, bool, bool]:
        p = self.provider
        sha = prompt_sha(p.name, p.model, p.params, cond.SYSTEM, user)
        rec = self.store.get(item.id, condition.name, turn)
        if rec is not None and self.mode != "fresh":
            if rec.prompt_sha == sha:
                return rec, False, False
            if self.mode == "replay":
                return rec, True, False
        if self.mode == "replay":
            raise ReplayMiss(f"no recorded answer for {item.id} · {condition.name} · "
                             f"turn {turn} · {p.model} in {self.store.path}")
        done = p.complete(cond.SYSTEM, user)
        rec = Record(item=item.id, condition=condition.name, turn=turn,
                     provider=p.name, model=p.model, prompt_sha=sha, raw=done.text,
                     input_tokens=done.input_tokens, output_tokens=done.output_tokens,
                     seconds=round(done.seconds, 3), recorded_at=now())
        self.store.put(rec)
        return rec, False, True
