"""The recording: every answer a model gave, keyed by what it was asked.

One append-only JSONL file per model under `evals/runs/<run>/`. A record is
looked up by (item, condition, turn) and carries the hash of the exact prompt
that produced it, so a replay can tell a faithful answer from one given to a
briefing that has since changed.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import ROOT

RUNS = ROOT / "runs"


@dataclass(frozen=True)
class Record:
    item: str
    condition: str
    turn: int
    provider: str
    model: str
    prompt_sha: str
    raw: str
    input_tokens: int | None
    output_tokens: int | None
    seconds: float
    recorded_at: str

    @property
    def key(self) -> tuple[str, str, int]:
        return self.item, self.condition, self.turn


def prompt_sha(provider: str, model: str, params: dict, system: str, user: str) -> str:
    blob = json.dumps({"provider": provider, "model": model, "params": params,
                       "system": system, "user": user}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._records: dict[tuple[str, str, int], Record] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = Record(**json.loads(line))
                    self._records[rec.key] = rec     # the last answer wins

    @classmethod
    def for_run(cls, run: str, model: str, root: Path = RUNS) -> "Store":
        return cls(root / run / f"{model_slug(model)}.jsonl")

    def get(self, item: str, condition: str, turn: int) -> Record | None:
        return self._records.get((item, condition, turn))

    def put(self, record: Record) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            self._records[record.key] = record

    def conditions(self) -> list[str]:
        """Which conditions this recording actually holds.

        Read rather than assumed: an ablation run carries whatever was asked
        for — single sections, pairs, an `only:` — and a report that iterates a
        fixed list silently omits the rest.
        """
        return sorted({condition for _, condition, _ in self._records})

    def __len__(self) -> int:
        return len(self._records)
