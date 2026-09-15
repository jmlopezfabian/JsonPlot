"""The dataset: requests, what a right answer looks like, and which frame.

A version is a directory under `evals/data/` and is never edited once results
have been published against it — fixing a gold contract means a new version.
Loading is strict for the same reason the contract is: a typo in an item should
fail here, not show up weeks later as a model that looks worse than it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from . import ROOT, frames

DATA = ROOT / "data"

Expect = Literal["valid", "reject"]
_ITEM_KEYS = {"id", "split", "frame", "request", "expect", "tags", "gold", "check"}


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Item:
    id: str
    split: str
    frame: str
    request: str
    #: `reject` marks a request the framework cannot serve: a contract that
    #: fails validation is the right answer there.
    expect: Expect = "valid"
    tags: tuple[str, ...] = ()
    #: Contracts that answer the request, any dialect. More than one when the
    #: request is genuinely ambiguous.
    gold: tuple[dict, ...] = ()
    #: Properties the answer must have beyond the plot frame (a requested title).
    check: dict = field(default_factory=dict)

    @property
    def expected_valid(self) -> bool:
        return self.expect == "valid"


@dataclass(frozen=True)
class Dataset:
    version: str
    items: tuple[Item, ...]
    #: frame name -> fingerprint the gold contracts were written against
    frames: dict[str, str]

    def select(self, splits=None, ids=None) -> list[Item]:
        chosen = [i for i in self.items
                  if (not splits or i.split in splits) and (not ids or i.id in ids)]
        if ids:
            unknown = set(ids) - {i.id for i in chosen}
            if unknown:
                raise DatasetError(f"unknown item id(s): {sorted(unknown)}")
        return chosen

    @property
    def splits(self) -> list[str]:
        return list(dict.fromkeys(i.split for i in self.items))


def load(version: str = "v1") -> Dataset:
    base = DATA / version
    if not base.is_dir():
        available = sorted(p.name for p in DATA.iterdir() if p.is_dir())
        raise DatasetError(f"no dataset {version!r}; available: {available}")
    manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    items = tuple(_items(base / "requests.jsonl"))
    _check(items, manifest)
    return Dataset(version=manifest["version"], items=items, frames=manifest["frames"])


def _items(path: Path):
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        extra = set(raw) - _ITEM_KEYS
        if extra:
            raise DatasetError(f"{path.name}:{n}: unknown key(s) {sorted(extra)}")
        try:
            yield Item(
                id=raw["id"], split=raw["split"], frame=raw["frame"],
                request=raw["request"], expect=raw.get("expect", "valid"),
                tags=tuple(raw.get("tags", ())), gold=tuple(raw.get("gold", ())),
                check=raw.get("check", {}),
            )
        except KeyError as exc:
            raise DatasetError(f"{path.name}:{n}: missing key {exc}") from None


def _check(items: tuple[Item, ...], manifest: dict) -> None:
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise DatasetError(f"duplicate id {item.id!r}")
        seen.add(item.id)
        if item.frame not in manifest["frames"]:
            raise DatasetError(f"{item.id}: frame {item.frame!r} is not in the manifest")
        if item.expect not in ("valid", "reject"):
            raise DatasetError(f"{item.id}: expect must be 'valid' or 'reject'")
        if item.expect == "reject" and item.gold:
            raise DatasetError(f"{item.id}: a request expected to be rejected has no gold")
        if item.expect == "valid" and not item.gold:
            raise DatasetError(f"{item.id}: a request expected to be valid needs a gold")

    for name, pinned in manifest["frames"].items():
        actual = frames.fingerprint(frames.load(name))
        if actual != pinned:
            raise DatasetError(
                f"frame {name!r} changed (fingerprint {actual}, dataset pinned "
                f"{pinned}); the gold contracts were written against the old one")
