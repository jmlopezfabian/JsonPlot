"""From a model's text to an outcome.

Scoring reads nothing but the recorded text, the item and the frame, so the
same recording scored by two versions of the framework shows exactly what the
framework change did.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import pandas as pd

import jsonplot as jp

from .dataset import Item

#: Pseudo-code for an answer with no JSON object in it. Not a `jp` code: the
#: validator never sees it.
NOT_JSON = "NOT_JSON"


def as_json(text: str):
    """The model's answer as an object, fence and preamble tolerated."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None


@dataclass
class Outcome:
    parsed: bool
    valid: bool
    expected_valid: bool
    spec: object = None
    errors: list[dict] = field(default_factory=list)

    @property
    def scored(self) -> bool:
        """The right outcome: valid when a chart was possible, rejected when not."""
        return self.valid is self.expected_valid

    @property
    def codes(self) -> list[str]:
        return [e["code"] for e in self.errors]


def score(item: Item, raw: str, df: pd.DataFrame) -> Outcome:
    spec = as_json(raw)
    if spec is None:
        return Outcome(parsed=False, valid=False, expected_valid=item.expected_valid,
                       errors=[{"code": NOT_JSON, "path": "", "message": raw[:120]}])
    errors = [e.to_dict() for e in jp.validate(spec, df)]
    return Outcome(parsed=True, valid=not errors, expected_valid=item.expected_valid,
                   spec=spec, errors=errors)
