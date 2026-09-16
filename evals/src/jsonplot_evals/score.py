"""From a model's text to an outcome.

Two questions, in order. Is the contract valid — does `jp.validate` accept it?
And is it right — does it draw what was asked? The second is answered by the
plot frame: `jp.build_frame` reduces any dialect to the exact rows that will be
drawn, keyed by visual role, so a flat contract, a canonical one and a
Vega-Lite one that mean the same chart compare equal, and a `sum` where a
`mean` was asked for does not.

The verdict comes from the frames. The mismatch codes that explain it come from
comparing the resolved contracts, and exist so an ablation can say which kind of
mistake came back — not only that one did.

Scoring reads nothing but the recorded text, the item and the frame, so the
same recording scored by two versions of the framework shows exactly what the
framework change did.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

import jsonplot as jp
from jsonplot.spec.models import CHANNELS, Spec

from .dataset import Item

# Pseudo-codes: never produced by `jp`, which only knows whether a contract can
# be drawn, not whether it is the chart that was asked for.

#: No JSON object in the answer.
NOT_JSON = "NOT_JSON"
#: A request the framework cannot serve came back as a contract that validates.
ACCEPTED_IMPOSSIBLE = "ACCEPTED_IMPOSSIBLE"
WRONG_VIZ_TYPE = "WRONG_VIZ_TYPE"
#: A channel the answer needed is missing, or one it did not need is present.
WRONG_CHANNELS = "WRONG_CHANNELS"
WRONG_FIELD = "WRONG_FIELD"
WRONG_AGGREGATE = "WRONG_AGGREGATE"
WRONG_TIME_UNIT = "WRONG_TIME_UNIT"
WRONG_FILTER = "WRONG_FILTER"
#: Sorting or limiting: the wrong rows, or the right rows in the wrong order.
WRONG_ORDER = "WRONG_ORDER"
#: The plotted data differs and none of the above says why.
WRONG_FRAME = "WRONG_FRAME"
#: Orientation, stacking, or a property the item checks (a requested title).
WRONG_STYLE = "WRONG_STYLE"

SEMANTIC_CODES = (ACCEPTED_IMPOSSIBLE, WRONG_VIZ_TYPE, WRONG_CHANNELS, WRONG_FIELD,
                  WRONG_AGGREGATE, WRONG_TIME_UNIT, WRONG_FILTER, WRONG_ORDER,
                  WRONG_FRAME, WRONG_STYLE)

#: Relative tolerance on plotted numbers: a float summed in another order.
RTOL = 1e-6


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
    #: what the validator said, as `SpecError` dicts
    errors: list[dict] = field(default_factory=list)
    #: why a valid contract is not the right one, in the same shape
    mismatches: list[dict] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        """Valid, and draws what was asked."""
        return self.valid and self.expected_valid and not self.mismatches

    @property
    def scored(self) -> bool:
        """The right outcome: the right chart when one was possible, a
        rejection when not."""
        return self.correct if self.expected_valid else not self.valid

    @property
    def codes(self) -> list[str]:
        return [e["code"] for e in self.errors + self.mismatches]


def score(item: Item, raw: str, df: pd.DataFrame) -> Outcome:
    spec = as_json(raw)
    if spec is None:
        return Outcome(parsed=False, valid=False, expected_valid=item.expected_valid,
                       errors=[_issue(NOT_JSON, "", raw[:120])])
    errors = [e.to_dict() for e in jp.validate(spec, df)]
    outcome = Outcome(parsed=True, valid=not errors, expected_valid=item.expected_valid,
                      spec=spec, errors=errors)
    if errors:
        return outcome
    if not item.expected_valid:
        outcome.mismatches = [_issue(ACCEPTED_IMPOSSIBLE, "viz_type",
                                     "the request cannot be drawn, and this contract "
                                     "validated anyway: it draws something else")]
    else:
        outcome.mismatches = compare(item, spec, df)
    return outcome


def compare(item: Item, contract, df: pd.DataFrame) -> list[dict]:
    """Why `contract` answers none of the item's golds; empty when it answers one.

    With several golds, the mismatches reported are those against the closest.
    """
    pred = _resolve(contract, df)
    closest: list[dict] | None = None
    for gold in item.gold:
        found = _differences(pred, _resolve(gold, df), item.check)
        if not found:
            return []
        if closest is None or len(found) < len(closest):
            closest = found
    return closest or []


# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Resolved:
    spec: Spec
    dump: dict
    frame: pd.DataFrame


def _resolve(contract, df: pd.DataFrame) -> _Resolved:
    spec = jp.resolve(contract, df).spec
    dump = spec.model_dump(mode="json", exclude_none=True)
    # A histogram's bin count is presentation: the same distribution drawn
    # coarser. Everything else that changes the frame changes the answer.
    if spec.viz_type == "hist" and "x" in dump["encoding"]:
        dump["encoding"]["x"].pop("bin", None)
    return _Resolved(spec, dump, jp.build_frame(dump, df))


def _differences(pred: _Resolved, gold: _Resolved, check: dict) -> list[dict]:
    found: list[dict] = []
    p, g = pred.spec, gold.spec
    if p.viz_type != g.viz_type:
        found.append(_issue(WRONG_VIZ_TYPE, "viz_type",
                            f"expected `{g.viz_type}`, got `{p.viz_type}`"))

    ordered = g.data.sort is not None
    if not _same_frame(pred.frame, gold.frame, ordered):
        explained = _explain(p, g)
        if not explained and _same_frame(pred.frame, gold.frame, ordered=False):
            explained = [_issue(WRONG_ORDER, "data.sort",
                                "the right rows, in the wrong order")]
        found += explained or [_issue(WRONG_FRAME, "",
                                      "the plotted data differs from what was asked")]

    if p.style.orientation != g.style.orientation:
        found.append(_issue(WRONG_STYLE, "style.orientation",
                            f"expected `{g.style.orientation}`"))
    if ("color" in gold.frame.columns and g.viz_type in ("bar", "area")
            and p.style.stacked != g.style.stacked):
        found.append(_issue(WRONG_STYLE, "style.stacked", f"expected `{g.style.stacked}`"))

    for path, expected in check.items():
        actual = _get(pred.dump, path)
        if not _same_value(actual, expected):
            found.append(_issue(WRONG_STYLE, path,
                                f"expected {expected!r}, got {actual!r}"))
    return found


def _explain(p: Spec, g: Spec) -> list[dict]:
    """The contract-level differences that plausibly change the frame."""
    found: list[dict] = []
    for role in CHANNELS:
        pc, gc = getattr(p.encoding, role), getattr(g.encoding, role)
        if pc is None and gc is None:
            continue
        path = f"encoding.{role}"
        if pc is None or gc is None:
            what = "missing" if pc is None else "not asked for"
            found.append(_issue(WRONG_CHANNELS, path, f"`{role}` is {what}"))
            continue
        if pc.field != gc.field:
            found.append(_issue(WRONG_FIELD, f"{path}.field",
                                f"expected `{gc.field}`, got `{pc.field}`"))
        if pc.aggregate != gc.aggregate:
            found.append(_issue(WRONG_AGGREGATE, f"{path}.aggregate",
                                f"expected `{gc.aggregate}`, got `{pc.aggregate}`"))
        if pc.time_unit != gc.time_unit:
            found.append(_issue(WRONG_TIME_UNIT, f"{path}.time_unit",
                                f"expected `{gc.time_unit}`, got `{pc.time_unit}`"))
    if _filters(p) != _filters(g):
        found.append(_issue(WRONG_FILTER, "data.filters",
                            "the rows kept are not the ones asked for"))
    # A sort the request never asked for is presentation: it reorders marks
    # without changing which ones are drawn. A different limit always changes
    # them, and a sort that *was* asked for is part of the answer.
    if p.data.limit != g.data.limit:
        found.append(_issue(WRONG_ORDER, "data.limit",
                            f"expected {g.data.limit!r} categories, got {p.data.limit!r}"))
    elif g.data.sort is not None and p.data.sort != g.data.sort:
        found.append(_issue(WRONG_ORDER, "data.sort",
                            "the order the request asked for is not the one drawn"))
    return found


def _same_frame(a: pd.DataFrame, b: pd.DataFrame, ordered: bool) -> bool:
    if set(a.columns) != set(b.columns) or len(a) != len(b):
        return False
    cols = list(b.columns)
    a, b = a[cols].reset_index(drop=True), b.reset_index(drop=True)
    if not ordered:
        a = a.sort_values(cols, kind="mergesort").reset_index(drop=True)
        b = b.sort_values(cols, kind="mergesort").reset_index(drop=True)
    for col in cols:
        x, y = a[col], b[col]
        if _numeric(x) and _numeric(y):
            if not np.allclose(x.to_numpy(float), y.to_numpy(float),
                               rtol=RTOL, atol=1e-9, equal_nan=True):
                return False
        elif not (x.astype(str).to_numpy() == y.astype(str).to_numpy()).all():
            return False
    return True


def _numeric(s: pd.Series) -> bool:
    return is_numeric_dtype(s) and not is_bool_dtype(s)


def _filters(spec: Spec) -> list[dict]:
    return sorted((f.model_dump(mode="json") for f in spec.data.filters),
                  key=lambda f: json.dumps(f, sort_keys=True))


def _get(tree: dict, path: str):
    node = tree
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _same_value(actual, expected) -> bool:
    if isinstance(actual, str) and isinstance(expected, str):
        return actual.strip().casefold() == expected.strip().casefold()
    return actual == expected


def _issue(code: str, path: str, message: str) -> dict:
    return {"code": code, "path": path, "message": message}
