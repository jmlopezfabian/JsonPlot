"""What an agent needs in order to write contracts that work.

The repair loop lives outside the framework — whoever orchestrates the agent
decides how many retries to allow — but the framework has to make it possible:
validate without rendering, and return structured errors. That is all this
module is.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pandas as pd

from .api import describe_dataframe, plot, validate
from .binding.errors import SpecError
from .spec.briefing import SECTION_NAMES, contract
from .spec.schema import capability_summary, json_schema, tool_definition

__all__ = [
    "context", "contract", "columns", "json_schema", "tool_definition",
    "capability_summary", "SECTION_NAMES", "repair", "RepairResult",
    "errors_as_json",
]

#: How many repair attempts are worth granting before degrading.
DEFAULT_ATTEMPTS = 2


def context(
    df: pd.DataFrame,
    examples: int = 3,
    sections: tuple[str, ...] | None = None,
) -> str:
    """Everything the model needs before it writes a contract: this data, and
    the contract itself.

    The contract half is generated from the live definitions (see
    `jsonplot.spec.briefing`), so it never describes a version of the framework
    that no longer exists. Narrow it with `sections` when the prompt budget is
    tight, but narrow it gently: `uv run evals ablation` puts dropping any one
    section inside the noise, and cutting down to `("types", "channels",
    "rules")` — with the columns kept — loses 58 of 144 requests. The document
    is redundant, not padded: no one part carries it, and most of it together
    does. The `df` is the exception. Its column list is 677 characters and 36
    requests, the only part this eval can show paying for itself.
    """
    doc = contract(df, include=sections, examples=examples)
    assert isinstance(doc, str)
    return doc


def columns(df: pd.DataFrame, examples: int = 3) -> str:
    """Just the column list.

    Naming the real columns eliminates most COLUMN_NOT_FOUND errors, which are
    by far the most common failure; when the contract is already in the system
    prompt, this is all a turn needs to add.
    """
    desc = describe_dataframe(df, examples)
    lines = [f"DataFrame: {desc['n_rows']} rows, {len(desc['columns'])} columns", ""]
    for name, info in desc["columns"].items():
        bits = [f"{name}: {info['dtype']} ({info['inferred_type']})",
                f"{info['n_unique']} distinct values"]
        if "min" in info:
            bits.append(f"range {info['min']} – {info['max']}")
        if info.get("examples"):
            bits.append("e.g. " + ", ".join(repr(v) for v in info["examples"]))
        if info["n_null"]:
            bits.append(f"{info['n_null']} nulls")
        lines.append("  - " + "; ".join(bits))
    return "\n".join(lines)


class RepairResult:
    """The outcome of a repair loop: what came out and what it cost."""

    def __init__(self, figure=None, spec=None, attempts=0, history=None):
        self.figure = figure
        self.spec = spec
        self.attempts = attempts
        self.history: list[list[SpecError]] = history or []

    @property
    def ok(self) -> bool:
        return self.figure is not None

    def report(self) -> str:
        if self.ok:
            return f"figure produced in {self.attempts} attempt(s)"
        return ("could not produce a figure; last error:\n"
                + "\n".join(f"  · {e}" for e in (self.history[-1] if self.history else [])))


def repair(
    spec: Any,
    df: pd.DataFrame,
    fix: Callable[[Any, list[dict]], Any],
    max_attempts: int = DEFAULT_ATTEMPTS,
) -> RepairResult:
    """Validate, and when it fails ask `fix` for a corrected contract. N times.

    `fix(spec, errors)` receives the contract that failed and the errors already
    serialized to dicts — ready to drop into a prompt — and returns the next
    contract. Returning the same one burns an attempt and ends the loop.
    """
    history: list[list[SpecError]] = []
    current = spec
    for attempt in range(1, max_attempts + 2):
        errors = validate(current, df)
        if not errors:
            return RepairResult(plot(current, df), current, attempt, history)
        history.append(errors)
        if attempt > max_attempts:
            break
        current = fix(current, [e.to_dict() for e in errors])
    return RepairResult(None, current, max_attempts + 1, history)


def errors_as_json(errors: list[SpecError], indent: int = 2) -> str:
    """The errors exactly as they go back to the model."""
    return json.dumps([e.to_dict() for e in errors], ensure_ascii=False, indent=indent)
