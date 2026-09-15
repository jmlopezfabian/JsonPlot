"""Results, written out for a terminal and for the README."""

from __future__ import annotations

from collections import Counter

from .runner import Result


def result_line(r: Result) -> str:
    mark = "ok  " if r.outcome.scored else "FAIL"
    note = "" if not r.outcome.errors else "  " + "; ".join(
        f"{e['code']}@{e.get('path', '')}" for e in r.outcome.errors[:2])
    if not r.outcome.expected_valid:
        note += "   (rejection is the expected outcome)"
    source = "" if r.called else "  ·rec" + ("·stale" if r.stale else "")
    return f"   {mark} {r.item.id:<24} {r.seconds:5.1f}s{source}{note}"


def condition_footer(results: list[Result]) -> str:
    ok = sum(r.outcome.scored for r in results)
    secs = sum(r.seconds for r in results)
    lines = [f"   → {ok}/{len(results)} right outcome  ({secs:.0f}s of model time)"]
    codes = Counter(c for r in results for c in r.outcome.codes)
    if codes:
        lines.append("     " + ", ".join(f"{c}×{n}" for c, n in codes.most_common(4)))
    return "\n".join(lines)


def table(results: list[Result], conditions: list[str]) -> str:
    """Models as rows, conditions as columns, `right/total` in each cell."""
    models = list(dict.fromkeys(r.model for r in results))
    cells: dict[tuple[str, str], list[Result]] = {}
    for r in results:
        cells.setdefault((r.model, r.condition), []).append(r)
    lines = ["| model | " + " | ".join(conditions) + " |",
             "| --- |" + " ---: |" * len(conditions)]
    for model in models:
        row = []
        for c in conditions:
            got = cells.get((model, c), [])
            row.append(f"{sum(r.outcome.scored for r in got)}/{len(got)}" if got else "—")
        lines.append(f"| `{model}` | " + " | ".join(row) + " |")
    return "\n".join(lines)


def detail(results: list[Result]) -> list[dict]:
    return [{"model": r.model, "condition": r.condition, "item": r.item.id,
             "valid": r.outcome.valid, "expected_valid": r.outcome.expected_valid,
             "scored": r.outcome.scored, "turns": r.turns, "stale": r.stale,
             "errors": r.outcome.errors, "spec": r.outcome.spec}
            for r in results]
