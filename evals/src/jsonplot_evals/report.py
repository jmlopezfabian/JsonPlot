"""Results, written out for a terminal and for the README."""

from __future__ import annotations

from collections import Counter

from .runner import Result

#: What a cell counts. `right` is the headline: the chart that was asked for,
#: or a rejection where none was possible. `valid` is the weaker question the
#: validator answers on its own, and the gap between them is the point.
METRICS = {
    "right": lambda r: r.outcome.scored,
    "valid": lambda r: r.outcome.valid,
}

HEADINGS = {
    "right": "right outcome — the chart that was asked for, or a rejection where "
             "nothing could be drawn",
    "valid": "valid contract — passes jp.validate, whatever it draws",
}


def result_line(r: Result) -> str:
    o = r.outcome
    mark = "ok  " if o.scored else "FAIL"
    issues = o.errors + o.mismatches
    note = "" if not issues else "  " + "; ".join(
        f"{e['code']}@{e.get('path', '')}" for e in issues[:2])
    if not o.expected_valid:
        note += "   (rejection is the expected outcome)"
    elif o.valid and not o.correct:
        note = "  valid but wrong:" + note
    source = "" if r.called else "  ·rec" + ("·stale" if r.stale else "")
    return f"   {mark} {r.item.id:<24} {r.seconds:5.1f}s{source}{note}"


def condition_footer(results: list[Result]) -> str:
    right = sum(r.outcome.scored for r in results)
    valid = sum(r.outcome.valid for r in results)
    secs = sum(r.seconds for r in results)
    lines = [f"   → {right}/{len(results)} right outcome, {valid}/{len(results)} valid "
             f"({secs:.0f}s of model time)"]
    codes = Counter(c for r in results for c in r.outcome.codes)
    if codes:
        lines.append("     " + ", ".join(f"{c}×{n}" for c, n in codes.most_common(4)))
    return "\n".join(lines)


def table(results: list[Result], conditions: list[str], metric: str = "right") -> str:
    """Models as rows, conditions as columns, `counted/total` in each cell."""
    counts = METRICS[metric]
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
            row.append(f"{sum(counts(r) for r in got)}/{len(got)}" if got else "—")
        lines.append(f"| `{model}` | " + " | ".join(row) + " |")
    return "\n".join(lines)


def detail(results: list[Result]) -> list[dict]:
    return [{"model": r.model, "condition": r.condition, "item": r.item.id,
             "valid": r.outcome.valid, "correct": r.outcome.correct,
             "expected_valid": r.outcome.expected_valid, "scored": r.outcome.scored,
             "turns": r.turns, "stale": r.stale, "errors": r.outcome.errors,
             "mismatches": r.outcome.mismatches, "spec": r.outcome.spec}
            for r in results]
