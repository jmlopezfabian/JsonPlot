"""Results, written out for a terminal and for the README."""

from __future__ import annotations

from collections import Counter
from math import comb

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


def mcnemar(full: dict[str, bool], ablated: dict[str, bool]) -> tuple[int, int, float]:
    """How surprising is the change, on the items that actually changed?

    The two conditions answer the same requests with the same model at the same
    temperature, so the items are paired and the only evidence is the items that
    flipped: `lost` were right with the section and wrong without it, `gained`
    the other way. Under the hypothesis that the section makes no difference,
    each flip is a coin toss, and the exact binomial tail says how often a split
    this lopsided would happen anyway.

    Returns (lost, gained, p). A delta of a few items over 144 usually does not
    survive this, which is the point of running it.
    """
    lost = sum(1 for item, right in full.items() if right and not ablated.get(item))
    gained = sum(1 for item, right in full.items() if not right and ablated.get(item))
    flips = lost + gained
    if not flips:
        return lost, gained, 1.0
    tail = sum(comb(flips, i) for i in range(min(lost, gained) + 1)) / 2 ** flips
    return lost, gained, min(1.0, 2 * tail)


def verdict(delta: int, p: float, alpha: float) -> str:
    """What the number is allowed to claim."""
    if p >= alpha:
        return "noise"
    return "pays its rent" if delta < 0 else "gets in the way"


def ablation_table(rows: list[dict], alpha: float) -> str:
    """One line per section, least surprising last."""
    lines = [f"{'section removed':<14} {'Δ right':>8} {'lost':>5} {'gained':>7} {'p':>8} "
             f"{'chars':>6} {'per 1k':>7}  verdict",
             "-" * 78]
    for row in sorted(rows, key=lambda r: r["p"]):
        rent = -row["delta"] / (row["chars"] / 1000) if row["chars"] else 0.0
        lines.append(
            f"{row['section']:<14} {row['delta']:>+8} {row['lost']:>5} {row['gained']:>7} "
            f"{row['p']:>8.4f} {row['chars']:>6} {rent:>7.1f}  "
            f"{verdict(row['delta'], row['p'], alpha)}")
    return "\n".join(lines)


def detail(results: list[Result]) -> list[dict]:
    return [{"model": r.model, "condition": r.condition, "item": r.item.id,
             "valid": r.outcome.valid, "correct": r.outcome.correct,
             "expected_valid": r.outcome.expected_valid, "scored": r.outcome.scored,
             "turns": r.turns, "stale": r.stale, "errors": r.outcome.errors,
             "mismatches": r.outcome.mismatches, "spec": r.outcome.spec}
            for r in results]
