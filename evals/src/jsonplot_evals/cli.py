"""`uv run evals` — put the dataset through models and score what comes back.

With no arguments it replays the committed baseline on the README split, so it
prints the README's table on a machine with no model installed:

    uv run evals                                        # the README table
    uv run evals run --model qwen2.5:7b-instruct --live # ask the model again
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from . import ROOT, conditions, dataset, report
from .providers import ProviderUnavailable, for_model
from .runner import ReplayMiss, Runner
from .store import Store

DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_SPLIT = "readme"
THRESHOLDS = ROOT / "thresholds.toml"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0].startswith("-"):
        argv = ["run", *argv]

    ap = argparse.ArgumentParser(prog="evals", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run models over the dataset and score them")
    run.add_argument("--dataset", default="v1")
    run.add_argument("--split", nargs="*", default=[DEFAULT_SPLIT],
                     help="splits to run (default: readme)")
    run.add_argument("--items", nargs="*", help="run only these item ids")
    run.add_argument("--model", nargs="*", default=[DEFAULT_MODEL])
    run.add_argument("--conditions", nargs="*", default=list(conditions.CONDITIONS))
    run.add_argument("--run", default=None,
                     help="recording to read and write (default: <dataset>-baseline)")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--live", dest="mode", action="store_const", const="auto",
                      help="call the model where the recording does not answer the "
                           "current prompt")
    mode.add_argument("--fresh", dest="mode", action="store_const", const="fresh",
                      help="call the model for everything and record over the old answers")
    run.set_defaults(mode="replay")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--metric", choices=[*report.METRICS, "both"], default="both",
                     help="which table to print (default: both)")
    run.add_argument("--out", type=Path, help="write the per-item detail here as JSON")
    run.add_argument("-q", "--quiet", action="store_true", help="only print the table")

    gate = sub.add_parser("gate", help="fail the build when the recorded numbers slip")
    gate.add_argument("--dataset", default="v1")
    gate.add_argument("--split", nargs="*", default=["readme", "core"])
    gate.add_argument("--items", nargs="*", help="gate only these item ids")
    gate.add_argument("--model", nargs="*", default=[DEFAULT_MODEL])
    gate.add_argument("--conditions", nargs="*", default=["briefing", "briefing+repair"])
    gate.add_argument("--run", default=None)
    gate.add_argument("--thresholds", type=Path, default=THRESHOLDS)

    args = ap.parse_args(argv)
    return _gate(args) if args.command == "gate" else _run(args)


def _run(args) -> int:
    ds = dataset.load(args.dataset)
    items = ds.select(splits=args.split, ids=args.items)
    chosen = [conditions.get(c) for c in args.conditions]
    run_name = args.run or f"{ds.version}-baseline"
    # flushed: an overnight run writes to a file, and progress should show there
    say = (lambda *_: None) if args.quiet else (lambda *a: print(*a, flush=True))

    results = []
    for model in args.model:
        try:
            provider = for_model(model)
        except ProviderUnavailable as exc:
            print(f"evals: {exc}", file=sys.stderr)
            return 2
        store = Store.for_run(run_name, model)
        runner = Runner(provider, store, mode=args.mode)
        say(f"model {model} · dataset {ds.version} · {len(items)} items · "
            f"mode {args.mode} · recording {_shown(store.path)}\n")
        for condition in chosen:
            say(f"── {condition.name}")
            got = []
            try:
                for result in runner.run(items, condition, workers=args.workers):
                    say(report.result_line(result))
                    got.append(result)
            except ProviderUnavailable as exc:
                print(f"evals: {exc}", file=sys.stderr)
                return 2
            except ReplayMiss as exc:
                print(f"evals: {exc}\n       run with --live to ask the model",
                      file=sys.stderr)
                return 3
            say(report.condition_footer(got) + "\n")
            results.extend(got)

    stale = sum(r.stale for r in results)
    metrics = list(report.METRICS) if args.metric == "both" else [args.metric]
    for n, metric in enumerate(metrics):
        if len(metrics) > 1:
            print(f"{'' if n == 0 else chr(10)}{report.HEADINGS[metric]}\n")
        print(report.table(results, [c.name for c in chosen], metric))
    if stale:
        print(f"\n{stale} recorded answer(s) were given to a prompt that has since "
              "changed; run with --live to refresh them.")
    if args.out:
        args.out.write_text(json.dumps(report.detail(results), indent=2, default=str),
                            encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def _gate(args) -> int:
    """Re-score the recorded answers and hold the line, without a model.

    Three ways to fail, and the middle one is the reason this exists. An answer
    can be missing. An answer can have been recorded against a prompt that has
    since changed — a briefing edited without re-running the eval, which is how
    a published number goes stale while every test still passes. Or the score
    can simply have dropped below the floor.
    """
    ds = dataset.load(args.dataset)
    items = ds.select(splits=args.split, ids=args.items)
    chosen = [conditions.get(c) for c in args.conditions]
    run_name = args.run or f"{ds.version}-baseline"
    floors = _floors(args.thresholds, run_name)
    failures: list[str] = []

    for model in args.model:
        store = Store.for_run(run_name, model)
        runner = Runner(for_model(model), store, mode="replay")
        for condition in chosen:
            try:
                got = list(runner.run(items, condition))
            except ReplayMiss as exc:
                print(f"evals: {exc}\n       record it with `uv run evals run --live "
                      f"--split {' '.join(args.split)}` and commit the recording",
                      file=sys.stderr)
                return 3
            stale = sum(r.stale for r in got)
            right = sum(r.outcome.scored for r in got)
            floor = floors.get((model, condition.name))
            state = "ok  "
            if stale:
                state = "STALE"
                failures.append(
                    f"{model} · {condition.name}: {stale} of {len(got)} answers were "
                    f"given to a prompt that has since changed. The briefing moved and "
                    f"the eval was not re-run: `uv run evals run --live --fresh`.")
            elif floor is not None and right < floor:
                state = "UNDER"
                failures.append(f"{model} · {condition.name}: {right}/{len(got)} right, "
                                f"below the floor of {floor}.")
            shown = "no floor" if floor is None else f"floor {floor}"
            print(f"{state} {model} · {condition.name}: {right}/{len(got)} right ({shown})")

    if failures:
        print("\n" + "\n".join(f"· {f}" for f in failures), file=sys.stderr)
        return 1
    print("\nthe recorded answers still hold.")
    return 0


def _floors(path: Path, run: str) -> dict[tuple[str, str], int]:
    """Minimum right outcomes per model and condition, from `thresholds.toml`."""
    if not path.exists():
        return {}
    table = tomllib.loads(path.read_text(encoding="utf-8")).get(run, {})
    return {(model, condition): floor
            for model, conditions_ in table.items()
            for condition, floor in conditions_.items()}


def _shown(path: Path) -> Path:
    cwd = Path.cwd()
    return path.relative_to(cwd) if path.is_relative_to(cwd) else path


if __name__ == "__main__":
    raise SystemExit(main())
