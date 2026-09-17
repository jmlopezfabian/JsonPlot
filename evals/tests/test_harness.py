"""The harness, checked without a model in the room."""

from __future__ import annotations

import json

import pytest

import jsonplot as jp
from jsonplot_evals import ROOT, cli, conditions, dataset, frames, report, tracing
from jsonplot_evals.dataset import Item
from jsonplot_evals.providers import Completion, for_model
from jsonplot_evals.runner import ReplayMiss, Runner
from jsonplot_evals.score import as_json, score
from jsonplot_evals.store import Record, Store

#: The twelve requests the README's numbers were first measured on.
LEGACY_TASKS = {"bar_simple", "bar_top_n", "line_time", "line_series", "scatter",
                "hist", "box", "filtered", "facet", "horizontal", "stacked",
                "impossible"}


@pytest.fixture(scope="module")
def ds():
    return dataset.load("v1")


# -- the dataset ------------------------------------------------------------


def test_the_readme_split_is_the_legacy_twelve(ds):
    ids = {i.id.removeprefix("readme.") for i in ds.select(splits=["readme"])}
    assert ids == LEGACY_TASKS


def test_every_gold_contract_draws_on_its_frame(ds):
    for item in ds.items:
        df = frames.load(item.frame)
        for gold in item.gold:
            assert jp.validate(gold, df) == [], item.id
            assert len(jp.build_frame(gold, df)), item.id


def test_every_gold_scores_right_against_its_own_item(ds):
    """The scorer's own integrity check: a gold answer has to score as right,
    `check` included. A gold that does not is a gold that is wrong."""
    for item in ds.items:
        df = frames.load(item.frame)
        for gold in item.gold:
            outcome = score(item, json.dumps(gold), df)
            assert outcome.correct, (item.id, outcome.mismatches)


def test_a_changed_frame_refuses_to_load(tmp_path, monkeypatch):
    version = tmp_path / "v1"
    version.mkdir()
    (version / "manifest.json").write_text(
        json.dumps({"version": "v1", "frames": {"sales": "0000000000000000"}}))
    (version / "requests.jsonl").write_text("")
    monkeypatch.setattr(dataset, "DATA", tmp_path)
    with pytest.raises(dataset.DatasetError, match="changed"):
        dataset.load("v1")


# -- scoring ----------------------------------------------------------------


@pytest.mark.parametrize("text", [
    '{"viz_type": "hist"}',
    '```json\n{"viz_type": "hist"}\n```',
    'Here you go: {"viz_type": "hist"} hope it helps',
])
def test_json_is_recovered_from_chatty_answers(text):
    assert as_json(text) == {"viz_type": "hist"}


def test_a_rejected_impossible_request_is_the_right_outcome():
    item = Item(id="x", split="t", frame="sales", request="3D", expect="reject")
    outcome = score(item, '{"viz_type": "surface3d"}', frames.sales())
    assert not outcome.valid and outcome.scored


# -- valid is not the same as right -----------------------------------------


@pytest.fixture(scope="module")
def sales():
    return frames.sales()


def item(ds, name):
    return next(i for i in ds.items if i.id == name)


def outcome_for(ds, name, contract, sales):
    return score(item(ds, name), json.dumps(contract), sales)


def test_the_same_chart_in_vega_lite_is_the_same_answer(ds, sales):
    """Swapping the channels is how Vega-Lite spells a horizontal bar."""
    got = outcome_for(ds, "readme.horizontal", {
        "mark": "bar",
        "encoding": {"x": {"field": "units", "type": "Q", "aggregate": "sum"},
                     "y": {"field": "region", "type": "N"}},
        "title": "Units"}, sales)
    assert got.correct, got.mismatches


def test_a_mean_where_a_sum_was_asked_for_is_valid_and_wrong(ds, sales):
    got = outcome_for(ds, "readme.bar_simple", {
        "viz_type": "bar", "x_axis": "region", "y_axis": "revenue", "agg": "mean"}, sales)
    assert got.valid and not got.correct
    assert "WRONG_AGGREGATE" in got.codes


def test_a_missing_series_is_wrong(ds, sales):
    got = outcome_for(ds, "readme.line_series", {
        "viz_type": "line",
        "encoding": {"x": {"field": "date", "time_unit": "month"},
                     "y": {"field": "revenue", "aggregate": "sum"}}}, sales)
    assert not got.correct and "WRONG_CHANNELS" in got.codes


def test_how_many_bins_a_histogram_uses_is_not_semantics(ds, sales):
    got = outcome_for(ds, "readme.hist", {
        "viz_type": "hist", "encoding": {"x": {"field": "satisfaction", "bin": 30}}}, sales)
    assert got.correct, got.mismatches


def test_the_right_rows_in_the_wrong_order_is_wrong(ds, sales):
    got = outcome_for(ds, "readme.bar_top_n", {
        "viz_type": "bar",
        "encoding": {"x": {"field": "region"}, "y": {"field": "revenue", "aggregate": "sum"}},
        "data": {"sort": {"by": "y", "order": "asc"}, "limit": 3}}, sales)
    assert not got.correct and "WRONG_ORDER" in got.codes


def test_a_requested_title_is_checked(ds, sales):
    got = outcome_for(ds, "readme.horizontal", {
        "viz_type": "bar", "x_axis": "region", "y_axis": "units", "agg": "sum",
        "orientation": "horizontal"}, sales)
    assert not got.correct
    assert [e["path"] for e in got.mismatches] == ["style.title"]


def test_an_impossible_request_answered_with_a_valid_chart_is_wrong(ds, sales):
    got = outcome_for(ds, "readme.impossible", {
        "viz_type": "scatter",
        "encoding": {"x": {"field": "price"}, "y": {"field": "units"}}}, sales)
    assert got.valid and not got.scored and "ACCEPTED_IMPOSSIBLE" in got.codes


# -- recording and replay ---------------------------------------------------


class Scripted:
    """A provider that answers from a script and counts the calls."""

    name = "scripted"

    def __init__(self, *answers: str, model: str = "fake"):
        self.model = model
        self.params = {}
        self.answers = list(answers)
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return Completion(self.answers.pop(0), 10, 5, 0.01)


GOOD = '{"viz_type": "hist", "x_axis": "satisfaction"}'
BAD = '{"viz_type": "hist", "x_axis": "Satisfaction"}'
ITEM = Item(id="t.hist", split="t", frame="sales", request="Distribution of satisfaction.",
            gold=({"viz_type": "hist", "x": "satisfaction"},))


def _runner(provider, tmp_path, mode="auto", condition="briefing"):
    runner = Runner(provider, Store(tmp_path / "rec.jsonl"), mode=mode)
    return runner, conditions.get(condition)


def test_a_recorded_answer_is_not_asked_for_twice(tmp_path):
    first = Scripted(GOOD)
    runner, c = _runner(first, tmp_path)
    assert runner.one(ITEM, c).called == 1

    again = Scripted()
    runner, c = _runner(again, tmp_path, mode="replay")
    result = runner.one(ITEM, c)
    assert again.calls == 0 and result.outcome.valid and not result.stale


def test_replay_fails_loudly_on_a_missing_answer(tmp_path):
    runner, c = _runner(Scripted(), tmp_path, mode="replay")
    with pytest.raises(ReplayMiss):
        runner.one(ITEM, c)


def test_a_changed_prompt_is_stale_in_replay_and_asked_again_live(tmp_path, monkeypatch):
    runner, c = _runner(Scripted(GOOD), tmp_path)
    runner.one(ITEM, c)
    monkeypatch.setattr(conditions, "SYSTEM", "a different system prompt")

    runner, c = _runner(Scripted(), tmp_path, mode="replay")
    assert runner.one(ITEM, c).stale == 1

    live = Scripted(GOOD)
    runner, c = _runner(live, tmp_path)
    assert runner.one(ITEM, c).called == 1


def test_repair_asks_once_and_only_when_invalid(tmp_path):
    fixing = Scripted(BAD, GOOD)
    runner, c = _runner(fixing, tmp_path, condition="briefing+repair")
    result = runner.one(ITEM, c)
    assert (fixing.calls, result.turns, result.outcome.valid) == (2, 2, True)

    right = Scripted(GOOD, model="other")
    runner = Runner(right, Store(tmp_path / "other.jsonl"))
    assert runner.one(ITEM, c).turns == 1 and right.calls == 1


# -- ablating the briefing --------------------------------------------------


@pytest.fixture(scope="module")
def full_briefing(sales):
    return conditions.get("briefing").preamble(sales)


def test_removing_a_section_removes_only_that_section(full_briefing, sales):
    shorter = conditions.get("minus:vega_lite").preamble(sales)
    assert len(shorter) < len(full_briefing)
    assert "## Vega-Lite spellings" not in shorter
    assert "## Rules the validator enforces" in shorter


def test_the_columns_can_be_ablated_like_a_section(full_briefing, sales):
    without = conditions.get("minus:columns").preamble(sales)
    assert "## The data" in full_briefing and "## The data" not in without


def test_a_pair_can_be_removed_together(full_briefing, sales):
    """Leave-one-out cannot see two sections that teach the same thing."""
    pair = conditions.get("minus:shape,flat").preamble(sales)
    assert len(pair) < min(len(conditions.get(f"minus:{s}").preamble(sales))
                           for s in ("shape", "flat"))


def test_only_keeps_what_it_names(sales):
    kept = conditions.get("only:types,channels,rules").preamble(sales)
    assert "## Chart types" in kept and "## The flat dialect" not in kept


def test_an_ablation_that_empties_the_briefing_is_refused():
    """`include=()` means every section to the briefing, so this would quietly
    measure the whole document instead of none of it."""
    with pytest.raises(KeyError, match="no sections"):
        conditions.get("only:columns")
    with pytest.raises(KeyError, match="no sections"):
        conditions.get("minus:" + ",".join(conditions.ABLATABLE))


@pytest.mark.parametrize("name", ["minus:nope", "only:", "briefing+nope"])
def test_an_unknown_ablation_is_refused(name):
    with pytest.raises(KeyError):
        conditions.get(name)


# -- tracing, which is off unless asked for ---------------------------------


@pytest.fixture
def no_langfuse(monkeypatch):
    for name in tracing.REQUIRED:
        monkeypatch.delenv(name, raising=False)


def test_without_langfuse_tracing_does_nothing(no_langfuse, ds):
    """The gate, the tests and a replay must never need a server."""
    tracer = tracing.tracer("v1-baseline", "v1")
    assert not tracer.enabled and not tracing.configured()
    with tracer.item(item(ds, "readme.hist"), "briefing", "a-model") as handle:
        handle.turn(None, "a-model", {}, "the prompt")
        handle.score(None)
    tracer.flush()


def test_asking_for_tracing_without_keys_is_an_error(no_langfuse):
    """Silently not tracing a run someone asked to trace wastes the run."""
    with pytest.raises(RuntimeError, match="LANGFUSE_PUBLIC_KEY"):
        tracing.tracer("v1-baseline", "v1", enabled=True)


def test_a_recording_lists_the_conditions_it_holds(tmp_path):
    """The ablation report reads this to decide what to compare: a condition
    missing from it is omitted from the table without saying so."""
    store = Store(tmp_path / "rec.jsonl")
    for condition in ("minus:flat", "only:types,channels,rules", "minus:flat"):
        store.put(Record(item="i", condition=condition, turn=0, provider="p",
                         model="m", prompt_sha="abc", raw="{}", input_tokens=1,
                         output_tokens=1, seconds=0.0, recorded_at="now"))
    assert store.conditions() == ["minus:flat", "only:types,channels,rules"]


# -- the statistics the ablation rests on -----------------------------------


def flips(lost: int, gained: int, same: int = 0):
    """Two conditions over the same items, differing in exactly this way."""
    full, ablated = {}, {}
    for n in range(lost):
        full[f"lost{n}"], ablated[f"lost{n}"] = True, False
    for n in range(gained):
        full[f"gained{n}"], ablated[f"gained{n}"] = False, True
    for n in range(same):
        full[f"same{n}"], ablated[f"same{n}"] = True, True
    return full, ablated


def test_a_section_that_changes_nothing_is_not_evidence():
    full, ablated = flips(0, 0, same=144)
    assert report.mcnemar(full, ablated) == (0, 0, 1.0)


def test_an_even_split_is_not_evidence():
    """Eight requests lost and nine gained is what noise looks like."""
    lost, gained, p = report.mcnemar(*flips(8, 9))
    assert (lost, gained) == (8, 9) and p == 1.0


def test_a_lopsided_split_is_evidence():
    lost, gained, p = report.mcnemar(*flips(40, 4))
    assert (lost, gained) == (40, 4) and p < 0.0001


def test_the_verdict_needs_both_a_direction_and_a_p_value():
    assert report.verdict(-36, 0.0000, 0.0042) == "pays its rent"
    assert report.verdict(+10, 0.0001, 0.0042) == "gets in the way"
    # significant on its own, not once the level is split across the sections
    assert report.verdict(-9, 0.049, 0.0042) == "noise"


# -- the gate CI runs -------------------------------------------------------


RIGHT_HIST = '{"viz_type": "hist", "x_axis": "satisfaction"}'
WRONG_HIST = '{"viz_type": "hist", "x_axis": "price"}'
GATE_ARGS = ["gate", "--items", "readme.hist", "--conditions", "briefing"]


def _recorded(monkeypatch, tmp_path, ds, answer=None):
    """Point the gate at a recording of this one answer, or at an empty one.

    Seeded through the provider the gate itself resolves, answering from a
    script instead of over the network: the provider's name, model and
    parameters are part of the prompt hash, so a stand-in would look stale for
    reasons that have nothing to do with the prompt.
    """
    store = Store(tmp_path / "rec.jsonl")
    if answer is not None:
        provider = for_model(cli.DEFAULT_MODEL)
        provider.complete = lambda system, user: Completion(answer, 10, 5, 0.01)
        Runner(provider, store).one(item(ds, "readme.hist"), conditions.get("briefing"))
    monkeypatch.setattr(cli.Store, "for_run",
                        classmethod(lambda cls, run, model, root=None: store))
    return store


def test_the_gate_fails_when_an_answer_was_never_recorded(monkeypatch, tmp_path, ds):
    _recorded(monkeypatch, tmp_path, ds)
    assert cli.main(GATE_ARGS) == 3


def test_the_gate_fails_when_the_briefing_moved(monkeypatch, tmp_path, ds, capsys):
    """The failure this gate exists for: a prompt edited without re-running the
    eval, which is how a published number goes stale while the tests pass."""
    _recorded(monkeypatch, tmp_path, ds, RIGHT_HIST)
    monkeypatch.setattr(conditions, "SYSTEM", "a different system prompt")
    assert cli.main(GATE_ARGS) == 1
    assert "since changed" in capsys.readouterr().err


def test_the_gate_fails_below_the_floor(monkeypatch, tmp_path, ds, capsys):
    _recorded(monkeypatch, tmp_path, ds, WRONG_HIST)
    floors = tmp_path / "thresholds.toml"
    floors.write_text('[v1-baseline."qwen2.5:7b-instruct"]\nbriefing = 1\n')
    assert cli.main([*GATE_ARGS, "--thresholds", str(floors)]) == 1
    # for the floor, not because the recording looked stale
    assert "below the floor" in capsys.readouterr().err


def test_the_gate_passes_when_the_recording_still_holds(monkeypatch, tmp_path, ds):
    _recorded(monkeypatch, tmp_path, ds, RIGHT_HIST)
    floors = tmp_path / "thresholds.toml"
    floors.write_text('[v1-baseline."qwen2.5:7b-instruct"]\nbriefing = 1\n')
    assert cli.main([*GATE_ARGS, "--thresholds", str(floors)]) == 0


# -- the promise in the README ----------------------------------------------


def test_the_readme_table_is_what_the_recording_replays(capsys):
    assert cli.main(["--quiet", "--metric", "right", "--split", "readme", "core"]) == 0
    table = capsys.readouterr().out.strip()
    readme = (ROOT.parent / "README.md").read_text(encoding="utf-8")
    assert table in readme, f"README.md no longer shows what `uv run evals` prints:\n{table}"
