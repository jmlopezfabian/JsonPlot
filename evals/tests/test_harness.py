"""The harness, checked without a model in the room."""

from __future__ import annotations

import json

import pytest

import jsonplot as jp
from jsonplot_evals import ROOT, cli, conditions, dataset, frames
from jsonplot_evals.dataset import Item
from jsonplot_evals.providers import Completion
from jsonplot_evals.runner import ReplayMiss, Runner
from jsonplot_evals.score import as_json, score
from jsonplot_evals.store import Store

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


# -- the promise in the README ----------------------------------------------


def test_the_readme_table_is_what_the_recording_replays(capsys):
    assert cli.main(["--quiet", "--metric", "right"]) == 0
    table = capsys.readouterr().out.strip()
    readme = (ROOT.parent / "README.md").read_text(encoding="utf-8")
    assert table in readme, f"README.md no longer shows what `uv run evals` prints:\n{table}"
