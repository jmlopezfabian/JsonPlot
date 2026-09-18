"""Phase 01: the shape of the contract."""

import pytest

from jsonplot.binding.errors import Code
from jsonplot.spec import parse_spec

VALID = [
    {"viz_type": "bar", "x_axis": "region", "y_axis": "revenue", "agg": "sum"},
    {"viz_type": "bar", "encoding": {"x": {"field": "a"}, "y": {"field": "b"}}},
    {"viz_type": "line", "x_axis": "date", "y_axis": "v", "time_unit": "month"},
    {"viz_type": "scatter", "x": "a", "y": "b", "size_by": "c"},
    {"viz_type": "hist", "x_axis": "a", "bins": 20},
    {"viz_type": "box", "encoding": {"y": {"field": "a"}}},
    {"viz_type": "bar", "x_axis": "a", "y_axis": "b", "sort": "-y", "limit": 5},
    {"viz_type": "bar", "x_axis": "a", "y_axis": "b",
     "filters": [{"field": "c", "op": "in", "value": ["x", "y"]}]},
    {"viz_type": "line", "x_axis": "a", "y_axis": "b", "facet_by": "c"},
    {"viz_type": "bar", "x_axis": "a", "y_axis": "b", "stacked": True,
     "orientation": "horizontal"},
    {"viz_type": "barplot", "x_axis": "a", "y_axis": "b"},           # synonym
    # bar, not area: seaborn has no area renderer, and a contract naming a pair
    # this installation cannot draw is now rejected here rather than at render
    {"version": "1.0", "viz_type": "bar", "backend": "seaborn",
     "encoding": {"x": {"field": "a"}, "y": {"field": "b", "aggregate": "mean"}},
     "style": {"palette": "safe3"}, "output": {"format": "png", "dpi": 200}},
]

INVALID = [
    ({"x_axis": "a", "y_axis": "b"}, Code.MISSING_FIELD, "viz_type"),
    ({"viz_type": "barr", "x_axis": "a", "y_axis": "b"}, Code.INVALID_VALUE, "viz_type"),
    ({"viz_type": "bar", "x_axi": "a", "y_axis": "b"}, Code.UNKNOWN_FIELD, "x_axi"),
    ({"viz_type": "bar", "encoding": {"x": {"colum": "a"}}}, Code.UNKNOWN_FIELD,
     "encoding.x.colum"),
    ({"viz_type": "bar", "x_axis": "a", "y_axis": "b", "agg": "avg"},
     Code.INVALID_VALUE, "encoding.y.aggregate"),
    ({"viz_type": "bar", "encoding": {"x": {"field": "a", "type": "numeric"}}},
     Code.INVALID_VALUE, "encoding.x.type"),
    ({"viz_type": "bar", "x_axis": "a", "y_axis": "b",
      "filters": [{"field": "c", "op": "gt"}]}, Code.INVALID_VALUE, "data.filters[0]"),
    ({"viz_type": "bar", "x_axis": "a", "y_axis": "b",
      "filters": [{"field": "c", "op": "in", "value": 3}]}, Code.INVALID_VALUE,
     "data.filters[0]"),
    ({"viz_type": "bar", "x_axis": "a", "y_axis": "b", "limit": 0},
     Code.INVALID_VALUE, "data.limit"),
    ({"viz_type": "bar", "x_axis": "a", "y_axis": "b", "backend": "plotly"},
     Code.INVALID_VALUE, "backend"),
    ({"version": "2.0", "viz_type": "bar"}, Code.UNSUPPORTED_VERSION, "version"),
    ({"viz_type": "bar", "encoding": {"x": {"type": "nominal"}}},
     Code.INVALID_VALUE, "encoding.x"),
]


@pytest.mark.parametrize("raw", VALID)
def test_valid_contracts(raw):
    spec, errors = parse_spec(raw)
    assert errors == []
    assert spec is not None


@pytest.mark.parametrize("raw,code,path", INVALID)
def test_invalid_contracts(raw, code, path):
    spec, errors = parse_spec(raw)
    assert spec is None
    assert any(e.code == code and e.path == path for e in errors), \
        f"expected {code} at {path}, got {[(e.code, e.path) for e in errors]}"


def test_both_dialects_produce_the_same_spec():
    flat = {"viz_type": "bar", "x_axis": "date", "y_axis": "revenue",
            "group_by": "region", "agg": "sum", "title": "T"}
    canonical = {"viz_type": "bar", "encoding": {
        "x": {"field": "date"}, "y": {"field": "revenue", "aggregate": "sum"},
        "color": {"field": "region"}}, "style": {"title": "T"}}
    a, _ = parse_spec(flat)
    b, _ = parse_spec(canonical)
    assert a == b


def test_the_canonical_block_beats_the_flat_shortcut():
    spec, _ = parse_spec({"viz_type": "bar", "x_axis": "a", "y_axis": "b",
                          "agg": "sum",
                          "encoding": {"y": {"field": "b", "aggregate": "mean"}}})
    assert spec.encoding.y.aggregate == "mean"


def test_an_unknown_key_suggests_the_close_one():
    _, errors = parse_spec({"viz_type": "bar", "x_axsi": "a", "y_axis": "b"})
    assert "x_axis" in errors[0].did_you_mean


def test_a_contract_given_as_text():
    spec, errors = parse_spec('{"viz_type": "bar", "x_axis": "a", "y_axis": "b"}')
    assert errors == [] and spec.viz_type == "bar"


def test_malformed_json():
    _, errors = parse_spec('{"viz_type": "bar",}')
    assert errors[0].code == Code.INVALID_JSON
