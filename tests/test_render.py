"""Phases 02 and 05: the drawing.

Assertions are on artists and labels, not on pixels: an image test breaks with
any matplotlib version bump and never says what actually went wrong.
"""

import base64

import pytest
from matplotlib.figure import Figure

import jsonplot as jp
from jsonplot.binding.errors import Code, SpecErrorGroup

TYPES = {
    "bar": {"viz_type": "bar", "x_axis": "region", "y_axis": "revenue", "agg": "sum"},
    "line": {"viz_type": "line", "x_axis": "date", "y_axis": "revenue", "agg": "sum",
             "time_unit": "month"},
    "area": {"viz_type": "area", "x_axis": "date", "y_axis": "revenue", "agg": "sum",
             "time_unit": "quarter"},
    "scatter": {"viz_type": "scatter", "x_axis": "price", "y_axis": "revenue"},
    "hist": {"viz_type": "hist", "x_axis": "satisfaction"},
    "box": {"viz_type": "box", "x_axis": "region", "y_axis": "price"},
}


@pytest.mark.parametrize("name", list(TYPES))
def test_every_type_draws(df, name):
    fig = jp.plot(TYPES[name], df)
    assert isinstance(fig, Figure)
    assert fig.axes


def test_one_bar_per_category(df):
    fig = jp.plot(TYPES["bar"], df)
    assert len(fig.axes[0].patches) == df["region"].nunique()


def test_one_bar_per_category_and_series(df):
    fig = jp.plot({**TYPES["bar"], "group_by": "channel"}, df)
    assert len(fig.axes[0].patches) == df["region"].nunique() * df["channel"].nunique()


def test_labels_come_from_the_contract(df):
    ax = jp.plot(TYPES["bar"], df).axes[0]
    assert ax.get_xlabel() == "region"
    assert ax.get_ylabel() == "sum(revenue)"


def test_labels_rotate_with_the_chart(df):
    ax = jp.plot({**TYPES["bar"], "orientation": "horizontal"}, df).axes[0]
    assert ax.get_xlabel() == "sum(revenue)"
    assert ax.get_ylabel() == "region"


def test_two_or_more_series_get_a_legend(df):
    fig = jp.plot({**TYPES["bar"], "group_by": "channel"}, df)
    assert fig.legends
    assert {t.get_text() for t in fig.legends[0].get_texts()} == set(df["channel"])


def test_a_single_series_gets_no_legend(df):
    assert not jp.plot(TYPES["bar"], df).legends


def test_color_follows_the_entity(df):
    """Filtering series must not repaint the survivors."""
    full = jp.plot({**TYPES["line"], "group_by": "region"}, df)
    colors = [line.get_color() for line in full.axes[0].lines]
    filtered = jp.plot(
        {**TYPES["line"], "group_by": "region",
         "filters": [{"field": "region", "op": "in",
                      "value": list(df["region"].unique()[:2])}]}, df)
    assert [line.get_color() for line in filtered.axes[0].lines] == colors[:2]


def test_one_facet_per_category(df):
    fig = jp.plot({**TYPES["line"], "encoding":
                   {"facet": {"field": "region", "columns": 2}}}, df)
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == df["region"].nunique()


def test_both_backends_aggregate_identically(df):
    spec = {**TYPES["bar"], "group_by": "channel"}
    a = jp.build_frame(spec, df)
    b = jp.build_frame({**spec, "backend": "seaborn"}, df)
    assert a.equals(b)


def test_a_backend_that_lacks_the_type(df):
    with pytest.raises(SpecErrorGroup) as exc:
        jp.plot({"viz_type": "area", "backend": "seaborn", "x_axis": "date",
                 "y_axis": "revenue", "agg": "sum"}, df)
    assert exc.value.errors[0].code == Code.RENDERER_NOT_FOUND
    assert "matplotlib" in exc.value.errors[0].hint


def test_validate_refuses_what_the_backend_cannot_draw(df):
    """The promise is that validating is enough. `violin` on matplotlib used to
    pass validation and then raise inside plot(), which is the one failure the
    contract exists to prevent."""
    spec = {"viz_type": "violin", "x_axis": "region", "y_axis": "revenue"}
    errors = jp.validate(spec, df)
    assert [e.code for e in errors] == [Code.RENDERER_NOT_FOUND]
    assert "seaborn" in errors[0].hint

    # and without a DataFrame: which backends exist is not a question about data
    assert [e.code for e in jp.validate(spec)] == [Code.RENDERER_NOT_FOUND]


def test_what_validate_accepts_draws(df):
    spec = {"viz_type": "violin", "backend": "seaborn",
            "x_axis": "region", "y_axis": "revenue"}
    assert jp.validate(spec, df) == []
    assert isinstance(jp.plot(spec, df), Figure)


@pytest.mark.parametrize("fmt,kind", [("png", bytes), ("svg", str), ("base64", str)])
def test_output_formats(df, fmt, kind):
    out = jp.plot(TYPES["bar"], df, output=fmt)
    assert isinstance(out, kind) and len(out) > 500
    if fmt == "png":
        assert out[:4] == b"\x89PNG"
    if fmt == "svg":
        assert "<svg" in out
    if fmt == "base64":
        assert base64.b64decode(out)[:4] == b"\x89PNG"


def test_an_invalid_contract_never_reaches_matplotlib(df):
    with pytest.raises(SpecErrorGroup) as exc:
        jp.plot({"viz_type": "bar", "x_axis": "missing", "y_axis": "revenue",
                 "agg": "sum"}, df)
    assert exc.value.errors[0].code == Code.COLUMN_NOT_FOUND
    assert exc.value.to_list()[0]["path"] == "encoding.x.field"
