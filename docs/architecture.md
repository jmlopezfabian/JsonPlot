# Architecture

A JSON contract and a `DataFrame` go in; a figure comes out. In between there
are six stages, each with a defined output type.

```
JSON contract ─▶ normalize ─▶ validate ─▶ bind ─▶ transform ─▶ render ─▶ export
                              │           │
DataFrame ────────────────────┘           └─▶ list[SpecError]  (JSON, to the agent)
```

| Stage | Module | In | Out |
|---|---|---|---|
| 1 · normalize | `spec/dialects.py` | raw dict | canonical dict |
| 2 · validate | `spec/parse.py` | canonical dict | `Spec` or `list[SpecError]` |
| 3 · bind | `binding/binder.py` | `Spec` × `DataFrame` | `BoundSpec` or errors |
| 4 · transform | `transform/pipeline.py` | `BoundSpec` | plot frame |
| 5 · render | `render/driver.py` | plot frame | `Figure` |
| 6 · export | `render/driver.py` | `Figure` | figure, png, svg or base64 |

Anything that can fail because of the contract fails in stages 2 and 3, before
matplotlib is touched at all.

## The plot frame

This is the core's boundary: a `DataFrame` whose columns are named after
channels (`x`, `y`, `color`, `size`, `style`, `facet`) rather than after the
user's columns. Past that point no renderer needs to know where a value came
from, and the core's tests can assert on data instead of on pixels:

```python
jp.build_frame({"viz_type": "bar", "x_axis": "region",
                "y_axis": "revenue", "agg": "sum"}, df)
#         x          y
# 0 Central  390340.21
# 1    East  412142.26
```

## Core and backends

`spec/`, `binding/` and `transform/` never import matplotlib. A new backend
implements `Renderer.draw(ax, ctx)` and registers itself:

```python
from jsonplot.render.registry import register
from jsonplot.render.base import Renderer

@register("waterfall", "matplotlib")
class Waterfall(Renderer):
    def draw(self, ax, ctx):
        ...   # ctx.pf is the plot frame; ctx.colors, the palette already assigned
```

Titles, grid, legend and facets are the driver's job, identical for every type.
If a renderer ever needed to filter or aggregate something, the abstraction
would be in the wrong place.

## Three dialects in, one shape out

`spec/dialects.py` is the only place a contract can be spelled more than one
way. Downstream, everything sees the canonical spec.

| Dialect | What it is | Why it is accepted |
|---|---|---|
| canonical | `encoding` with a channel per role | what the framework executes |
| flat | `x_axis`, `y_axis`, `agg`, `group_by` | the first thing a model writes unprompted |
| Vega-Lite | `mark`, `timeUnit`, `shape`, `row`/`column`, a channel-level `sort` | a decade old and all over the training data; the canonical spec was shaped after it |

The Vega-Lite half is a lookup table plus four rewrites that move a value to
where it lives here: a channel's `sort` becomes `data.sort`, its `stack` becomes
`style.stacked`, `width`/`height` in pixels become `figsize` in inches, and a
bar chart with the measure on x and the category on y — Vega-Lite's way of
saying "horizontal" — swaps its channels and sets `style.orientation`. That last
one fires only when the contract declares the types or an aggregate; guessing
from the data would belong in the binder, not in a dialect.

What Vega-Lite has and this does not (`transform`, `layer`, `params`, `config`,
a `data` source) is **not** dropped. It survives into validation, and `parse.py`
turns each into a hint naming the nearest thing that does exist. The only
exception is `$schema`, which names a spec language this is not.

### What it is worth, measured

A condition in `uv run evals` asks the model for a Vega-Lite spec, says nothing
at all about this framework, and lets the dialect translate:

| Prompt | Correct outcomes |
|---|---|
| columns + a sentence naming the flat keys | 6 / 12 |
| columns + "write me Vega-Lite" | 5 / 12 |
| columns + the generated contract | 11 / 12 |
| …plus one repair round | 11 / 12 |

So the dialect does **not** replace the briefing, and it was worth measuring
rather than assuming. What is left failing in that column is not spelling — it
is `transform`, `params`, an `axis` config, and a missing `aggregate`, which are
things this framework genuinely does not do or genuinely requires. The value is
narrower than it looks from the alias table: a Vega-Lite-shaped contract stops
dying on vocabulary, and when it still dies, it says what to use instead.

## Order of operations

Fixed, and declared in `transform/pipeline.py`:

`filter → derive (time_unit, bin) → aggregate → sort → limit`

`limit` runs after `sort`, so `sort: "-y"` + `limit: 10` gives the top 10 by
value and `sort: "x"` + `limit: 10` gives the first ten in axis order. It trims
categories, never loose rows: splitting a group would produce a wrong figure.

Filters are applied in `bind` rather than in `transform`, because the duplicate,
cardinality and empty-result checks have to see the rows that will actually be
drawn.

## Errors

`SpecError(code, path, message, hint, did_you_mean)`, JSON-serializable. They
are all collected rather than raised one at a time: an agent repairs better with
the full list.

| Code | When |
|---|---|
| `INVALID_JSON` | the contract is not a JSON object |
| `UNKNOWN_FIELD` | a key that does not exist at that level |
| `MISSING_FIELD` | a required schema field is absent |
| `INVALID_VALUE` | a value outside the vocabulary |
| `UNSUPPORTED_VERSION` | a `version` this build does not speak |
| `COLUMN_NOT_FOUND` | the column is not in the DataFrame |
| `TYPE_MISMATCH` | the declared type does not fit the dtype, or the channel rejects it |
| `AGGREGATE_REQUIRED` | repeated rows on the axis with no `aggregate` |
| `AGGREGATE_INVALID` | an aggregation impossible for that dtype |
| `CARDINALITY_TOO_HIGH` | more categories than the channel can distinguish |
| `MISSING_CHANNEL` | a channel the chart type requires is absent |
| `CHANNEL_NOT_SUPPORTED` | that `viz_type` does not use that channel |
| `EMPTY_RESULT` | the filters leave zero rows |
| `RENDERER_NOT_FOUND` | that backend does not draw that type |

## Color

Categorical hues are assigned **in fixed order and never recycled**: color
follows the entity, not its rank, so filtering series does not repaint the
survivors. The default palette is validated (lightness band, chroma, separation
for color vision deficiencies, and contrast) on both a light and a dark surface,
and it has eight slots. A ninth series does not generate a new color: it is a
`CARDINALITY_TOO_HIGH` resolved by folding, filtering or faceting.

A quantitative `color` channel consumes no slots: it goes to a single-hue
continuous ramp.

## Hosts

A host is a consumer of stage 6, not a stage: the CLI (`cli.py`) and the
Streamlit adapter (`streamlit.py`) both sit past `export` and neither is
imported by the core, so `import jsonplot` does not import Streamlit and the
dependency stays optional (`pip install "jsonplot[streamlit]"`).

The Streamlit adapter exists because that host has three constraints the
library's defaults get wrong, and all three live at the boundary:

| Constraint | What the adapter does |
|---|---|
| the script reruns on every interaction | closes the figure; `png()` returns bytes, which `st.cache_data` can memoize |
| the app has its own appearance | resolves `theme="auto"` from `st.context.theme`, unless the contract set a theme |
| a traceback kills the page | catches `SpecErrorGroup` and renders the errors as an element, returning them to the caller |

It renders PNG bytes and hands them to `st.image` rather than calling
`st.pyplot`, which fixes its own dpi and deprecates savefig arguments: the
contract's `dpi` and a transparent surface would both be lost.

## The briefing

`spec/briefing.py` writes the contract out as a document for whoever has to
produce one — a model, or a person. Every fact in it is read at call time from
the definition that the framework actually executes:

| Section | Source |
|---|---|
| chart types, channels, accepted scale types | `CAPABILITIES` |
| which backends exist here | `render.registry.available()` |
| aggregations, time units, filter operators, scales, output formats | the `Literal`s in `spec/models.py`, via `get_type_hints` |
| palettes and themes | `theme.PALETTES`, `theme.THEMES` |
| the flat dialect's accepted keys and synonyms | the alias tables in `spec/dialects.py` |
| the columns of this DataFrame | `describe_dataframe` |

Nothing is transcribed. A briefing maintained by hand alongside the code is
stale by the second commit, and a stale briefing is worse than none: it tells
the model a chart type exists that no longer does, and hides the one that
replaced it.

`coverage()` is what enforces that. It returns every enumerated value the
document fails to mention, and `test_briefing.py` asserts it is empty — so
adding an aggregation without a line of prose about it fails the build.
`docs/CONTRACT.md` is a generated snapshot of the same document, checked by
`jsonplot contract -o docs/CONTRACT.md --check`.

Consumers: `jp.contract(df)`, `agent.context(df)` (the briefing plus the
columns), `tool_definition()` (the briefing, minus what the JSON Schema already
says), `jsonplot contract`, the Streamlit editor, and
`examples/pydantic_ai_agent.ipynb`, where it is an agent's `instructions` while
`Spec` is its `output_type` and `jp.validate` its `output_validator` — the
repair loop of `agent.repair`, driven by someone else's framework.

Sections can be dropped with `include=` when the prompt budget is tight; the
load-bearing ones are `types`, `channels` and `rules`.

### Whether it earns its size

`uv run evals` runs twelve natural-language requests through a model and
validates every contract that comes back. With qwen2.5:7b-instruct:

| Prompt | Correct outcomes |
|---|---|
| columns + a sentence naming the flat keys | 6 / 12 |
| columns + the briefing | 11 / 12 |
| ...plus one repair round | 11 / 12 |

One request asks for a 3D surface, which the framework does not draw; there a
rejection is the correct outcome and is scored as such. A model is not
deterministic across machines even at temperature 0, so the harness records
every answer in `evals/runs/` (one JSONL per model, each answer keyed by the
hash of the exact prompt that produced it) and scores the recording. Replaying
needs no model; `--live` asks again wherever the prompt has changed.

That recording is how a stale number got caught. The repair row read 12 / 12
when the briefing gained its Vega-Lite section, and nobody re-ran the eval.
Against the longer briefing the repair round still fixes the one invalid
contract, but it also takes the rejected 3D surface and, shown the errors,
turns it into a valid scatter: a repair loop that optimizes for passing the
validator will launder a request it should refuse.

The first run of that eval is also what produced rule 5 in its current form.
The briefing fixed every vocabulary error but left the model omitting
`aggregate` on six of twelve contracts, because the rule was stated abstractly.
Naming the failure ("the data is one row per record"), giving a default (`sum`
for amounts, `mean` for rates) and repeating it in the type table removed
almost all of them, without a repair round. The eval is how the document
gets edited; the alternative is guessing.

## The site

`mkdocs.yml` serves this `docs/` directory. Four pages are not in it because
`scripts/gen_docs.py` writes them during the build: the gallery (the contracts
in `examples/gallery.py`, actually rendered), the CLI reference (argparse's own
help), the error table (the `Code` class), and the notebook (copied from
`examples/`). `docs/CONTRACT.md` is committed rather than generated at build
time so that `--check` can fail a pull request, but it is the same document.

The rule is the one the briefing follows: if a page states a fact the code
already knows, the page is built from the code. `.github/workflows/docs.yml`
runs the tests and the snapshot check before it deploys, so the published site
cannot describe a library that failed its own tests.

## Status

Implemented: phases 01 through 07 of the plan. Outstanding: entry point
registration for external packages.
