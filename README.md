# JsonPlot

A JSON contract and a `DataFrame` go in; a matplotlib figure comes out.

**[Documentation →](https://jmlopezfabian.github.io/JsonPlot/)** · [gallery](https://jmlopezfabian.github.io/JsonPlot/gallery/) · [the contract](https://jmlopezfabian.github.io/JsonPlot/CONTRACT/) · [agents](https://jmlopezfabian.github.io/JsonPlot/agents/)

```python
import jsonplot as jp

spec = {
    "viz_type": "bar",
    "x_axis": "region",
    "y_axis": "revenue",
    "agg": "sum",
    "title": "Revenue by region",
}
fig = jp.plot(spec, df)
```

Built for contracts written by an agent: a spec can be validated before anything
is drawn, and failures come back as structured objects (`code`, `path`, `hint`,
`did_you_mean`) rather than tracebacks.

```python
errs = jp.validate(spec, df)          # -> list[SpecError], draws nothing
ctx  = jp.contract(df)                # -> the whole contract, for the prompt
```

## Why a contract instead of generated code

Asking a model for a chart usually means asking it for matplotlib code: you
execute arbitrary code, it fails in unpredictable ways, and there is no way to
check the request before running it. A contract inverts that. The agent produces
**data**, not code — an object declaring the chart type and which column goes in
which visual channel — and the framework is the only thing that touches
matplotlib. That buys three things generated code cannot: the request can be
validated up front, it can be versioned and cached like any other data, and when
it is wrong the failure is actionable.

## Install

```bash
uv sync            # or: pip install -e ".[seaborn,streamlit]"
```

## Three dialects

The flat dialect is what an agent writes first. It normalizes into the canonical
spec, which is what the framework actually executes:

```json
{"viz_type": "bar", "x_axis": "date", "y_axis": "revenue",
 "group_by": "region", "agg": "sum"}
```

```json
{
  "viz_type": "bar",
  "encoding": {
    "x":     {"field": "date",    "type": "temporal"},
    "y":     {"field": "revenue", "type": "quantitative", "aggregate": "sum"},
    "color": {"field": "region",  "type": "nominal"}
  }
}
```

Both are accepted everywhere. When a contract carries both, the canonical block
wins.

**Vega-Lite is accepted too.** `mark`, `timeUnit`, `shape`, `row`/`column`,
`"type": "Q"`, a channel-level `sort` or `stack`, `width`/`height`, `$schema` —
all translated. What has no equivalent here (`transform`, `layer`, `params`, a
`data` source) is not silently ignored: it fails with a hint naming what to use
instead.

```json
{"mark": "bar",
 "encoding": {"x": {"field": "revenue", "type": "Q", "aggregate": "sum"},
              "y": {"field": "region", "type": "N"}}}
```

That one draws horizontal bars, because in Vega-Lite that is what swapping the
channels means.

## Chart types

`bar`, `line`, `area`, `scatter`, `hist`, `box` on matplotlib; `bar`, `line`,
`scatter`, `hist`, `box`, `violin` on seaborn. Channels: `x`, `y`, `color`,
`size`, `style`, `facet`, with aggregation, `time_unit`, `bin`, filters, sorting,
limits, stacking and horizontal orientation.

## Telling a model what to write

`jp.contract(df)` writes the contract out: the chart types and the channels each
one takes, the vocabulary, the flat dialect's keys, the rules the validator
enforces, and the columns of this DataFrame. It is **generated from the
definitions the framework executes** — the capability table, the `Literal`s in
the models, the installed renderers, the palettes — so it cannot describe a
version of the framework that no longer exists. Add a chart type and it appears
in the next prompt; a test fails if you add one and say nothing about it.

```python
jp.contract(df)                                   # markdown, for a prompt
jp.contract(format="json")                        # schema + capabilities + vocabulary
jp.tool_definition()                              # for tool-calling
agent.context(df, sections=("types", "rules"))    # when the budget is tight
```

It measurably helps. `uv run evals` puts twelve natural-language requests
through a model under four prompts and validates what comes back:

| model | bare | vega_lite | briefing | briefing+repair |
| --- | ---: | ---: | ---: | ---: |
| `qwen2.5:7b-instruct` | 6/12 | 5/12 | 11/12 | 11/12 |

`bare` is the columns plus a sentence naming the flat keys; `vega_lite` asks for
Vega-Lite and lets the dialect translate it; `briefing` is the generated
contract; `briefing+repair` sends the validator's errors back once. One request
asks for a 3D surface, which this does not draw, so there a rejection is the
right outcome — and it is the one the repair round loses: shown the errors, the
model turns the surface into a valid scatter nobody asked for.

Every answer is recorded in `evals/runs/`, so the command replays the table
above on a machine with no model installed, and a test fails when the two stop
matching. To ask the model again:

```bash
ollama serve &
uv run evals --live
```

`docs/CONTRACT.md` is a committed snapshot of the same document.

## From the terminal

```bash
jsonplot render contract.json data.csv -o chart.png
jsonplot validate contract.json data.csv   # exits 1 and writes the errors as JSON
jsonplot describe data.csv                 # the context block for the prompt
jsonplot contract data.csv                 # the whole contract, for the prompt
jsonplot contract -o docs/CONTRACT.md --check   # exits 1 if the snapshot drifted
jsonplot schema                            # the contract's JSON Schema
jsonplot types                             # available types and backends
```

## In Streamlit

```python
import jsonplot.streamlit as jps

jps.st_plot(spec, df)                 # draws it, or shows the errors
png = jps.png(spec, df)               # bytes, for st.download_button or a cache
```

`st_plot` returns `[]` when it drew and the list of `SpecError` when it could
not: in an app, a contract the data does not support should show what is wrong,
not take the page down. It also picks up the app's light or dark appearance
(unless the contract chose a theme), draws on a transparent surface so the chart
sits on the app's own background, and never leaves a figure open — otherwise
every rerun would leak one.

```python
jps.st_plot(spec, df, target=col2, theme="dark", errors="raise")
```

## Examples

```bash
uv run python examples/gallery.py out/
uv run streamlit run examples/streamlit_app.py   # a live contract editor
```

`examples/pydantic_ai_agent.ipynb` wires the whole thing to a
[Pydantic AI](https://ai.pydantic.dev) agent running on a local model: the
contract is the agent's `output_type`, `jp.validate` is an `output_validator`,
and `ModelRetry` carries the structured errors back — the repair loop with no
loop to write. It also holds a conversation (`message_history`), diffing each
follow-up against the previous contract so the drift a small model introduces is
visible rather than merely plausible. Committed with its output, so it reads
without running.

## Tests

```bash
uv run pytest
```

See `docs/architecture.md` for the pipeline, the error model, and how to add a
chart type or a backend.

## The documentation site

```bash
uv sync --group docs
uv run --group docs mkdocs serve
```

The gallery, the CLI reference and the contract page are **generated during the
build** by `scripts/gen_docs.py` — the charts are rendered, the CLI help comes
from argparse, the error codes from the `Code` class. A push to `main` deploys
it to GitHub Pages (`.github/workflows/docs.yml`), and the workflow refuses to
deploy if `docs/CONTRACT.md` no longer matches what the code generates.

## License

MIT — see [LICENSE](LICENSE). Use it, change it, ship it in something you sell;
just keep the copyright notice. It comes with no warranty.
