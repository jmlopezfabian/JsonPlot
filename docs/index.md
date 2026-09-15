---
hide:
  - navigation
---

# JsonPlot

**A JSON contract and a `DataFrame` go in; a matplotlib figure comes out.**

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

[Quickstart](quickstart.md){ .md-button .md-button--primary }
[Gallery](gallery.md){ .md-button }
[The contract](CONTRACT.md){ .md-button }

## Why a contract instead of generated code

Asking a model for a chart usually means asking it for matplotlib code: you
execute arbitrary code, it fails in unpredictable ways, and there is no way to
check the request before running it.

A contract inverts that. The model produces **data** — an object declaring the
chart type and which column goes in which visual channel — and the framework is
the only thing that touches matplotlib.

<div class="grid cards" markdown>

-   :material-shield-check:{ .lg .middle } **Checkable before it runs**

    ---

    `jp.validate(spec, df)` returns every problem as a structured object with a
    `code`, a `path`, a `hint` and a `did_you_mean`. Nothing is drawn, nothing
    is executed.

-   :material-content-save-outline:{ .lg .middle } **Storable like any data**

    ---

    A contract can be versioned, diffed, cached and put in a database next to
    the dashboard it produced. Generated code can only be run.

-   :material-account-edit-outline:{ .lg .middle } **Correctable by a person**

    ---

    It is the one part of an LLM pipeline a non-programmer can read and fix. A
    wrong column is a wrong string, not a wrong program.

-   :material-lock-outline:{ .lg .middle } **Nothing to sandbox**

    ---

    The model never emits code, so there is no `exec`, no import to police, and
    no shell to escape from.

</div>

## The contract writes itself

The document you hand a model — chart types, channels, vocabulary, the rules the
validator enforces — is **generated from the definitions the framework
executes**, never transcribed:

```python
jp.contract(df)          # markdown, for a prompt
jp.tool_definition()     # for tool-calling
```

Add a chart type and it appears in the next prompt. A test fails if you add one
and document nothing. That page is [The contract](CONTRACT.md), and it is built
fresh on every deploy of this site.

It measurably helps. `uv run evals` runs twelve natural-language requests
through a model and validates every contract that comes back:

| Prompt | Correct outcomes |
| --- | --- |
| columns + a sentence naming the keys | 6 / 12 |
| columns + "write me Vega-Lite" | 5 / 12 |
| columns + the generated contract | 11 / 12 |
| …plus one repair round | 11 / 12 |

<small>`qwen2.5:7b-instruct` via Ollama, replayed from the answers recorded in
`evals/runs/`. One request asks for a 3D surface, which this does not draw; a
rejection is the correct outcome and is scored as one. The repair round fixes
the other miss and loses that one, turning the surface into a valid scatter.</small>

## Where to go next

<div class="grid cards" markdown>

-   [**Quickstart**](quickstart.md) — install, plot, validate, in five minutes.
-   [**Gallery**](gallery.md) — every chart type with the contract that drew it.
-   [**Driving it from an agent**](agents.md) — the prompt, the repair loop, tool-calling.
-   [**A Pydantic AI agent**](notebook.ipynb) — a notebook, run against a local model.
-   [**Python API**](api.md) — every public function.
-   [**Architecture**](architecture.md) — the pipeline, and how to add a chart type.

</div>

---

MIT licensed. Copyright © 2026 Jesús Manuel López Fabián.
