# Driving it from an agent

This is what the library was built for. The repair loop lives *outside* the
framework — whoever orchestrates the model decides how many retries to grant —
but the framework has to make it possible: validate without rendering, and
return errors a model can act on.

## 1. Tell it what it may produce

```python
import jsonplot as jp

prompt = jp.contract(df)     # the whole contract, plus this DataFrame's columns
```

That document is **generated from the definitions the framework executes** — the
capability table, the `Literal`s in the models, the installed renderers, the
palettes, the dialect's alias tables. Not one line of it is transcribed, so it
cannot describe a version of the library that no longer exists.

Read it: [The contract](CONTRACT.md).

=== "Narrow it"

    ```python
    jp.contract(df, include=("types", "channels", "rules"))
    ```

    The load-bearing third, for when the prompt budget is tight. Sections:
    `overview`, `shape`, `types`, `channels`, `data`, `style`, `output`, `flat`,
    `rules`, `example`.

=== "Machine-readable"

    ```python
    jp.contract(format="json")
    # {"version", "schema", "capabilities", "vocabulary", "flat_dialect", "data"}
    ```

=== "Tool-calling"

    ```python
    jp.tool_definition()
    # {"name": "render_chart", "description": ..., "input_schema": ...}
    ```

    The description is the same generated document, minus what the JSON Schema
    already states — so the tool and a text prompt cannot drift apart.

=== "Just the columns"

    ```python
    from jsonplot import agent
    agent.columns(df)
    ```

    When the contract already sits in a system prompt, this is all a turn needs
    to add.

!!! info "Why this is generated and not written"

    A prompt describing the framework by hand is stale by the second commit, and
    a stale prompt is worse than none: it offers a chart type that no longer
    exists and hides the one that replaced it. `jsonplot.spec.briefing.coverage()`
    returns every value in the contract's vocabulary the document fails to
    mention, and the test suite asserts it is empty.

## 2. Validate what comes back

```python
errors = jp.validate(spec, df)     # draws nothing
```

Every error is a JSON object with `code`, `path`, `message`, `hint` and
`did_you_mean` — written to be read by a model, not by a human staring at a
traceback. See [Errors](errors.md).

## 3. Repair

The loop, if you want one already written:

```python
from jsonplot import agent

result = agent.repair(spec, df, fix=my_model_call, max_attempts=2)
result.ok        # bool
result.figure    # the figure, when ok
result.spec      # the contract that worked (or the last one tried)
result.history   # the errors from each attempt
```

`fix(spec, errors)` receives the contract that failed and the errors already
serialized to dicts, ready to drop into a prompt, and returns the next contract.

## With Pydantic AI

The contract is a pydantic model and Pydantic AI is an agent framework built on
pydantic, so there is no glue to write: `Spec` is the agent's `output_type`, the
generated contract is its `instructions`, `jp.validate` is an
`output_validator`, and `ModelRetry` runs the repair loop.

```python
chart_agent = Agent(
    model,
    deps_type=Deps,
    output_type=NativeOutput(Spec),
    instructions=jp.contract(df),
    retries=2,
)


@chart_agent.output_validator
def must_run_on_this_data(ctx: RunContext[Deps], spec: Spec) -> Spec:
    errors = jp.validate(spec, ctx.deps.df)
    if errors:
        raise ModelRetry(jp_agent.errors_as_json(errors))
    return spec
```

The full version — follow-up questions with `message_history`, a diff of what
each turn changed in the contract, and the drift a small model introduces — is a
notebook, run against a local model and committed with its output:
[A Pydantic AI agent](notebook.ipynb).

## Does it work?

`uv run evals` puts twelve natural-language requests through a model and
validates every contract that comes back. Every answer is recorded in
`evals/runs/`, so without `--live` it replays them and needs no model.

```bash
uv run evals                 # replay the recorded answers
ollama serve &
uv run evals --live          # ask qwen2.5:7b-instruct again
```

| Prompt | Correct outcomes |
| --- | --- |
| columns + a sentence naming the flat keys | 6 / 12 |
| columns + "write me Vega-Lite", translated by the dialect | 5 / 12 |
| columns + the generated contract | 11 / 12 |
| …plus one repair round | 11 / 12 |

Accepting Vega-Lite's spelling is worth doing — it costs a lookup table and it
means a contract in the dialect a model knows best does not die on vocabulary —
but the second row is the honest result: it does not replace the document. What
still fails there is semantics, not spelling.

The eval is also how the document gets edited. Its first run showed the model
omitting `aggregate` on half the contracts: the rule was in the document, stated
abstractly. Naming the failure ("the data is one row per record") and giving a
default (`sum` for amounts, `mean` for rates) fixed almost all of them without a
repair round. Guessing at prompt wording is how prompts get long; measuring is
how they get short.
