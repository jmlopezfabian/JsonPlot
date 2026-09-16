# The eval harness

Natural-language requests in, contracts scored. Two questions of every answer:
does it validate, and does it draw the chart that was asked for?

```bash
uv run evals                      # replay the recorded answers — no model needed
uv run evals run --live           # ask the model wherever the prompt has changed
uv run evals gate                 # what CI runs
```

## Why answers are recorded

Every answer a model gives is written to `runs/<run>/<model>.jsonl`, keyed by a
hash of the exact prompt that produced it, and committed. Scoring reads the
recording, so it is a pure function of the recording and the code under test.
That buys three things: a published number replays on a machine with no model
installed, a change to the validator or the dialects can be measured against
answers that were given before it, and a prompt edited without re-running the
eval is detectable — the hashes stop matching.

That last one is not hypothetical. The README claimed 12/12 for a repair round
for weeks after the briefing gained a section, because nothing re-ran.

| mode | when the model is called |
| --- | --- |
| *(default)* | never — replay only, and a missing answer is an error |
| `--live` | when the recording does not answer the current prompt |
| `--fresh` | always, recording over the old answers |

## Scoring

`valid` is `jp.validate` accepting the contract. `right` is the chart that was
asked for — decided by comparing plot frames against a gold contract, so the
flat, canonical and Vega-Lite spellings of one chart compare equal while a `sum`
where a `mean` was asked for does not. Where they disagree, the mismatch codes
(`WRONG_AGGREGATE`, `WRONG_FIELD`, `WRONG_ORDER`, …) come from comparing the
resolved contracts, so an ablation can say which mistake came back.

17 of the 144 requests cannot be drawn at all. There a rejection is the right
outcome and is scored as one.

## The gate

`uv run evals gate` fails three ways: an answer that was never recorded, a score
below the floor in `thresholds.toml`, and an answer recorded against a prompt
that has since changed. No model runs in CI.

## Ablating the briefing

```bash
uv run evals run --live --conditions minus:vega_lite minus:columns
uv run evals run --live --conditions only:types,channels,rules
```

`minus:` drops sections, `only:` keeps them, and `columns` names the DataFrame
description, which ablates like a section though it is prepended rather than
being one. Removing every section is refused: `include=()` means *every* section
to the briefing, so that condition would quietly measure the whole document.

## Tracing to Langfuse

Off unless `--trace` is passed; keys sitting in a shell are not a request to
ship a run to a server.

```bash
export LANGFUSE_HOST=http://localhost:3000
export LANGFUSE_PUBLIC_KEY=pk-lf-…
export LANGFUSE_SECRET_KEY=sk-lf-…
uv sync --package jsonplot-evals --extra tracing
uv run evals run --live --trace
```

One trace per item and condition, a generation per turn with the prompt, the
answer and the provider's token counts, and three scores: parsed, valid, right.

A local instance is `docker compose up -d` in a clone of
[langfuse/langfuse](https://github.com/langfuse/langfuse). Two things that bite:
a host already running PostgreSQL collides with the stack's own on 5432 (drop
the published port in a `docker-compose.override.yml` — the containers reach it
over the compose network), and a v4 deployment in `events_only` mode does not
serve `/api/public/traces`, so the UI is how you read them back. Keys can be
provisioned without the UI through the `LANGFUSE_INIT_*` variables, whose values
must not be quoted.
