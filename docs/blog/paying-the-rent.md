# What part of your tool description is paying the rent

This README used to say that a 7B model wrote 6 of 12 chart contracts correctly
from the column names alone, 11 of 12 once it was handed the generated contract,
and 12 of 12 with one repair round.

Every one of those numbers was wrong, in four different ways, and it took
building an eval harness to find out how. The fourth way was a bug in this
library, which the harness found by asking a question the test suite never had.

## The first thing wrong: the number was stale

The 12/12 was measured on 3 September. On 4 September the briefing gained a
section about Vega-Lite spellings, which made it 2.4k characters longer. Nobody
re-ran the eval, because re-running it meant having Ollama up and waiting four
minutes, and the test suite was green either way.

Replaying the same twelve requests against the current briefing gives 11/12 for
the repair round, not 12. The repair round now takes the one request that cannot
be drawn — *"a 3D surface of revenue against price and units"* — and, shown the
validator's complaint, turns it into a perfectly valid scatter plot of price
against units, coloured by revenue. Nobody asked for that chart. It validates.

So the harness records every answer a model gives, keyed by a hash of the exact
prompt that produced it, and commits the recording. Scoring reads the recording.
That means a published number replays on a machine with no model installed, and
a prompt edited without re-running the eval is detectable: the hashes stop
matching. CI fails on it.

## The second thing wrong: it was measuring the wrong thing

`11/12` meant eleven contracts passed validation. Validation checks that the
columns exist, the types fit the chart, the aggregate is legal. It cannot check
that the chart answers the question.

So the dataset grew to 144 requests over three DataFrames, each with the
contract — or contracts, where the request is genuinely ambiguous — that answers
it. A contract is *right* when the data it would plot matches the gold: same
chart type, same columns in the same visual roles, same aggregation, same rows.
Dialects normalize, so the flat, canonical and Vega-Lite spellings of one chart
compare equal; a `sum` where a `mean` was asked for does not.

| prompt | right chart | valid contract |
| --- | ---: | ---: |
| the columns, and a sentence naming the keys | 30 / 144 | 51 / 144 |
| the columns, and "write me Vega-Lite" | 41 / 144 | 44 / 144 |
| the columns, and the generated briefing | **68 / 144** | 103 / 144 |
| …plus one repair round | 77 / 144 | 125 / 144 |

The briefing is worth having: 30 → 68 is not a subtle effect. But the column a
validator can see runs 35 points ahead of the column that matters, and the gap
is widest exactly where you would most like to trust it. The repair round buys
twenty-two contracts that validate and nine that are right: most of what it
fixes still draws the wrong chart, because a loop that retries until the
validator is happy is optimizing for the validator.

## The third thing wrong: nobody knew which part was working

The briefing is 12,530 characters of generated documentation: an overview, the
shape of a spec, the chart types, the channels, filtering and sorting, styling,
output, the flat dialect, the Vega-Lite spellings, the rules, a worked example —
and, prepended, a description of the actual DataFrame.

Which of those is doing the work? The honest way to find out is to remove one at
a time and re-measure. 144 requests × 17 conditions, with the answers recorded
so the scoring is reproducible.

The catch is that a difference of five requests out of 144 looks like a finding
and is usually a coin toss. The two conditions answer the same requests with the
same model at the same temperature, so the requests are paired, and the only
evidence is the ones that flipped. McNemar's exact test asks how often a split
that lopsided happens by chance. And comparing seventeen sections against one
baseline means roughly one of them looks load-bearing by accident, so the
significance level is split across the comparisons.

```
condition                   Δ right  lost  gained        p  chars cut  per 1k  verdict
--------------------------------------------------------------------------------------
only:columns,types,channels,rules  -51    53       2   0.0000     7642     6.7  pays rent
only:types,channels,rules          -50    52       2   0.0000     8319     6.0  pays rent
minus:columns                      -34    38       4   0.0000      677    50.2  pays rent
minus:types,channels               +13     5      18   0.0106     2982    -4.4  noise
minus:flat                         +14     7      21   0.0125     1680    -8.3  noise
minus:rules,example                +16    13      29   0.0195     1765    -9.1  noise
minus:channels                     +11     6      17   0.0347     1789    -6.1  noise
minus:data                          -8    12       4   0.0768      712    11.2  noise
minus:output                        -6     8       2   0.1094      191    31.4  noise
minus:example                      +10    12      22   0.1214      565   -17.7  noise
minus:vega_lite                     -8    19      11   0.2005     2408     3.3  noise
minus:shape,flat                    -8    21      13   0.2295     2517     3.2  noise
minus:shape                         -6    19      13   0.3771      837     7.2  noise
minus:overview                      +4     4       8   0.3877      461    -8.7  noise
minus:rules                         +3     9      12   0.6636     1200    -2.5  noise
minus:style                         +2     7       9   0.8036      788    -2.5  noise
minus:types                         +2     8      10   0.8145     1193    -1.7  noise
```

**One section survives: the column list.** 677 characters — 5% of the document —
and removing it costs 34 requests, with `COLUMN_NOT_FOUND` coming back 121 times.
Nothing else, removed on its own, moves the score by an amount this eval can
distinguish from noise. Not the chart types. Not the channel table. Not the
rules the validator enforces.

## But "each part is noise" does not mean "the parts are noise"

Read the top of that table again. Removing any single section is free. Keeping
*only* the columns, the types, the channels and the rules — cutting 7,642
characters in one go — costs **51 of 144 requests**, nearly everything the
briefing was buying.

Both are true, and the combination is the actual finding: the document is
**redundant, not padded**. Rule 5 is stated in the rules section, repeated in the
chart-type table, and demonstrated in the worked example. Drop any one of the
three and the other two carry it. Drop all three and the model stops aggregating.
Leave-one-out cannot see redundancy — that is what it is blind to, by
construction — which is why the pairs and the `only:` conditions are in the table.

This matters for the advice people actually follow. The docstring of this
library's own `agent.context` said, for months, that `("types", "channels",
"rules")` was the load-bearing part, to keep when the prompt budget is tight.
Measured, that is the worst advice in the document. It has been corrected.

## The number that was plausible and completely false

Running a second model produced this:

| | right | valid |
| --- | ---: | ---: |
| `gemma4:e4b`, the briefing | 17 / 144 | **0 / 144** |

Zero valid contracts, and yet 17 right — because the dataset has exactly 17
requests that cannot be drawn, where failing *is* the right answer. A model that
produces nothing at all scores 17.

The cause was not the model. Ollama's context window bounds the prompt and the
generated answer *together*, and the default is 4,096 tokens. The briefing
tokenizes to about 3,850 for one model and 4,071 for the other — the same
document, a different tokenizer, six percent apart. The first left roughly 250
tokens to answer in, which is enough for a chart contract. The second left
twenty-five, and 110 of its 144 answers came back empty. Nothing was reported:
the generation simply stopped at the ceiling.

The median answer length tells the whole story — 25 tokens at a window of 4,096,
112 tokens at 8,192, from identical prompts. The harness was measuring its own
ceiling and reporting it as a property of the model.

If that number had been published, it would have looked entirely reasonable.
Small model, long document, poor results — of course. It would have been a lie,
and nothing in the test suite would have caught it. The context window is now
part of the recorded conditions, so a run with a different window is a different
number rather than the same one.

## Three models, measured properly

With the window set wide enough to answer in, the same 144 requests across three
local models:

| model | columns only | Vega-Lite | briefing | +repair |
| --- | ---: | ---: | ---: | ---: |
| `qwen2.5:7b-instruct` | 30 | 41 | **68** | 77 |
| `gemma4:e4b` | 26 | 19 | **88** | 90 |
| `llama3.2:3b` | 17 | 17 | **33** | 32 |

<small>Right charts out of 144. Validity, in the same order: 51/44/103/125,
50/3/113/130, 0/0/104/109.</small>

Three things in that table are worth more than the ranking.

**The document changes the ranking.** Without it, qwen and gemma4 are level (30
and 26) and you would call them equivalent. With it, gemma4 is twenty requests
ahead. Benchmarking models on a bare prompt would have told you the wrong thing
about which one to ship.

**17 is not a score.** `llama3.2:3b` scores exactly 17 from the columns alone,
and the dataset has exactly 17 requests that cannot be drawn. It never once drew
the right chart; it scored only where failing was correct. The answers are not
garbage, which is the interesting part — they are clean JSON in a schema it
invented, `"viz_type": "bar_chart"` with `x_axis` as an object instead of a
column name. Plausible, well-formed, and unusable in every single case. A schema
in the prompt takes it from 0 valid contracts to 104.

**Validity flatters the weak model most.** `llama3.2:3b` reaches 104 valid
contracts and 33 right ones: more than two thirds of what passes the validator
draws the wrong chart. The gap between the two columns is not a constant — it widens
as the model gets weaker, which is exactly when you are most likely to be
relying on the validator to tell you things are fine.

## The fourth thing wrong: the library

Every number above moved on the last day of this work, because the eval found a
bug that four hundred tests had not.

`jp.validate` accepted `violin` on the matplotlib backend. Matplotlib has no
violin renderer here — seaborn does — so the contract validated cleanly and then
raised `RENDERER_NOT_FOUND` inside `plot()`. That is the one failure the whole
design exists to rule out: the pitch for a contract over generated code is that
you can check it *before* anything is drawn, and for that chart type you could
not. An agent doing exactly what the documentation says would have been told its
contract was fine and then handed an exception.

Across the three models, 32 recorded contracts were in that state: valid,
undrawable, and several of them scored as *right*, because the plot frame they
described matched the gold. The architecture page had been claiming for months
that "anything that can fail because of the contract fails in stages 2 and 3,
before matplotlib is touched at all". It was a good sentence. It was not true.

Validation now asks the registry whether the installation can draw the pair, so
`violin` without seaborn is an error from `validate` rather than an exception
from `plot`. The numbers in this post are the ones after that fix — the briefing
dropped from 75 right to 68, because seven of those contracts had been counted
as correct answers to requests they would have crashed on.

And the repair round changed meaning entirely. It used to look like a loop that
bought validity and nothing else. Now those contracts come back as errors the
model can read, and it fixes most of them: 68 → 77, where before the fix it was
75 → 77. Telling the model the truth earlier is worth more than another retry.

## What to take from this

- **Name the things the model has to copy exactly.** Column names, enum values,
  identifiers. That is what the 677 characters are. Everything the model can
  guess it will guess; everything it must reproduce verbatim it cannot.
- **Measure the answer, not the schema.** Validation passing is not the task
  being done. Here the gap is 35 points.
- **Be careful what your repair loop optimizes for.** Ours bought validity and
  laundered a request that should have been refused.
- **Redundancy in a prompt is doing work you cannot see by ablating one thing at
  a time.** If you trim, trim gently, and re-measure.
- **A number you cannot trace back to the exact prompt that produced it is not a
  measurement.** It is a claim with a citation to nothing.

## Reproducing this

```bash
uv run evals                  # replay the recorded answers — no model needed
uv run evals ablation         # the table above
uv run evals gate             # what CI fails on
```

Every answer is committed under `evals/runs/`. The dataset, the gold contracts
and the scoring are in `evals/`.

<small>qwen2.5:7b-instruct via Ollama, temperature 0, 144 requests over three
DataFrames, single run per condition. One model and one domain: a section that
this eval cannot distinguish from noise may well matter for a different model,
or for requests this dataset under-samples. "Noise" here means "not shown to
help", which is not the same as "shown not to help".</small>
