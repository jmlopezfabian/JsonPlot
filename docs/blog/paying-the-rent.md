# What part of your tool description is paying the rent

This README used to say that a 7B model wrote 6 of 12 chart contracts correctly
from the column names alone, 11 of 12 once it was handed the generated contract,
and 12 of 12 with one repair round.

Every one of those numbers was wrong, in three different ways, and it took
building an eval harness to find out how.

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
| the columns, and "write me Vega-Lite" | 43 / 144 | 46 / 144 |
| the columns, and the generated briefing | **75 / 144** | 112 / 144 |
| …plus one repair round | 77 / 144 | 125 / 144 |

The briefing is worth having: 30 → 75 is not a subtle effect. But the column a
validator can see runs 37 points ahead of the column that matters, and the gap
is widest exactly where you would most like to trust it. The repair round buys
thirteen contracts that validate and two that are right. A loop that retries
until the validator is happy is optimizing for the validator.

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
only:columns,types,channels,rules  -58    60       2   0.0000     7642     7.6  pays rent
only:types,channels,rules          -57    59       2   0.0000     8319     6.9  pays rent
minus:columns                      -36    40       4   0.0000      677    53.2  pays rent
minus:data                         -10    14       4   0.0309      712    14.0  noise
minus:rules,example                +15    14      29   0.0315     1765    -8.5  noise
minus:types,channels               +11     7      18   0.0433     2982    -3.7  noise
minus:flat                         +10     7      17   0.0639     1680    -6.0  noise
minus:output                        -7     9       2   0.0654      191    36.6  noise
minus:channels                      +8     8      16   0.1516     1789    -4.5  noise
minus:vega_lite                     -8    19      11   0.2005     2408     3.3  noise
minus:overview                      +5     3       8   0.2266      461   -10.8  noise
minus:shape,flat                    -7    20      13   0.2962     2517     2.8  noise
minus:example                       +7    15      22   0.3240      565   -12.4  noise
minus:shape                         -6    19      13   0.3771      837     7.2  noise
minus:types                         +3     7      10   0.6291     1193    -2.5  noise
minus:rules                         +3     8      11   0.6476     1200    -2.5  noise
minus:style                         +1     8       9   1.0000      788    -1.3  noise
```

**One section survives: the column list.** 677 characters — 5% of the document —
and removing it costs 36 requests, with `COLUMN_NOT_FOUND` coming back 127 times.
Nothing else, removed on its own, moves the score by an amount this eval can
distinguish from noise. Not the chart types. Not the channel table. Not the
rules the validator enforces.

## But "each part is noise" does not mean "the parts are noise"

Read the top of that table again. Removing any single section is free. Keeping
*only* the columns, the types, the channels and the rules — cutting 7,642
characters in one go — costs **58 of 144 requests**, nearly everything the
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

## What to take from this

- **Name the things the model has to copy exactly.** Column names, enum values,
  identifiers. That is what the 677 characters are. Everything the model can
  guess it will guess; everything it must reproduce verbatim it cannot.
- **Measure the answer, not the schema.** Validation passing is not the task
  being done. Here the gap is 37 points.
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
