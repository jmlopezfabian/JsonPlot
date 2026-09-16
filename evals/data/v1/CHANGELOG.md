# Dataset v1

144 natural-language requests, each with the contract (or contracts) that answer
it. A version is frozen once results are published against it: fixing a gold
means `v2`, so that a number can always be traced to the exact requests it was
measured on.

## What is in it

| | |
| --- | --- |
| requests | 144 — 12 in the `readme` split, 132 in `core` |
| frames | `sales` 79, `survey` 34, `weather` 31 |
| answerable | 127; 17 are `expect: reject` |
| ambiguous | 27 carry more than one acceptable gold |
| `check`ed | 11 assert something the plot frame cannot show (a title, a bin count, a log scale, a facet grid) |
| Spanish | 12, tagged `es` |

The three frames exist to keep the eval from rewarding memorization of one
schema. `sales` is the original. `weather` has column names a model has to copy
rather than guess — `Station Name`, `Max Temp`, `Rainfall (mm)` — and real gaps
in the rainfall column, so `isnull` means something. `survey` has an id column
too unique to plot (1200 categories), 28 countries, and 6 products, which
straddles the cardinality limits: 28 categories are fine on an axis, too many
for a colour (8) or a facet grid (16).

## The reject items

17 requests have no answer this framework can draw, and a contract that fails
validation is the right outcome. Ten are chart types that do not exist here
(pie, heatmap, map, sankey, word cloud, trend line, dual axis, candlestick,
radar, gauge, 3D surface). Three are cardinality: one line per country, one
small chart per country, one bar per respondent. The rest are a mean over a
text column, and a filter that matches nothing.

They are scored inverted, so they measure whether the briefing stops a model
from inventing a chart type rather than whether it complies.

## Frames are pinned

`manifest.json` records a fingerprint of every frame. Change the data a gold was
written against and loading the dataset fails, rather than quietly scoring the
old answers against new numbers.

## History

- **v1** (2026-09-16) — first frozen version. Grew out of the twelve requests in
  `evals/local_llm.py`, which are kept verbatim as the `readme` split so the
  number published in the README stays traceable.
