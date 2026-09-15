"""What the model is told before the request: one condition per prompt.

The wording of `SYSTEM`, `BARE` and `VEGA`, and the layout of both turns, are
the ones the README's numbers were measured with. Changing them changes what is
being measured, so they change with a new baseline, not in passing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from jsonplot import agent

SYSTEM = ("You produce visualization contracts. Reply with one JSON object and "
          "nothing else: no prose, no markdown fence, no explanation.")

#: What a naive integration ships — the columns, and nothing that explains the
#: vocabulary.
BARE = ("Produce a JSON chart specification for the DataFrame below. Use keys "
        "like viz_type, x_axis, y_axis, agg, title.\n\n")

#: Asks for a language the model already knows and leans on the dialect to
#: translate it. Nothing about this framework is in the prompt.
VEGA = ("Produce a Vega-Lite v5 specification for the DataFrame below. "
        "The data is already loaded, so omit the `data` block.\n\n")


@dataclass(frozen=True)
class Condition:
    name: str
    preamble: Callable[[pd.DataFrame], str]
    #: Grant one repair round: send the validator's errors back once.
    repair: bool = False


CONDITIONS: dict[str, Condition] = {
    c.name: c for c in (
        Condition("bare", lambda df: BARE + agent.columns(df)),
        Condition("vega_lite", lambda df: VEGA + agent.columns(df)),
        Condition("briefing", agent.context),
        Condition("briefing+repair", agent.context, repair=True),
    )
}


def get(name: str) -> Condition:
    try:
        return CONDITIONS[name]
    except KeyError:
        raise KeyError(f"unknown condition {name!r}; available: {list(CONDITIONS)}") from None


def first_turn(preamble: str, request: str) -> str:
    return f"{preamble}\nRequest: {request}\n"


def repair_turn(preamble: str, request: str, spec, errors: list[dict]) -> str:
    return (f"{preamble}\nRequest: {request}\n\n"
            f"Your contract:\n{json.dumps(spec)}\n\n"
            f"The validator rejected it:\n{json.dumps(errors, indent=2)}\n"
            "Send the whole corrected contract.")
