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
from jsonplot.spec.briefing import SECTION_NAMES, contract

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


#: The DataFrame's columns are not one of the briefing's sections — they are
#: prepended when a frame is passed — but they are ablatable like one, and the
#: likeliest to be load-bearing. This names them.
COLUMNS = "columns"

ABLATABLE = (COLUMNS, *SECTION_NAMES)


def get(name: str) -> Condition:
    """A named condition, or an ablation of the briefing.

    `minus:vega_lite` is the briefing without that section; `minus:shape,flat`
    without either, because leave-one-out cannot see two sections that teach the
    same thing; `only:types,channels,rules` is the part `agent.context` used to
    call load-bearing, until running this scored it 57 requests below the whole
    briefing — with the column list kept, 58 — which is why the docstring now
    says something else.
    """
    if name in CONDITIONS:
        return CONDITIONS[name]
    for prefix, invert in (("minus:", True), ("only:", False)):
        if name.startswith(prefix):
            named = _named(name, name[len(prefix):])
            chosen = tuple(s for s in ABLATABLE if (s in named) is not invert)
            if not [s for s in chosen if s != COLUMNS]:
                # `include=()` means "every section" to the briefing, so an
                # ablation that removes them all would quietly measure the whole
                # document instead. The columns on their own are `bare`.
                raise KeyError(f"{name!r} leaves the briefing with no sections; "
                               f"for the columns and nothing else, use 'bare'")
            return Condition(name, _briefing(chosen))
    raise KeyError(f"unknown condition {name!r}; available: {list(CONDITIONS)}, "
                   f"or minus:/only: over {list(ABLATABLE)}")


def _named(condition: str, listed: str) -> set[str]:
    named = {s.strip() for s in listed.split(",") if s.strip()}
    unknown = named - set(ABLATABLE)
    if unknown or not named:
        raise KeyError(f"{condition!r} names {sorted(unknown) or 'no'} section(s); "
                       f"available: {list(ABLATABLE)}")
    return named


def _briefing(sections: tuple[str, ...]) -> Callable[[pd.DataFrame], str]:
    """The briefing with only these parts of it."""
    keep = tuple(s for s in sections if s != COLUMNS)

    def preamble(df: pd.DataFrame) -> str:
        if COLUMNS in sections:
            return agent.context(df, sections=keep)
        doc = contract(None, include=keep)       # the contract, with no data
        assert isinstance(doc, str)
        return doc

    return preamble


def first_turn(preamble: str, request: str) -> str:
    return f"{preamble}\nRequest: {request}\n"


def repair_turn(preamble: str, request: str, spec, errors: list[dict]) -> str:
    return (f"{preamble}\nRequest: {request}\n\n"
            f"Your contract:\n{json.dumps(spec)}\n\n"
            f"The validator rejected it:\n{json.dumps(errors, indent=2)}\n"
            "Send the whole corrected contract.")
