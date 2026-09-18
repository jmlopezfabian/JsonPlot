"""Stages 1 and 2: normalize the contract and validate its shape.

Translates pydantic errors into `SpecError`, which is what an agent can read.
Never raises on an invalid contract: it returns the full list.
"""

from __future__ import annotations

import json
import types
import typing
from typing import Any, get_args, get_origin

from pydantic import BaseModel, ValidationError

from ..binding.errors import Code, SpecError, close_matches
from . import dialects
from .models import Spec

_CODE_BY_PYDANTIC_TYPE = {
    "extra_forbidden": Code.UNKNOWN_FIELD,
    "missing": Code.MISSING_FIELD,
    "literal_error": Code.INVALID_VALUE,
}


def parse_spec(raw: Any) -> tuple[Spec | None, list[SpecError]]:
    """Raw contract (dict or JSON string) -> validated canonical Spec.

    Returns `(spec, [])` when valid and `(None, errors)` when not.
    """
    if isinstance(raw, Spec):
        unavailable = _unavailable(raw)
        return (None, [unavailable]) if unavailable else (raw, [])
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, [SpecError(Code.INVALID_JSON, "$", f"malformed JSON: {exc.msg}",
                                    hint=f"line {exc.lineno}, column {exc.colno}")]
    if not isinstance(raw, dict):
        return None, [SpecError(Code.INVALID_JSON, "$",
                                f"the contract must be an object, not {type(raw).__name__}")]

    canonical = dialects.normalize(raw)

    version = canonical.get("version")
    if version is not None and version != "1.0":
        return None, [SpecError(
            Code.UNSUPPORTED_VERSION, "version",
            f"unsupported contract version: {version!r}",
            hint="this build speaks version '1.0'",
        )]

    try:
        spec = Spec.model_validate(canonical)
    except ValidationError as exc:
        return None, _translate(exc)

    unavailable = _unavailable(spec)
    return (None, [unavailable]) if unavailable else (spec, [])


def _unavailable(spec: Spec) -> SpecError | None:
    """Whether this installation can draw the contract at all.

    A `viz_type` the chosen backend does not implement is caught here rather
    than at render time, because the promise of the contract is that validating
    it is enough: `violin` on matplotlib used to pass `validate` and then raise
    inside `plot`, which is the one failure mode the whole design exists to
    prevent. Same error object either way, so the message does not depend on
    which door the contract came through.

    Imported late: the registry reaches back into this package for capabilities.
    """
    from ..render.registry import get_renderer

    _, error = get_renderer(spec.viz_type, spec.backend)
    return error


# --------------------------------------------------------------------------


def _translate(exc: ValidationError) -> list[SpecError]:
    out: list[SpecError] = []
    for err in exc.errors():
        loc = tuple(err["loc"])
        etype = err["type"]
        code = _CODE_BY_PYDANTIC_TYPE.get(etype, Code.INVALID_VALUE)
        path = _fmt_path(loc)

        if code is Code.UNKNOWN_FIELD:
            name = str(loc[-1]) if loc else "?"
            canonical, every = _candidates(loc[:-1])
            generic = ("fields accepted here: " + ", ".join(canonical)) if canonical else None
            out.append(SpecError(
                code, path, f"unknown field: {name!r}",
                hint=_vega_lite_hint(loc[:-1], name) or generic,
                did_you_mean=close_matches(name, every),
            ))
        elif code is Code.MISSING_FIELD:
            out.append(SpecError(code, path, "required field is missing",
                                 hint=_missing_hint(loc)))
        elif etype == "literal_error":
            expected = err.get("ctx", {}).get("expected", "")
            out.append(SpecError(
                Code.INVALID_VALUE, path,
                f"value not allowed: {err['input']!r}",
                hint=f"valid values: {expected}" if expected else None,
                did_you_mean=close_matches(str(err["input"]), _literal_options(expected)),
            ))
        else:
            out.append(SpecError(Code.INVALID_VALUE, path,
                                 err["msg"].removeprefix("Value error, ")))
    return _dedupe(out)


def _vega_lite_hint(parent: tuple, name: str) -> str | None:
    """A hint for the Vega-Lite keys this framework deliberately does not have.

    The dialect accepts Vega-Lite's spelling of what it does have, so a model
    that knows Vega-Lite gets much further — and then trips on the parts that
    have no equivalent. Saying what to use instead turns a dead end into one
    more repair round.
    """
    if not parent and name in dialects.VL_UNSUPPORTED:
        return (f"Vega-Lite's {name!r} has no equivalent here; "
                f"{dialects.VL_UNSUPPORTED[name]}")
    if parent == ("data",) and name in dialects.VL_DATA_SOURCE_KEYS:
        return ("the DataFrame is an argument to plot(), not part of the contract; "
                "'data' here means filters, sort and limit")
    return None


def _fmt_path(loc: tuple) -> str:
    parts: list[str] = []
    for p in loc:
        if isinstance(p, int):
            parts.append(f"[{p}]")
        else:
            parts.append(("." if parts else "") + str(p))
    return "".join(parts) or "$"


def _literal_options(expected: str) -> list[str]:
    # pydantic gives "'a', 'b' or 'c'"; return it as a clean list.
    return [t.strip().strip("'\"") for t in expected.replace(" or ", ", ").split(",") if t.strip()]


def _missing_hint(loc: tuple) -> str | None:
    if loc and loc[-1] == "viz_type":
        return "for example: 'bar', 'line', 'scatter', 'hist', 'box'"
    if loc and loc[-1] == "encoding":
        return "for example: {'x': {'field': 'date'}, 'y': {'field': 'revenue'}}"
    return None


def _candidates(loc: tuple) -> tuple[list[str], list[str]]:
    """Names accepted at the level `loc` points to.

    Returns (canonical, all). The canonical names go into the `hint` — those are
    the ones worth showing; the full set, aliases and flat shortcuts included, is
    used only to find the closest match.
    """
    model = _model_at(loc)
    canonical: set[str] = set()
    every: set[str] = set()
    if model is not None:
        for name, fld in model.model_fields.items():
            canonical.add(name)
            every.add(name)
            alias = fld.validation_alias
            for choice in getattr(alias, "choices", []) or []:
                if isinstance(choice, str):
                    every.add(choice)
    if not loc:
        every |= dialects.KNOWN_KEYS
    return sorted(canonical), sorted(every)


def _model_at(loc: tuple) -> type[BaseModel] | None:
    model: Any = Spec
    for part in loc:
        if isinstance(part, int):
            continue
        fld = getattr(model, "model_fields", {}).get(str(part))
        if fld is None:
            return None
        model = _unwrap(fld.annotation)
        if model is None:
            return None
    return model


def _unwrap(annotation: Any) -> type[BaseModel] | None:
    """Pull the BaseModel out of `X | None`, `list[X]`, `Annotated[X, ...]`."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin in (typing.Union, types.UnionType, list, tuple, set):
        for arg in get_args(annotation):
            found = _unwrap(arg)
            if found is not None:
                return found
    if origin is not None and hasattr(annotation, "__metadata__"):
        return _unwrap(get_args(annotation)[0])
    return None


def _dedupe(errors: list[SpecError]) -> list[SpecError]:
    seen, out = set(), []
    for e in errors:
        key = (e.code, e.path, e.message)
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out
