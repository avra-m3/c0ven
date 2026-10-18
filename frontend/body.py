"""Reading a request's JSON body: the checks both APIs share. A wrong body is a BadRequest (400)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import protocol
from frontend.generate import BadRequest

type JSONObject = dict[str, Any]


def obj(value: object, what: str) -> JSONObject:
    if not isinstance(value, dict):
        raise BadRequest(f"{what} must be an object")
    return cast(JSONObject, value)


def array(value: object, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise BadRequest(f"{what} must be an array")
    return cast(list[Any], value)


def boolean(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def number(body: JSONObject, key: str, kind: type[int] | type[float]) -> int | float | None:
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BadRequest(f"{key} must be a number")
    return kind(value)


def settings(defaults: protocol.Settings, values: dict[str, int | float]) -> protocol.Settings:
    """The model's defaults with the request's values over them."""
    try:
        return replace(defaults, **values)
    except protocol.ProtocolError as exc:
        raise BadRequest(str(exc)) from None
