"""
Core type definitions for DAG specifications and runtime ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence


_NO_DEFAULT = object()


@dataclass(frozen=True)
class ParameterSpec:
    """Graph level parameter descriptor."""

    name: str
    default: Any = _NO_DEFAULT

    @property
    def required(self) -> bool:
        return self.default is _NO_DEFAULT


@dataclass(frozen=True)
class ParameterRefValue:
    """Placeholder referencing a graph parameter."""

    name: str
    default: Any = _NO_DEFAULT


@dataclass(frozen=True)
class PortDefinition:
    """Description of a node input/output port."""

    name: str
    type: Any = Any
    default: Any = _NO_DEFAULT
    description: str | None = None

    @property
    def required(self) -> bool:
        return self.default is _NO_DEFAULT


_SER_PARAM_KEY = "__dag_param__"
_SER_TYPE_KEY = "__dag_type__"
_SER_ITEMS_KEY = "items"
_SER_TYPE_TUPLE = "tuple"
_SER_TYPE_SET = "set"


def encode_config_value(value: Any) -> Any:
    if isinstance(value, ParameterRefValue):
        payload: Dict[str, Any] = {_SER_PARAM_KEY: value.name}
        if value.default is not _NO_DEFAULT:
            payload["default"] = value.default
        return payload

    if isinstance(value, Mapping):
        return {key: encode_config_value(val) for key, val in value.items()}

    if isinstance(value, tuple):
        return {
            _SER_TYPE_KEY: _SER_TYPE_TUPLE,
            _SER_ITEMS_KEY: [encode_config_value(item) for item in value],
        }

    if isinstance(value, set):
        return {
            _SER_TYPE_KEY: _SER_TYPE_SET,
            _SER_ITEMS_KEY: [encode_config_value(item) for item in value],
        }

    if isinstance(value, list):
        return [encode_config_value(item) for item in value]

    return value


def decode_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value.keys()) <= {_SER_PARAM_KEY, "default"} and _SER_PARAM_KEY in value:
            default = value.get("default", _NO_DEFAULT)
            return ParameterRefValue(name=value[_SER_PARAM_KEY], default=default)

        type_tag = value.get(_SER_TYPE_KEY)
        if type_tag == _SER_TYPE_TUPLE:
            items = value.get(_SER_ITEMS_KEY, [])
            return tuple(decode_config_value(item) for item in items)
        if type_tag == _SER_TYPE_SET:
            items = value.get(_SER_ITEMS_KEY, [])
            return {decode_config_value(item) for item in items}

        return {key: decode_config_value(val) for key, val in value.items()}

    if isinstance(value, list):
        return [decode_config_value(item) for item in value]

    return value


def ensure_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"Expected mapping for '{label}' configuration")


def normalise_output_definitions(
    value: Mapping[str, Any] | Sequence[str],
) -> Dict[str, PortDefinition]:
    if isinstance(value, Mapping):
        return {
            name: PortDefinition(
                name=name,
                type=type_hint if type_hint is not None else Any,
            )
            for name, type_hint in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {name: PortDefinition(name=name) for name in value}
    raise TypeError(
        "Output definition should be a mapping (name->type) or sequence of names"
    )


__all__ = [
    "_NO_DEFAULT",
    "ParameterSpec",
    "ParameterRefValue",
    "PortDefinition",
    "encode_config_value",
    "decode_config_value",
    "ensure_mapping",
    "normalise_output_definitions",
]
