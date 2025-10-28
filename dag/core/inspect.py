"""
Callable inspection utilities shared across core modules.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Mapping, MutableMapping, Sequence, Tuple, Union

from .ports import (
    PortDefinition,
    normalise_output_definitions,
    _NO_DEFAULT,
)


class InspectionError(TypeError):
    """Raised when callable inspection fails."""


def infer_input_ports(
    signature: inspect.Signature,
    *,
    skip_bound_argument: bool = False,
) -> Dict[str, PortDefinition]:
    parameters = list(signature.parameters.values())
    if skip_bound_argument and parameters:
        parameters = parameters[1:]

    ports: Dict[str, PortDefinition] = {}
    for param in parameters:
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise InspectionError(
                "Variable positional or keyword arguments are not supported "
                "in operator signatures"
            )
        annotation = param.annotation if param.annotation is not inspect._empty else Any
        default = param.default if param.default is not inspect._empty else _NO_DEFAULT
        ports[param.name] = PortDefinition(
            name=param.name,
            type=annotation,
            default=default,
        )
    return ports


def infer_output_ports(
    obj: Callable[..., Any],
    *,
    explicit: Union[Sequence[str], Mapping[str, Any], None] = None,
) -> Tuple[Dict[str, PortDefinition], Sequence[str]]:
    if explicit is not None:
        ports = normalise_output_definitions(explicit)
        return ports, list(ports.keys())

    annotated = getattr(obj, "__dag_returns__", None)
    if annotated is not None:
        ports = normalise_output_definitions(annotated)
        return ports, list(ports.keys())

    return {
        "_return": PortDefinition(name="_return"),
    }, ["_return"]


def infer_default_kwargs(
    signature: inspect.Signature,
    *,
    skip_bound_argument: bool = False,
) -> Dict[str, Any]:
    parameters = list(signature.parameters.values())
    if skip_bound_argument and parameters:
        parameters = parameters[1:]

    defaults: Dict[str, Any] = {}
    for param in parameters:
        if param.default is not inspect._empty:
            defaults[param.name] = param.default
    return defaults


def gather_callable_metadata(
    func: Callable[..., Any],
    *,
    explicit_outputs: Union[Sequence[str], Mapping[str, Any], None] = None,
) -> Tuple[
    Dict[str, PortDefinition],
    Sequence[str],
    Dict[str, PortDefinition],
]:
    """Return (input ports, output keys, output ports)."""
    signature = inspect.signature(func)
    input_ports = infer_input_ports(signature)
    output_ports, output_keys = infer_output_ports(func, explicit=explicit_outputs)
    return input_ports, output_keys, output_ports


def gather_class_metadata(
    operator_cls: type,
    *,
    forward: str = "forward",
    explicit_outputs: Union[Sequence[str], Mapping[str, Any], None] = None,
) -> Tuple[
    Dict[str, PortDefinition],
    Sequence[str],
    Dict[str, PortDefinition],
    Dict[str, Any],
]:
    if not hasattr(operator_cls, forward):
        raise InspectionError(
            f"Operator class '{operator_cls.__name__}' missing '{forward}' method"
        )

    forward_fn = getattr(operator_cls, forward)
    forward_sig = inspect.signature(forward_fn)
    init_sig = inspect.signature(operator_cls.__init__)

    input_ports = infer_input_ports(
        forward_sig,
        skip_bound_argument=True,
    )
    output_ports, output_keys = infer_output_ports(
        forward_fn, explicit=explicit_outputs
    )
    init_defaults = infer_default_kwargs(
        init_sig,
        skip_bound_argument=True,
    )
    return input_ports, output_keys, output_ports, init_defaults


__all__ = [
    "InspectionError",
    "infer_input_ports",
    "infer_output_ports",
    "infer_default_kwargs",
    "gather_callable_metadata",
    "gather_class_metadata",
]
