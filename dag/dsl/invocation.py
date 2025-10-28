"""Operator helper utilities for the DAG DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .program import DSLProgram


@dataclass
class OperatorInvocation:
    operator: Any
    config: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass
class RefInvocation:
    graph_name: str
    config: Mapping[str, Any]
    metadata: Mapping[str, Any]
    init_args: Mapping[str, Any] = field(default_factory=dict)


def op(
    operator: Any,
    *,
    config: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    init: Optional[Mapping[str, Any]] = None,
    call: Optional[Mapping[str, Any]] = None,
    **init_kwargs: Any,
) -> OperatorInvocation:
    config_map: Dict[str, Any] = {}
    if config:
        config_map.update(config)
    if init or init_kwargs:
        config_map.setdefault("init", {}).update(dict(init or {}))
        config_map["init"].update(init_kwargs)
    if call:
        config_map.setdefault("call", {}).update(dict(call))
    return OperatorInvocation(
        operator=operator,
        config=config_map,
        metadata=dict(metadata or {}),
    )


class OperatorsProxy:
    """Namespace providing registered operators as callables."""

    def __init__(self, registry: Mapping[str, Any]):
        self._registry = dict(registry)
        self._cache: Dict[str, Any] = {}

    def refresh(self, registry: Mapping[str, Any]) -> None:
        for name, operator in registry.items():
            if self._registry.get(name) is not operator:
                self._registry[name] = operator
                self._cache.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._registry

    def __getattr__(self, name: str):
        if name in self._cache:
            return self._cache[name]
        operator_ref = self._registry.get(name, name)

        def _call(**kwargs: Any) -> OperatorInvocation:
            return op(operator_ref, **kwargs)

        self._cache[name] = _call
        return _call


class _RefHandle:
    def __init__(self, owner: "DSLProgram", graph_name: str):
        self._owner = owner
        self._graph_name = graph_name

    def __call__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        init: Optional[Mapping[str, Any]] = None,
        **init_kwargs: Any,
    ) -> RefInvocation:
        cfg: Dict[str, Any] = {}
        if config:
            cfg.update(config)
        init_map: Dict[str, Any] = {}
        if init:
            init_map.update(init)
        if init_kwargs:
            init_map.update(init_kwargs)
        return RefInvocation(
            graph_name=self._graph_name,
            config=cfg,
            metadata=dict(metadata or {}),
            init_args=init_map,
        )


class _RefResolver:
    def __init__(self, owner: "DSLProgram"):
        self._owner = owner

    def __getattr__(self, name: str) -> _RefHandle:
        if name not in self._owner.graph_names:
            raise AttributeError(f"Unknown graph reference '{name}'")
        return _RefHandle(self._owner, name)


__all__ = [
    "OperatorInvocation",
    "OperatorsProxy",
    "RefInvocation",
    "_RefResolver",
    "op",
]
