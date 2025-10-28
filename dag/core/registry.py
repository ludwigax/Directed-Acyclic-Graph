"""
Operator registry built on top of node templates.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

from .nodes import NodeTemplate
from .specs import GraphSpec


class RegistrationError(RuntimeError):
    """Raised when operator registration fails."""


class OperatorRegistry:
    """Registry mapping operator names to templates or graph specs."""

    def __init__(self) -> None:
        self._entries: Dict[str, Any] = {}

    def register(self, name: str, entry: Any) -> Any:
        if name in self._entries:
            raise RegistrationError(f"Operator '{name}' already registered")
        self._entries[name] = entry
        return entry

    def register_template(
        self,
        template: NodeTemplate,
    ) -> NodeTemplate:
        return self.register(template.name, template)

    def register_function(
        self,
        func: Optional[Callable[..., Any]] = None,
        *,
        name: Optional[str] = None,
        outputs: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        if func is None:
            def wrapper(f: Callable[..., Any]) -> Callable[..., Any]:
                self.register_function(
                    f, name=name, outputs=outputs, metadata=metadata
                )
                return f

            return wrapper

        template = NodeTemplate.from_function(
            func, name=name, outputs=outputs, metadata=metadata
        )
        self.register_template(template)
        return func

    def register_class(
        self,
        cls: Optional[type] = None,
        *,
        name: Optional[str] = None,
        forward: str = "forward",
        outputs: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        if cls is None:
            def wrapper(klass: type) -> type:
                self.register_class(
                    klass,
                    name=name,
                    forward=forward,
                    outputs=outputs,
                    metadata=metadata,
                )
                return klass

            return wrapper

        template = NodeTemplate.from_class(
            cls,
            name=name,
            forward=forward,
            outputs=outputs,
            metadata=metadata,
        )
        self.register_template(template)
        return cls

    def register_graph(
        self,
        name: str,
        graph_spec: GraphSpec,
    ) -> GraphSpec:
        return self.register(name, graph_spec)

    def get(self, name: str) -> Any:
        try:
            return self._entries[name]
        except KeyError as exc:
            raise RegistrationError(f"Unknown operator '{name}'") from exc

    def items(self):
        return self._entries.items()


registry_default = OperatorRegistry()


def register_function(
    func: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    outputs: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
):
    return registry_default.register_function(
        func, name=name, outputs=outputs, metadata=metadata
    )


def register_class(
    cls: Optional[type] = None,
    *,
    name: Optional[str] = None,
    forward: str = "forward",
    outputs: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
):
    return registry_default.register_class(
        cls,
        name=name,
        forward=forward,
        outputs=outputs,
        metadata=metadata,
    )


def register_graph(
    name: str,
    graph_spec: GraphSpec,
):
    return registry_default.register_graph(name, graph_spec)


def returns_keys(**outputs: Any):
    """
    Decorator recording explicit output names (and optional type hints).

    Example::

        @returns_keys(result=int, remainder=int)
        def divide(a: int, b: int):
            return {"result": a // b, "remainder": a % b}
    """

    def decorator(obj: Callable[..., Any]) -> Callable[..., Any]:
        obj.__dag_returns__ = dict(outputs)
        return obj

    return decorator


__all__ = [
    "RegistrationError",
    "OperatorRegistry",
    "registry_default",
    "register_function",
    "register_class",
    "register_graph",
    "returns_keys",
]
