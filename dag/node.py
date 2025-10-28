"""
Public API surface for DAG core execution.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

from .core.builder import BuildError, compile_graph, materialise
from .core.nodes import GraphTemplate
from .core.registry import (
    OperatorRegistry,
    RegistrationError,
    register_class,
    register_function,
    register_graph,
    registry_default,
    returns_keys,
)
from .core.runtime.plan import ExecutionError, ExecutionPlan
from .core.specs import (
    EdgeSpec,
    GraphSpec,
    NodeSpec,
    ParameterRefValue,
    ParameterSpec,
)


SpecLike = Union[GraphSpec, GraphTemplate]


def compile_template(
    spec: SpecLike,
    *,
    registry: Optional[OperatorRegistry] = None,
) -> GraphTemplate:
    """Compile a GraphSpec into a flattened GraphTemplate."""
    if isinstance(spec, GraphTemplate):
        return spec
    reg = registry or registry_default
    return compile_graph(spec, reg)


def build_graph(
    spec: SpecLike,
    *,
    registry: Optional[OperatorRegistry] = None,
    parameters: Optional[Mapping[str, Any]] = None,
) -> ExecutionPlan:
    """Materialise a GraphSpec or GraphTemplate into an executable plan."""
    template = compile_template(spec, registry=registry)
    return materialise(template, parameters=parameters)


class GraphBuilder:
    """Backwards-compatible builder facade using the new pipeline."""

    def __init__(self, registry: Optional[OperatorRegistry] = None):
        self.registry = registry or registry_default

    def compile(self, spec: GraphSpec) -> GraphTemplate:
        return compile_template(spec, registry=self.registry)

    def materialise(
        self,
        spec: SpecLike,
        *,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> ExecutionPlan:
        template = compile_template(spec, registry=self.registry)
        return materialise(template, parameters=parameters)

    def build(
        self,
        spec: SpecLike,
        *,
        parameters: Optional[Mapping[str, Any]] = None,
    ) -> ExecutionPlan:
        return self.materialise(spec, parameters=parameters)


# ---------------------------------------------------------------------------
# Built-in operators registered on the default registry
# ---------------------------------------------------------------------------


@register_class(name="constant")
class Constant:
    """Emit a constant value supplied via config."""

    def __init__(self, value: Any):
        self.value = value

    def forward(self) -> Any:
        return self.value


@register_function(name="addition")
@returns_keys(result=float)
def addition(a: float, b: float) -> float:
    return {"result": a + b}


@register_function(name="multiplication")
@returns_keys(result=float)
def multiplication(a: float, b: float) -> float:
    return {"result": a * b}


# Backwards-compatible alias for code expecting the old name.
GraphRuntime = ExecutionPlan


__all__ = [
    "BuildError",
    "GraphBuilder",
    "GraphRuntime",
    "GraphSpec",
    "NodeSpec",
    "EdgeSpec",
    "ParameterSpec",
    "ParameterRefValue",
    "ExecutionPlan",
    "ExecutionError",
    "OperatorRegistry",
    "RegistrationError",
    "register_function",
    "register_class",
    "register_graph",
    "returns_keys",
    "registry_default",
    "build_graph",
    "compile_template",
]
