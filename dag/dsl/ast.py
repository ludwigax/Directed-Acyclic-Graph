"""AST node definitions for the DAG DSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional


@dataclass
class BindingDecl:
    port: str
    source: Optional[str]
    default_expr: Optional[str]
    line: int


@dataclass
class OutputDecl:
    alias: str
    source: str
    line: int


@dataclass
class NodeDecl:
    name: str
    operator_expr: str
    bindings: List[BindingDecl]
    metadata: Mapping[str, Any]
    line: int


@dataclass
class ParameterDecl:
    name: str
    default_expr: Optional[str]
    line: int


@dataclass
class GraphDecl:
    name: str
    parameters: List[ParameterDecl] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[OutputDecl] = field(default_factory=list)
    nodes: List[NodeDecl] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)


__all__ = [
    "BindingDecl",
    "GraphDecl",
    "NodeDecl",
    "OutputDecl",
    "ParameterDecl",
]
