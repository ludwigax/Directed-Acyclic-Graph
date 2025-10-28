"""
Declarative specification layer for DAG graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, List

from .ports import (
    ParameterSpec,
    ParameterRefValue,
    encode_config_value,
    decode_config_value,
    _NO_DEFAULT,
)


class SpecError(ValueError):
    """Raised when graph specifications are malformed."""


@dataclass(frozen=True)
class NodeSpec:
    """Declarative node descriptor."""

    id: str
    operator: Any
    config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeSpec:
    """Declarative edge descriptor (src and dst use '<node>.<port>' syntax)."""

    src: str
    dst: str

    def unpack(self) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        return parse_endpoint(self.src), parse_endpoint(self.dst)


@dataclass(frozen=True)
class GraphSpec:
    """Top-level graph declaration."""

    nodes: Mapping[str, NodeSpec]
    edges: Sequence[EdgeSpec]
    parameters: Mapping[str, ParameterSpec] = field(default_factory=dict)
    inputs: Mapping[str, Union[str, Sequence[str]]] = field(default_factory=dict)
    outputs: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraphSpec":
        nodes_raw = data.get("nodes", {})
        edges_raw = data.get("edges", [])
        parameters_raw = data.get("parameters", {})
        inputs = dict(data.get("inputs", {}))
        outputs = dict(data.get("outputs", {}))
        metadata = dict(data.get("metadata", {}))

        parameters: Dict[str, ParameterSpec] = {}
        for name, spec in parameters_raw.items():
            if isinstance(spec, ParameterSpec):
                parameters[name] = spec
            elif isinstance(spec, Mapping):
                default = spec.get("default", _NO_DEFAULT)
                parameters[name] = ParameterSpec(name=name, default=default)
            else:
                parameters[name] = ParameterSpec(name=name, default=spec)

        nodes: Dict[str, NodeSpec] = {}
        for node_id, spec in nodes_raw.items():
            if isinstance(spec, NodeSpec):
                nodes[node_id] = spec
            else:
                operator_spec = spec["operator"]
                if isinstance(operator_spec, GraphSpec):
                    operator_value = operator_spec
                elif isinstance(operator_spec, Mapping):
                    operator_value = GraphSpec.from_dict(operator_spec)
                else:
                    operator_value = operator_spec
                nodes[node_id] = NodeSpec(
                    id=node_id,
                    operator=operator_value,
                    config=decode_config_value(spec.get("config", {})),
                    metadata=dict(spec.get("metadata", {})),
                )

        edges: List[EdgeSpec] = []
        for spec in edges_raw:
            if isinstance(spec, EdgeSpec):
                edges.append(spec)
            else:
                edges.append(EdgeSpec(src=spec["src"], dst=spec["dst"]))

        if not outputs:
            raise SpecError("Graph specification requires at least one output")

        return cls(
            nodes=nodes,
            edges=edges,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        nodes: Dict[str, Any] = {}
        for node_id, node_spec in self.nodes.items():
            operator_ref = node_spec.operator
            if isinstance(operator_ref, GraphSpec):
                operator_value: Any = operator_ref.to_dict()
            else:
                operator_value = operator_ref
            nodes[node_id] = {
                "operator": operator_value,
                "config": encode_config_value(dict(node_spec.config)),
                "metadata": dict(node_spec.metadata),
            }

        edges = [{"src": edge.src, "dst": edge.dst} for edge in self.edges]

        parameters: Dict[str, Any] = {}
        for name, spec in self.parameters.items():
            if spec.default is _NO_DEFAULT:
                parameters[name] = {}
            else:
                parameters[name] = {"default": spec.default}

        inputs: Dict[str, Any] = {}
        for alias, endpoint_spec in self.inputs.items():
            if isinstance(endpoint_spec, (list, tuple, set)):
                inputs[alias] = list(endpoint_spec)
            else:
                inputs[alias] = endpoint_spec

        return {
            "nodes": nodes,
            "edges": edges,
            "parameters": parameters,
            "inputs": inputs,
            "outputs": dict(self.outputs),
            "metadata": dict(self.metadata),
        }


def parse_endpoint(value: str) -> Tuple[str, str]:
    if "." not in value:
        raise SpecError(f"Endpoint '{value}' must use '<node>.<port>' notation")
    node_id, port = value.split(".", 1)
    if not node_id or not port:
        raise SpecError(f"Endpoint '{value}' must include both node id and port")
    return node_id, port


__all__ = [
    "SpecError",
    "GraphSpec",
    "NodeSpec",
    "EdgeSpec",
    "ParameterRefValue",
    "ParameterSpec",
    "parse_endpoint",
]
