"""Console-friendly inspection helpers for execution plans."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from ..node import ExecutionPlan

Serializable = Dict[str, Any]


def runtime_to_dict(plan: ExecutionPlan) -> Serializable:
    """Convert an ExecutionPlan into a plain python dictionary."""
    data = plan.describe()
    data["topology"] = list(plan.topo_order)
    return data


def render_runtime_text(
    plan: ExecutionPlan,
    *,
    indent: int = 0,
    indent_step: int = 2,
) -> str:
    """Render an ExecutionPlan into a readable text block with indentation."""
    graph_dict = runtime_to_dict(plan)
    lines = _format_graph_dict(graph_dict, indent=indent, indent_step=indent_step)
    return "\n".join(lines)


def print_runtime(
    plan: ExecutionPlan,
    *,
    indent: int = 0,
    indent_step: int = 2,
) -> None:
    """Pretty-print an ExecutionPlan to stdout."""
    print(render_runtime_text(plan, indent=indent, indent_step=indent_step))


def _format_graph_dict(
    graph_dict: Serializable,
    *,
    indent: int,
    indent_step: int,
) -> List[str]:
    prefix = " " * indent
    lines: List[str] = []

    name = graph_dict.get("metadata", {}).get("name") or "<graph>"
    lines.append(f"{prefix}Graph {name}")

    metadata = graph_dict.get("metadata") or {}
    if metadata:
        lines.append(
            f"{prefix}{' ' * indent_step}metadata: {_format_mapping(metadata)}"
        )

    inputs = graph_dict.get("inputs", {})
    if inputs:
        lines.append(f"{prefix}{' ' * indent_step}Inputs:")
        for alias, endpoints in sorted(inputs.items()):
            targets = ", ".join(endpoints)
            lines.append(f"{prefix}{' ' * (2 * indent_step)}{alias} -> {targets}")

    outputs = graph_dict.get("outputs", {})
    if outputs:
        lines.append(f"{prefix}{' ' * indent_step}Outputs:")
        for alias, endpoint in sorted(outputs.items()):
            lines.append(f"{prefix}{' ' * (2 * indent_step)}{alias} <- {endpoint}")

    nodes = graph_dict.get("nodes", {})
    if nodes:
        lines.append(f"{prefix}{' ' * indent_step}Nodes:")
        for node_id, info in sorted(nodes.items()):
            lines.extend(
                _format_node(
                    node_id,
                    info,
                    indent=indent + 2 * indent_step,
                    indent_step=indent_step,
                )
            )

    topology = graph_dict.get("topology", [])
    if topology:
        lines.append(
            f"{prefix}{' ' * indent_step}topological_order: {', '.join(topology)}"
        )

    return lines


def _format_node(
    node_id: str,
    info: Mapping[str, Any],
    *,
    indent: int,
    indent_step: int,
) -> List[str]:
    prefix = " " * indent
    lines: List[str] = []

    lines.append(f"{prefix}- {node_id} :: operator:{info.get('operator')}")

    inputs = info.get("inputs", [])
    if inputs:
        lines.append(f"{prefix}{' ' * indent_step}inputs: {', '.join(inputs)}")

    outputs = info.get("outputs", [])
    if outputs:
        lines.append(f"{prefix}{' ' * indent_step}outputs: {', '.join(outputs)}")

    config = info.get("config", {})
    if config:
        lines.append(
            f"{prefix}{' ' * indent_step}config: {_format_mapping(config)}"
        )

    metadata = info.get("metadata", {})
    if metadata:
        lines.append(
            f"{prefix}{' ' * indent_step}metadata: {_format_mapping(metadata)}"
        )

    return lines


def _format_mapping(mapping: Mapping[str, Any]) -> str:
    pairs = [f"{key}={repr(value)}" for key, value in sorted(mapping.items())]
    return ", ".join(pairs) if pairs else "{}"


__all__ = [
    "runtime_to_dict",
    "render_runtime_text",
    "print_runtime",
]
