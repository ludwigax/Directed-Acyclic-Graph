"""Console-friendly inspection helpers for graph runtimes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from ..node import (
    GraphInputRef,
    GraphRuntime,
    GraphSpec,
    NodeOutputRef,
    OperatorTemplate,
    PortDefinition,
)


Serializable = Dict[str, Any]


def runtime_to_dict(runtime: GraphRuntime) -> Serializable:
    """Convert a GraphRuntime into a plain python dictionary."""
    visited: Dict[int, Serializable] = {}
    return _serialise_graph(runtime, visited)


def render_runtime_text(
    runtime: GraphRuntime,
    *,
    indent: int = 0,
    indent_step: int = 2,
) -> str:
    """Render a GraphRuntime into a readable text block with indentation."""
    graph_dict = runtime_to_dict(runtime)
    lines = _format_graph_dict(graph_dict, indent=indent, indent_step=indent_step)
    return "\n".join(lines)


def print_runtime(
    runtime: GraphRuntime,
    *,
    indent: int = 0,
    indent_step: int = 2,
) -> None:
    """Pretty-print a GraphRuntime to stdout."""
    print(render_runtime_text(runtime, indent=indent, indent_step=indent_step))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _serialise_graph(
    runtime: GraphRuntime,
    visited: Dict[int, Serializable],
) -> Serializable:
    runtime_id = id(runtime)
    if runtime_id in visited:
        return {"type": "graph_ref", "id": runtime_id}

    spec = runtime.spec
    graph_dict: Serializable = {
        "type": "graph",
        "id": runtime_id,
        "name": spec.metadata.get("name"),
        "metadata": dict(spec.metadata),
        "inputs": _serialise_graph_inputs(runtime),
        "outputs": _serialise_graph_outputs(runtime),
        "nodes": [],
        "edges": _serialise_edges(runtime),
        "topology": list(runtime.topological_order),
    }
    visited[runtime_id] = graph_dict

    for node_id in runtime.topological_order:
        node_runtime = runtime.node_runtimes[node_id]
        node_spec = spec.nodes[node_id]
        node_dict: Serializable = {
            "id": node_id,
            "operator": _serialise_operator(node_spec, node_runtime, visited),
            "config": dict(node_spec.config),
            "metadata": dict(node_spec.metadata),
            "inputs": _serialise_node_inputs(runtime, node_id, node_runtime.input_ports),
            "outputs": sorted(node_runtime.output_ports.keys()),
        }
        graph_dict["nodes"].append(node_dict)

    return graph_dict


def _serialise_graph_inputs(runtime: GraphRuntime) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for alias, endpoints in runtime.graph_inputs.items():
        result[alias] = [f"{node}.{port}" for node, port in endpoints]
    return result


def _serialise_graph_outputs(runtime: GraphRuntime) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for alias, ref in runtime.graph_outputs.items():
        result[alias] = f"{ref.node_id}.{ref.port}"
    return result


def _serialise_edges(runtime: GraphRuntime) -> List[Serializable]:
    edges: List[Serializable] = []
    for dst_node, port_map in runtime._inbound.items():  # pylint: disable=protected-access
        for dst_port, ref in port_map.items():
            if isinstance(ref, NodeOutputRef):
                edges.append(
                    {
                        "src": {"node": ref.node_id, "port": ref.port},
                        "dst": {"node": dst_node, "port": dst_port},
                    }
                )
    return edges


def _serialise_operator(
    node_spec,
    node_runtime,
    visited: Dict[int, Serializable],
) -> Serializable:
    operator_ref = node_spec.operator

    if isinstance(operator_ref, GraphSpec):
        nested_runtime = _extract_nested_runtime(node_runtime)
        nested_graph = (
            _serialise_graph(nested_runtime, visited) if nested_runtime else None
        )
        return {
            "type": "inline_graph",
            "name": node_runtime.template.name,
            "graph": nested_graph,
        }

    if isinstance(operator_ref, OperatorTemplate):
        return {"type": "template", "name": operator_ref.name}

    if isinstance(operator_ref, str):
        return {"type": "registered", "name": operator_ref}

    return {"type": "object", "repr": repr(operator_ref)}


def _extract_nested_runtime(node_runtime) -> Optional[GraphRuntime]:
    runner = node_runtime.runner
    return getattr(runner, "_runtime", None)


def _serialise_node_inputs(
    runtime: GraphRuntime,
    node_id: str,
    port_defs: Mapping[str, PortDefinition],
) -> Dict[str, Serializable]:
    inputs: Dict[str, Serializable] = {}
    inbound = runtime._inbound.get(node_id, {})  # pylint: disable=protected-access

    for port_name in sorted(port_defs.keys()):
        ref = inbound.get(port_name)
        port_def = port_defs[port_name]
        inputs[port_name] = _serialise_input_ref(ref, port_def)

    return inputs


def _serialise_input_ref(
    ref: Optional[object],
    port_def: PortDefinition,
) -> Serializable:
    if isinstance(ref, GraphInputRef):
        return {"type": "graph_input", "name": ref.name}

    if isinstance(ref, NodeOutputRef):
        return {"type": "node_output", "node": ref.node_id, "port": ref.port}

    if not port_def.required:
        return {"type": "default", "value": port_def.default}

    return {"type": "unbound"}


def _format_graph_dict(
    graph_dict: Serializable,
    *,
    indent: int,
    indent_step: int,
) -> List[str]:
    prefix = " " * indent
    lines: List[str] = []

    name = graph_dict.get("name") or "<graph>"
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

    nodes = graph_dict.get("nodes", [])
    if nodes:
        lines.append(f"{prefix}{' ' * indent_step}Nodes:")
        for node in nodes:
            lines.extend(
                _format_node(
                    node,
                    indent=indent + 2 * indent_step,
                    indent_step=indent_step,
                )
            )

    return lines


def _format_node(
    node: Serializable,
    *,
    indent: int,
    indent_step: int,
) -> List[str]:
    prefix = " " * indent
    lines: List[str] = []
    op_info = node.get("operator", {})

    op_desc = _describe_operator(op_info)
    lines.append(f"{prefix}- {node['id']} :: {op_desc}")

    inputs = node.get("inputs", {})
    if inputs:
        lines.append(f"{prefix}{' ' * indent_step}inputs:")
        for port, ref in sorted(inputs.items()):
            lines.append(
                f"{prefix}{' ' * (2 * indent_step)}{port} <- {_describe_input_ref(ref)}"
            )

    outputs = node.get("outputs", [])
    if outputs:
        formatted = ", ".join(outputs)
        lines.append(f"{prefix}{' ' * indent_step}outputs: {formatted}")

    config = node.get("config", {})
    if config:
        lines.append(
            f"{prefix}{' ' * indent_step}config: {_format_mapping(config)}"
        )

    metadata = node.get("metadata", {})
    if metadata:
        lines.append(
            f"{prefix}{' ' * indent_step}metadata: {_format_mapping(metadata)}"
        )

    if op_info.get("type") == "inline_graph" and op_info.get("graph"):
        nested_lines = _format_graph_dict(
            op_info["graph"],
            indent=indent + 3 * indent_step,
            indent_step=indent_step,
        )
        lines.append(f"{prefix}{' ' * indent_step}nested:")
        lines.extend(nested_lines)

    return lines


def _describe_operator(op: Mapping[str, Any]) -> str:
    op_type = op.get("type")
    name = op.get("name")

    if op_type == "inline_graph":
        return f"inline graph [{name or 'anonymous'}]"

    if op_type == "registered":
        return f"operator:{name}"

    if op_type == "template":
        return f"template:{name}"

    if op_type == "object":
        return f"object:{name or op.get('repr')}"

    return op_type or "unknown"


def _describe_input_ref(ref: Mapping[str, Any]) -> str:
    ref_type = ref.get("type")
    if ref_type == "graph_input":
        return f"input({ref['name']})"
    if ref_type == "node_output":
        return f"{ref['node']}.{ref['port']}"
    if ref_type == "default":
        return f"default={repr(ref.get('value'))}"
    if ref_type == "unbound":
        return "<unbound>"
    return "<unknown>"


def _format_mapping(mapping: Mapping[str, Any]) -> str:
    pairs = [f"{key}={repr(value)}" for key, value in sorted(mapping.items())]
    return ", ".join(pairs) if pairs else "{}"


# Backwards compatibility -----------------------------------------------------------------


def print_module(*args: Any, **kwargs: Any) -> None:
    """Alias retained for older codepaths."""
    print_runtime(*args, **kwargs)
