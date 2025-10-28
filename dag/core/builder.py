"""
Compilation pipeline: GraphSpec -> GraphTemplate -> ExecutionPlan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Set, TYPE_CHECKING

from .nodes import GraphTemplate, NodeShell, NodeTemplate, NodeRuntime
from .ports import (
    ParameterSpec,
    ParameterRefValue,
    ensure_mapping,
    _NO_DEFAULT,
)
from .runtime.plan import ExecutionPlan
from .specs import GraphSpec, parse_endpoint, SpecError

if TYPE_CHECKING:
    from .registry import OperatorRegistry


class BuildError(RuntimeError):
    """Raised when graph compilation fails."""


@dataclass
class NodeInterface:
    inputs: Dict[str, List[Tuple[str, str]]]
    outputs: Dict[str, Tuple[str, str]]


def compile_graph(
    spec: GraphSpec,
    registry: "OperatorRegistry",
) -> GraphTemplate:
    shell_index: Dict[str, List[str]] = {}
    nodes, edges, interface_map = _expand_graph(
        spec,
        registry,
        prefix=(),
        shell_index=shell_index,
        resolve_defaults=False,
    )
    inputs = _resolve_graph_inputs(spec, interface_map, prefix=())
    outputs = _resolve_graph_outputs(spec, interface_map, prefix=())
    return GraphTemplate(
        nodes=nodes,
        edges=edges,
        inputs=inputs,
        outputs=outputs,
        parameters=spec.parameters,
        metadata=dict(spec.metadata),
        shell_index={key: tuple(sorted(dict.fromkeys(value))) for key, value in shell_index.items()},
    )


def materialise(
    template: GraphTemplate,
    *,
    parameters: Optional[Mapping[str, Any]] = None,
) -> ExecutionPlan:
    resolved_parameters = _resolve_parameter_values(template.parameters, parameters)
    node_runtimes: Dict[str, NodeRuntime] = {}
    inbound: Dict[str, Dict[str, Tuple[str, str] | Tuple[str]]] = {}
    adjacency: Dict[str, Set[str]] = {}
    in_degree: Dict[str, int] = {}

    for node_id, shell in template.nodes.items():
        config = _substitute_parameter_refs(
            shell.config,
            resolved_parameters,
            template.parameters,
            node_id=node_id,
        )
        runtime = shell.template.instantiate(config=config, runtime_id=node_id)
        node_runtime = NodeRuntime(
            node_id=node_id,
            template=shell.template,
            runner=runtime,
            metadata=shell.metadata,
        )
        node_runtimes[node_id] = node_runtime
        inbound[node_id] = {}
        adjacency[node_id] = set()
        in_degree[node_id] = 0

    for src_node, src_port, dst_node, dst_port in template.edges:
        if src_node not in node_runtimes:
            raise BuildError(f"Edge source node '{src_node}' not found")
        if dst_node not in node_runtimes:
            raise BuildError(f"Edge target node '{dst_node}' not found")
        src_runtime = node_runtimes[src_node]
        dst_runtime = node_runtimes[dst_node]
        if src_port not in src_runtime.output_ports:
            raise BuildError(
                f"Edge references unknown output port '{src_port}' on node '{src_node}'"
            )
        if dst_port not in dst_runtime.input_ports:
            raise BuildError(
                f"Edge references unknown input port '{dst_port}' on node '{dst_node}'"
            )
        if dst_port in inbound[dst_node]:
            raise BuildError(
                f"Input port '{dst_port}' on node '{dst_node}' already bound"
            )
        inbound[dst_node][dst_port] = (src_node, src_port)
        adjacency[src_node].add(dst_node)
        in_degree[dst_node] += 1

    for alias, endpoints in template.inputs.items():
        for node_id, port in endpoints:
            if node_id not in inbound:
                raise BuildError(f"Graph input '{alias}' targets unknown node '{node_id}'")
            if port in inbound[node_id]:
                raise BuildError(
                    f"Input port '{port}' on node '{node_id}' already bound"
                )
            inbound[node_id][port] = (alias,)

    graph_outputs = dict(template.outputs)

    topo_order = _topological_sort(node_runtimes.keys(), adjacency, in_degree)

    return ExecutionPlan(
        template=template,
        node_runtimes=node_runtimes,
        inbound=inbound,
        graph_inputs={alias: list(endpoints) for alias, endpoints in template.inputs.items()},
        graph_outputs=graph_outputs,
        adjacency={node: set(targets) for node, targets in adjacency.items()},
        topo_order=topo_order,
        parameters=resolved_parameters,
    )


def _expand_graph(
    spec: GraphSpec,
    registry: "OperatorRegistry",
    *,
    prefix: Sequence[str],
    shell_index: Dict[str, List[str]],
    parameter_values: Optional[Mapping[str, Any]] = None,
    resolve_defaults: bool = True,
) -> Tuple[
    Dict[str, NodeShell],
    List[Tuple[str, str, str, str]],
    Dict[str, NodeInterface],
]:
    nodes: Dict[str, NodeShell] = {}
    edges: List[Tuple[str, str, str, str]] = []
    interface_map: Dict[str, NodeInterface] = {}
    if resolve_defaults:
        resolved_params = _resolve_parameter_values(
            spec.parameters,
            parameter_values,
            include_defaults=True,
        )
    else:
        resolved_params = dict(parameter_values or {})

    for node_id, node_spec in spec.nodes.items():
        path = prefix + (node_id,)
        path_key = ".".join(path)
        operator_ref = node_spec.operator
        if resolve_defaults:
            config = _substitute_parameter_refs(
                node_spec.config,
                resolved_params,
                spec.parameters,
                node_id=".".join(path),
            )
        else:
            config = dict(node_spec.config)
        metadata = dict(node_spec.metadata)

        entry_template: Optional[NodeTemplate] = None
        nested_spec: Optional[GraphSpec] = None

        if isinstance(operator_ref, NodeTemplate):
            entry_template = operator_ref
        elif isinstance(operator_ref, str):
            entry = registry.get(operator_ref)
            if isinstance(entry, NodeTemplate):
                entry_template = entry
            elif isinstance(entry, GraphSpec):
                nested_spec = entry
            else:
                raise BuildError(
                    f"Registry entry '{operator_ref}' for node '{node_id}' "
                    f"is not a template or graph spec: {type(entry)!r}"
                )
        elif isinstance(operator_ref, GraphSpec):
            nested_spec = operator_ref
        elif isinstance(operator_ref, Mapping):
            nested_spec = GraphSpec.from_dict(operator_ref)
        else:
            raise BuildError(
                f"Unsupported operator reference for node '{node_id}': {type(operator_ref)!r}"
            )

        if entry_template is not None:
            flat_id = "__".join(path)
            shell = NodeShell(
                id=flat_id,
                template=entry_template,
                config=config,
                metadata=metadata,
            )
            nodes[flat_id] = shell
            shell_index.setdefault(path_key, []).append(flat_id)
            interface_map[path_key] = NodeInterface(
                inputs={port: [(flat_id, port)] for port in entry_template.input_ports},
                outputs={port: (flat_id, port) for port in entry_template.output_ports},
            )
            continue

        assert nested_spec is not None
        nested_overrides: Dict[str, Any] = {}
        nested_config = dict(config)
        if "parameters" in nested_config:
            nested_overrides.update(
                ensure_mapping(nested_config.pop("parameters"), "parameters")
            )
        if "init" in nested_config:
            nested_overrides.update(
                ensure_mapping(nested_config.pop("init"), "init")
            )
        if nested_config:
            raise BuildError(
                f"Unsupported config keys for graph operator '{path_key}': "
                f"{', '.join(sorted(nested_config))}"
            )
        sub_nodes, sub_edges, sub_interfaces = _expand_graph(
            nested_spec,
            registry,
            prefix=path,
            shell_index=shell_index,
            parameter_values=nested_overrides,
            resolve_defaults=True,
        )
        nodes.update(sub_nodes)
        edges.extend(sub_edges)

        flat_prefix = "__".join(path)
        shell_index.setdefault(path_key, []).extend(
            node_name for node_name in sub_nodes if node_name.startswith(flat_prefix)
        )

        inputs_map: Dict[str, List[Tuple[str, str]]] = {}
        for alias, endpoint_spec in nested_spec.inputs.items():
            endpoints = endpoint_spec
            if isinstance(endpoint_spec, (list, tuple)):
                endpoints = list(endpoint_spec)
            else:
                endpoints = [endpoint_spec]
            resolved: List[Tuple[str, str]] = []
            for endpoint in endpoints:
                inner_node, inner_port = parse_endpoint(endpoint)
                resolved.append(
                    (
                        "__".join(path + (inner_node,)),
                        inner_port,
                    )
                )
            inputs_map[alias] = resolved

        outputs_map: Dict[str, Tuple[str, str]] = {}
        for alias, endpoint in nested_spec.outputs.items():
            inner_node, inner_port = parse_endpoint(endpoint)
            outputs_map[alias] = (
                "__".join(path + (inner_node,)),
                inner_port,
            )

        interface_map[path_key] = NodeInterface(
            inputs=inputs_map,
            outputs=outputs_map,
        )

    # Build edges for this level using interface map
    for edge in spec.edges:
        (src_node, src_port), (dst_node, dst_port) = edge.unpack()
        src_key = ".".join(prefix + (src_node,))
        dst_key = ".".join(prefix + (dst_node,))
        if src_key not in interface_map:
            raise BuildError(f"Edge source '{src_node}' in graph missing after expansion")
        if dst_key not in interface_map:
            raise BuildError(f"Edge target '{dst_node}' in graph missing after expansion")
        try:
            src_endpoint = interface_map[src_key].outputs[src_port]
        except KeyError as exc:
            raise BuildError(
                f"Output port '{src_port}' not found on node '{src_node}'"
            ) from exc
        dst_options = interface_map[dst_key].inputs.get(dst_port)
        if not dst_options:
            raise BuildError(
                f"Input port '{dst_port}' not found on node '{dst_node}'"
            )
        if len(dst_options) != 1:
            raise BuildError(
                f"Input port '{dst_port}' on node '{dst_node}' is multi-bound after expansion"
            )
        dst_endpoint = dst_options[0]
        edges.append(
            (
                src_endpoint[0],
                src_endpoint[1],
                dst_endpoint[0],
                dst_endpoint[1],
            )
        )

    return nodes, edges, interface_map


def _resolve_graph_inputs(
    spec: GraphSpec,
    interface_map: Mapping[str, NodeInterface],
    *,
    prefix: Sequence[str],
) -> Dict[str, List[Tuple[str, str]]]:
    result: Dict[str, List[Tuple[str, str]]] = {}
    for alias, endpoints in spec.inputs.items():
        bound_points = endpoints if isinstance(endpoints, (list, tuple)) else [endpoints]
        resolved: List[Tuple[str, str]] = []
        for endpoint in bound_points:
            node_id, port = parse_endpoint(endpoint)
            key = ".".join(prefix + (node_id,))
            if key not in interface_map:
                raise BuildError(f"Input '{alias}' references unknown node '{node_id}'")
            port_list = interface_map[key].inputs.get(port)
            if not port_list:
                raise BuildError(
                    f"Input '{alias}' references unknown port '{port}' on node '{node_id}'"
                )
            resolved.extend(port_list)
        result[alias] = resolved
    return result


def _resolve_graph_outputs(
    spec: GraphSpec,
    interface_map: Mapping[str, NodeInterface],
    *,
    prefix: Sequence[str],
) -> Dict[str, Tuple[str, str]]:
    result: Dict[str, Tuple[str, str]] = {}
    for alias, endpoint in spec.outputs.items():
        node_id, port = parse_endpoint(endpoint)
        key = ".".join(prefix + (node_id,))
        if key not in interface_map:
            raise BuildError(
                f"Output '{alias}' references unknown node '{node_id}'"
            )
        try:
            result[alias] = interface_map[key].outputs[port]
        except KeyError as exc:
            raise BuildError(
                f"Output '{alias}' references unknown port '{port}' on node '{node_id}'"
            ) from exc
    return result


def _topological_sort(
    nodes: Iterable[str],
    adjacency: Mapping[str, Set[str]],
    in_degree: MutableMapping[str, int],
) -> List[str]:
    node_list = list(nodes)
    queue: List[str] = [node for node in node_list if in_degree[node] == 0]
    order: List[str] = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbour in adjacency.get(node, set()):
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(order) != len(node_list):
        raise BuildError("Graph contains a cycle and cannot be executed")
    return order


def _resolve_parameter_values(
    parameters: Mapping[str, ParameterSpec],
    overrides: Optional[Mapping[str, Any]],
    *,
    include_defaults: bool = True,
) -> Dict[str, Any]:
    resolved: Dict[str, Any] = {}
    overrides_map = dict(overrides or {})
    for name, value in overrides_map.items():
        if name not in parameters:
            raise BuildError(
                f"Graph parameter '{name}' is not defined on specification"
        )
        resolved[name] = value

    if include_defaults:
        for name, param in parameters.items():
            if name not in resolved and not param.required:
                resolved[name] = param.default

    return resolved


def _substitute_parameter_refs(
    value: Any,
    resolved: Mapping[str, Any],
    parameters: Mapping[str, ParameterSpec],
    *,
    node_id: str,
) -> Any:
    if isinstance(value, ParameterRefValue):
        param_name = value.name
        if param_name not in parameters:
            raise BuildError(
                f"Node '{node_id}' references unknown parameter '{param_name}'"
            )
        if param_name not in resolved:
            if value.default is not _NO_DEFAULT:
                return value.default
            raise BuildError(
                f"Parameter '{param_name}' required by node '{node_id}' "
                "is missing a value"
            )
        return resolved[param_name]

    if isinstance(value, Mapping):
        return {
            key: _substitute_parameter_refs(val, resolved, parameters, node_id=node_id)
            for key, val in value.items()
        }

    if isinstance(value, tuple):
        return tuple(
            _substitute_parameter_refs(item, resolved, parameters, node_id=node_id)
            for item in value
        )

    if isinstance(value, list):
        return [
            _substitute_parameter_refs(item, resolved, parameters, node_id=node_id)
            for item in value
        ]

    if isinstance(value, set):
        return {
            _substitute_parameter_refs(item, resolved, parameters, node_id=node_id)
            for item in value
        }

    return value


__all__ = [
    "BuildError",
    "compile_graph",
    "materialise",
]
