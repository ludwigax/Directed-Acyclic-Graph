"""
Runtime execution plan for DAG nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from ..nodes import GraphTemplate, NodeRuntime


class ExecutionError(RuntimeError):
    """Raised when execution of the plan fails."""


@dataclass
class ExecutionPlan:
    """Executable plan produced from a compiled template."""

    template: GraphTemplate
    node_runtimes: Mapping[str, NodeRuntime]
    inbound: Mapping[str, Dict[str, Tuple[str, str] | Tuple[str]]]
    graph_inputs: Mapping[str, List[Tuple[str, str]]]
    graph_outputs: Mapping[str, Tuple[str, str]]
    adjacency: Mapping[str, Set[str]]
    topo_order: Sequence[str]
    parameters: Mapping[str, Any]

    def run(
        self,
        inputs: Optional[Mapping[str, Any]] = None,
        *,
        outputs: Optional[Sequence[str]] = None,
        use_cache: bool = True,
        force_nodes: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        inputs = inputs or {}
        missing_inputs = [
            name for name in self.graph_inputs if name not in inputs
        ]
        if missing_inputs:
            raise ExecutionError(
                f"Missing required plan inputs: {', '.join(sorted(missing_inputs))}"
            )

        requested = list(outputs) if outputs is not None else list(self.graph_outputs)
        undefined = [name for name in requested if name not in self.graph_outputs]
        if undefined:
            raise ExecutionError(
                f"Requested outputs not defined: {', '.join(sorted(undefined))}"
            )

        force_set = set(force_nodes or [])

        state = _ExecutionContext(self, inputs, use_cache=use_cache, force_nodes=force_set)
        state.execute()

        results: Dict[str, Any] = {}
        for alias in requested:
            node_id, port = self.graph_outputs[alias]
            node_outputs = state.node_values[node_id]
            if port not in node_outputs:
                raise ExecutionError(
                    f"Node '{node_id}' did not produce output '{port}'"
                )
            results[alias] = node_outputs[port]
        return results

    def clear_cache(self, node_id: Optional[str] = None) -> None:
        if node_id is None:
            for runtime_node in self.node_runtimes.values():
                runtime_node.clear_cache()
        else:
            if node_id not in self.node_runtimes:
                raise ExecutionError(f"Unknown node '{node_id}' when clearing cache")
            self.node_runtimes[node_id].clear_cache()

    def describe(self) -> Dict[str, Any]:
        """Return a serialisable view of the plan for inspection/UI."""
        template = self.template
        return {
            "parameters": dict(self.parameters),
            "nodes": {
                node_id: {
                    "operator": runtime.template.name,
                    "inputs": list(runtime.input_ports.keys()),
                    "outputs": list(runtime.output_ports.keys()),
                    "metadata": runtime.metadata,
                    "config": dict(template.nodes[node_id].config)
                    }
                for node_id, runtime in self.node_runtimes.items()
            },
            "edges": [
                {
                    "src": f"{src_node}.{src_port}",
                    "dst": f"{node_id}.{port}",
                }
                for node_id, port_map in self.inbound.items()
                for port, ref in port_map.items()
                if isinstance(ref, tuple) and len(ref) == 2
                for src_node, src_port in [ref]  # unpack tuple
            ],
            "inputs": {
                alias: [f"{node}.{port}" for node, port in endpoints]
                for alias, endpoints in self.graph_inputs.items()
            },
            "outputs": {
                alias: f"{node}.{port}"
                for alias, (node, port) in self.graph_outputs.items()
            },
            "metadata": dict(template.metadata),
            "shell_index": {
                key: list(value) for key, value in template.shell_index.items()
            },
        }


class _ExecutionContext:
    """Per-run execution helper."""

    def __init__(
        self,
        plan: ExecutionPlan,
        external_inputs: Mapping[str, Any],
        *,
        use_cache: bool,
        force_nodes: Set[str],
    ):
        self.plan = plan
        self.external_inputs = dict(external_inputs)
        self.node_values: Dict[str, Dict[str, Any]] = {}
        self.use_cache = use_cache
        self.force_nodes = force_nodes

    def execute(self) -> None:
        for node_id in self.plan.topo_order:
            runtime_node = self.plan.node_runtimes[node_id]
            kwargs = self._collect_inputs(node_id, runtime_node)
            outputs = runtime_node.run(
                kwargs,
                use_cache=self.use_cache,
                force=node_id in self.force_nodes,
            )
            self.node_values[node_id] = outputs

    def _collect_inputs(
        self,
        node_id: str,
        runtime_node: NodeRuntime,
    ) -> Dict[str, Any]:
        inbound = self.plan.inbound.get(node_id, {})
        kwargs: Dict[str, Any] = {}

        for port_name, ref in inbound.items():
            if isinstance(ref, tuple) and len(ref) == 1:
                (input_name,) = ref
                kwargs[port_name] = self.external_inputs[input_name]
            else:
                src_node, src_port = ref  # type: ignore[misc]
                producer_outputs = self.node_values[src_node]
                kwargs[port_name] = producer_outputs[src_port]

        for port_name, port_def in runtime_node.input_ports.items():
            if port_name not in kwargs and not port_def.required:
                kwargs[port_name] = port_def.default

        return kwargs


__all__ = [
    "ExecutionError",
    "ExecutionPlan",
]
