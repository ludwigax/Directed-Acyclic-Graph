"""
Runtime execution plan for DAG nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from ..nodes import GraphTemplate, NodeRuntime


class ExecutionError(RuntimeError):
    """Raised when execution of the plan fails."""


Hook = Callable[..., Any]


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
    _pre_hooks: Dict[str, List[Hook]] = field(init=False, default_factory=dict, repr=False)
    _post_hooks: Dict[str, List[Hook]] = field(init=False, default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(
        self,
        inputs: Optional[Mapping[str, Any]] = None,
        *,
        outputs: Optional[Sequence[str]] = None,
        use_cache: bool = True,
        force_nodes: Optional[Iterable[str]] = None,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        inputs = dict(inputs or {})
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

        if max_workers is not None and max_workers < 1:
            raise ExecutionError("max_workers must be positive")

        force_set = set(self._resolve_targets(force_nodes, default_all=False))

        topo_index = {node: idx for idx, node in enumerate(self.topo_order)}
        dependency_counts: Dict[str, int] = {
            node: sum(1 for ref in self.inbound.get(node, {}).values() if len(ref) == 2)
            for node in self.node_runtimes
        }

        ready_heap: List[Tuple[int, str]] = []
        for node in self.topo_order:
            if dependency_counts.get(node, 0) == 0:
                heapq.heappush(ready_heap, (topo_index[node], node))

        node_values: Dict[str, Dict[str, Any]] = {}
        in_flight: Dict[Any, str] = {}
        completed: Set[str] = set()

        pool_kwargs = {}
        if max_workers is not None:
            pool_kwargs["max_workers"] = max_workers

        with ThreadPoolExecutor(**pool_kwargs) as executor:
            while ready_heap or in_flight:
                # submit as many ready nodes as allowed
                while ready_heap and (max_workers is None or len(in_flight) < max_workers):
                    _, node_id = heapq.heappop(ready_heap)
                    arguments = self._collect_inputs_for_node(node_id, inputs, node_values)
                    future = executor.submit(
                        self._execute_node_worker,
                        node_id,
                        arguments,
                        use_cache,
                        node_id in force_set,
                    )
                    in_flight[future] = node_id

                if not in_flight:
                    break

                done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    node_id = in_flight.pop(future)
                    try:
                        outputs_map = future.result()
                    except Exception as exc:  # pragma: no cover - propagate worker failures
                        for pending_future in in_flight:
                            pending_future.cancel()
                        raise exc

                    node_values[node_id] = outputs_map
                    completed.add(node_id)

                    dependents = self.adjacency.get(node_id, set())
                    for dependent in sorted(dependents, key=topo_index.__getitem__):
                        dependency_counts[dependent] -= 1
                        if dependency_counts[dependent] == 0:
                            heapq.heappush(ready_heap, (topo_index[dependent], dependent))

        if len(completed) != len(self.node_runtimes):
            missing = set(self.node_runtimes) - completed
            raise ExecutionError(
                f"Execution did not finish for nodes: {', '.join(sorted(missing))}"
            )

        results: Dict[str, Any] = {}
        for alias in requested:
            node_id, port = self.graph_outputs[alias]
            node_outputs = node_values[node_id]
            if port not in node_outputs:
                raise ExecutionError(
                    f"Node '{node_id}' did not produce output '{port}'"
                )
            results[alias] = node_outputs[port]
        return results

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self, target: Optional[Union[str, Iterable[str]]] = None) -> None:
        for node_id in self._resolve_targets(target, default_all=True):
            self.node_runtimes[node_id].clear_cache()

    def set_cache(
        self,
        target: Optional[Union[str, Iterable[str]]] = None,
        *,
        enabled: bool,
        clear: bool = False,
    ) -> None:
        for node_id in self._resolve_targets(target, default_all=True):
            runtime_node = self.node_runtimes[node_id]
            runtime_node.cache_enabled = bool(enabled)
            if clear or not enabled:
                runtime_node.clear_cache()

    def enable_cache(self, target: Optional[Union[str, Iterable[str]]] = None) -> None:
        self.set_cache(target, enabled=True)

    def disable_cache(self, target: Optional[Union[str, Iterable[str]]] = None) -> None:
        self.set_cache(target, enabled=False)

    def get_cached_outputs(
        self,
        target: Union[str, Iterable[str]],
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        nodes = self._resolve_targets(target, default_all=False)
        if len(nodes) != 1:
            raise ExecutionError("get_cached_outputs expects a single node or path resolving to one node")
        return self.node_runtimes[nodes[0]].get_cached()

    def set_cached_outputs(
        self,
        target: Union[str, Iterable[str]],
        *,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
    ) -> None:
        nodes = self._resolve_targets(target, default_all=False)
        if len(nodes) != 1:
            raise ExecutionError("set_cached_outputs expects a single node or path resolving to one node")
        self.node_runtimes[nodes[0]].set_cache(inputs, outputs)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def register_hook(
        self,
        target: Union[str, Iterable[str]],
        *,
        when: str = "post",
        hook: Hook,
    ) -> None:
        if when not in {"pre", "post"}:
            raise ExecutionError("Hook 'when' must be 'pre' or 'post'")
        nodes = self._resolve_targets(target, default_all=False)
        store = self._pre_hooks if when == "pre" else self._post_hooks
        for node_id in nodes:
            store.setdefault(node_id, []).append(hook)

    def clear_hooks(
        self,
        target: Optional[Union[str, Iterable[str]]] = None,
        *,
        when: Optional[str] = None,
    ) -> None:
        nodes = self._resolve_targets(target, default_all=True)
        if when is not None and when not in {"pre", "post"}:
            raise ExecutionError("Hook 'when' must be 'pre', 'post', or None")
        for node_id in nodes:
            if when in (None, "pre"):
                self._pre_hooks.pop(node_id, None)
            if when in (None, "post"):
                self._post_hooks.pop(node_id, None)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def resolve_path(self, path: str) -> Tuple[str, ...]:
        nodes = self.template.shell_index.get(path)
        if nodes is None:
            raise ExecutionError(f"Unknown path '{path}'")
        return nodes

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
                    "config": dict(template.nodes[node_id].config),
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
                for src_node, src_port in [ref]
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_targets(
        self,
        target: Optional[Union[str, Iterable[str]]],
        *,
        default_all: bool,
    ) -> List[str]:
        if target is None:
            return list(self.node_runtimes.keys()) if default_all else []

        if isinstance(target, str):
            if target in self.node_runtimes:
                return [target]
            nodes = self.template.shell_index.get(target)
            if nodes:
                return list(nodes)
            raise ExecutionError(f"Unknown node or path '{target}'")

        if isinstance(target, Iterable):
            result: List[str] = []
            for item in target:
                result.extend(self._resolve_targets(item, default_all=False))
            # dedupe while preserving order
            return list(dict.fromkeys(result))

        raise ExecutionError(f"Unsupported target type: {type(target)!r}")

    def _apply_pre_hooks(
        self,
        node_id: str,
        inputs: Mapping[str, Any],
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        adjusted_inputs = dict(inputs)
        override_outputs: Optional[Dict[str, Any]] = None
        for hook in self._pre_hooks.get(node_id, []):
            result = hook(node_id=node_id, inputs=dict(adjusted_inputs), plan=self)
            if result is None:
                continue
            if isinstance(result, tuple):
                new_inputs, maybe_outputs = result
                if new_inputs is not None:
                    adjusted_inputs = dict(new_inputs)
                if maybe_outputs is not None:
                    override_outputs = dict(maybe_outputs)
            elif isinstance(result, Mapping):
                adjusted_inputs = dict(result)
            else:
                raise ExecutionError(
                    f"Pre-hook for node '{node_id}' returned unsupported value {type(result)!r}"
                )
        return adjusted_inputs, override_outputs

    def _apply_post_hooks(
        self,
        node_id: str,
        inputs: Mapping[str, Any],
        outputs: Mapping[str, Any],
    ) -> Dict[str, Any]:
        adjusted_outputs = dict(outputs)
        for hook in self._post_hooks.get(node_id, []):
            result = hook(
                node_id=node_id,
                inputs=dict(inputs),
                outputs=dict(adjusted_outputs),
                plan=self,
            )
            if result is None:
                continue
            if isinstance(result, Mapping):
                adjusted_outputs = dict(result)
            else:
                raise ExecutionError(
                    f"Post-hook for node '{node_id}' returned unsupported value {type(result)!r}"
                )
        return adjusted_outputs


    def _collect_inputs_for_node(
        self,
        node_id: str,
        external_inputs: Mapping[str, Any],
        node_values: Mapping[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        inbound = self.inbound.get(node_id, {})
        kwargs: Dict[str, Any] = {}

        for port_name, ref in inbound.items():
            if isinstance(ref, tuple) and len(ref) == 1:
                (input_name,) = ref
                kwargs[port_name] = external_inputs[input_name]
            else:
                src_node, src_port = ref  # type: ignore[misc]
                producer_outputs = node_values[src_node]
                kwargs[port_name] = producer_outputs[src_port]

        runtime_node = self.node_runtimes[node_id]
        for port_name, port_def in runtime_node.input_ports.items():
            if port_name not in kwargs and not port_def.required:
                kwargs[port_name] = port_def.default

        return kwargs

    def _execute_node_worker(
        self,
        node_id: str,
        inputs: Mapping[str, Any],
        use_cache: bool,
        force: bool,
    ) -> Dict[str, Any]:
        kwargs, override_outputs = self._apply_pre_hooks(node_id, inputs)

        runtime_node = self.node_runtimes[node_id]
        if override_outputs is not None:
            outputs = dict(override_outputs)
        else:
            outputs = runtime_node.run(kwargs, use_cache=use_cache, force=force)

        outputs = self._apply_post_hooks(node_id, kwargs, outputs)
        runtime_node.update_cache(kwargs, outputs)
        return outputs


__all__ = [
    "ExecutionError",
    "ExecutionPlan",
]
