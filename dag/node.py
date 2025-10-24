"""
Declarative DAG core module.

This module introduces a declaration/execution split for building directed
acyclic computation graphs. Users describe graphs with lightweight specs
(`NodeSpec`, `EdgeSpec`, `GraphSpec`) and let a builder materialise runnable
graphs. Operators (functions, classes, or even other graphs) are registered in
an `OperatorRegistry`, which automatically infers input/output ports through
Python introspection.
"""

from __future__ import annotations

import inspect
from collections import defaultdict, deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Set,
)

from .dbg import Debug, get_debug_state


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DAGError(Exception):
    """Base class for all DAG related errors."""


class RegistrationError(DAGError):
    """Raised when operator registration fails."""


class GraphBuildError(DAGError):
    """Raised when a graph specification is invalid or inconsistent."""


class ExecutionError(DAGError):
    """Raised when graph execution fails."""


# ---------------------------------------------------------------------------
# Port and operator declarations
# ---------------------------------------------------------------------------

_NO_DEFAULT = object()


@dataclass(frozen=True)
class PortDefinition:
    """Description of a node input/output port."""

    name: str
    type: Any = Any
    default: Any = _NO_DEFAULT
    description: Optional[str] = None

    @property
    def required(self) -> bool:
        return self.default is _NO_DEFAULT


class OperatorRunner(Debug):
    """Runtime wrapper around an operator implementation."""

    def __init__(self, *, name: str):
        super().__init__()
        self.name = name

    def compute(self, **kwargs) -> Mapping[str, Any]:
        raise NotImplementedError

    def __call__(self, **kwargs) -> Dict[str, Any]:
        if get_debug_state():
            self._start_timer()
        result = self.compute(**kwargs)
        if get_debug_state():
            self._stop_timer()
        if not isinstance(result, Mapping):
            raise ExecutionError(
                f"Operator '{self.name}' returned non-mapping output: {type(result)!r}"
            )
        return dict(result)


class FunctionRunner(OperatorRunner):
    """Runner for plain Python callables."""

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str,
        output_keys: Sequence[str],
        call_defaults: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(name=name)
        self._func = func
        self._output_keys = list(output_keys)
        self._call_defaults = dict(call_defaults or {})

    def compute(self, **kwargs) -> Mapping[str, Any]:
        call_kwargs = {**self._call_defaults, **kwargs}
        result = self._func(**call_kwargs)
        return _normalise_output(result, self._output_keys)


class ClassRunner(OperatorRunner):
    """Runner for class instances exposing a forward-like method."""

    def __init__(
        self,
        *,
        instance: Any,
        forward: Callable[..., Any],
        name: str,
        output_keys: Sequence[str],
        call_defaults: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(name=name)
        self._instance = instance
        self._forward = forward
        self._output_keys = list(output_keys)
        self._call_defaults = dict(call_defaults or {})

    def compute(self, **kwargs) -> Mapping[str, Any]:
        call_kwargs = {**self._call_defaults, **kwargs}
        result = self._forward(**call_kwargs)
        return _normalise_output(result, self._output_keys)


class GraphOperatorRunner(OperatorRunner):
    """Runner that delegates execution to an embedded graph runtime."""

    def __init__(
        self,
        *,
        runtime: "GraphRuntime",
        name: str,
        output_keys: Sequence[str],
    ):
        super().__init__(name=name)
        self._runtime = runtime
        self._output_keys = list(output_keys)

    def compute(self, **kwargs) -> Mapping[str, Any]:
        return self._runtime.run(kwargs, outputs=self._output_keys)


@dataclass(frozen=True)
class OperatorTemplate:
    """Factory descriptor for runnable operator instances."""

    name: str
    create_runner: Callable[[Mapping[str, Any], str], OperatorRunner]
    input_ports: Mapping[str, PortDefinition] = field(default_factory=dict)
    output_ports: Mapping[str, PortDefinition] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def instantiate(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        runtime_id: Optional[str] = None,
    ) -> OperatorRunner:
        runner = self.create_runner(dict(config or {}), runtime_id or self.name)
        if not isinstance(runner, OperatorRunner):
            raise RegistrationError(
                f"Operator template '{self.name}' produced an invalid runner: "
                f"{type(runner)!r}"
            )
        return runner

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
        outputs: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "OperatorTemplate":
        if not callable(func):
            raise RegistrationError(f"Expected callable, got {type(func)!r}")

        signature = inspect.signature(func)
        input_ports = _infer_input_ports(signature)
        output_ports, output_keys = _infer_output_ports(
            func, explicit=outputs
        )

        def factory(config: Mapping[str, Any], runtime_id: str) -> OperatorRunner:
            call_defaults = dict(config)
            return FunctionRunner(
                func,
                name=runtime_id or func.__name__,
                output_keys=output_keys,
                call_defaults=call_defaults,
            )

        return cls(
            name=name or func.__name__,
            create_runner=factory,
            input_ports=MappingProxyType(input_ports),
            output_ports=MappingProxyType(output_ports),
            metadata=MappingProxyType(dict(metadata or {})),
        )

    @classmethod
    def from_class(
        cls,
        operator_cls: type,
        *,
        name: Optional[str] = None,
        forward: str = "forward",
        outputs: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "OperatorTemplate":
        if not inspect.isclass(operator_cls):
            raise RegistrationError(f"Expected class type, got {type(operator_cls)!r}")

        if not hasattr(operator_cls, forward):
            raise RegistrationError(
                f"Operator class '{operator_cls.__name__}' missing '{forward}' method"
            )

        forward_fn = getattr(operator_cls, forward)
        forward_sig = inspect.signature(forward_fn)
        init_sig = inspect.signature(operator_cls.__init__)

        input_ports = _infer_input_ports(forward_sig, skip_first=True)
        output_ports, output_keys = _infer_output_ports(
            forward_fn, explicit=outputs
        )
        init_defaults = _infer_default_kwargs(init_sig, skip_first=True)

        def factory(config: Mapping[str, Any], runtime_id: str) -> OperatorRunner:
            cfg = dict(config)
            init_kwargs = dict(init_defaults)
            call_defaults: Dict[str, Any] = {}

            if "init" in cfg:
                init_kwargs.update(_ensure_mapping(cfg.pop("init"), "init"))
            if "call" in cfg:
                call_defaults.update(_ensure_mapping(cfg.pop("call"), "call"))

            init_kwargs.update(cfg)

            instance = operator_cls(**init_kwargs)
            forward_method = getattr(instance, forward)
            return ClassRunner(
                instance=instance,
                forward=forward_method,
                name=runtime_id or operator_cls.__name__,
                output_keys=output_keys,
                call_defaults=call_defaults,
            )

        return cls(
            name=name or operator_cls.__name__,
            create_runner=factory,
            input_ports=MappingProxyType(input_ports),
            output_ports=MappingProxyType(output_ports),
            metadata=MappingProxyType(dict(metadata or {})),
        )


# ---------------------------------------------------------------------------
# Graph specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeSpec:
    """Declarative node descriptor."""

    id: str
    operator: str
    config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeSpec:
    """Declarative edge descriptor (src and dst use '<node>.<port>' syntax)."""

    src: str
    dst: str

    def unpack(self) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        return _parse_endpoint(self.src), _parse_endpoint(self.dst)


@dataclass(frozen=True)
class GraphSpec:
    """Top-level graph declaration."""

    nodes: Mapping[str, NodeSpec]
    edges: Sequence[EdgeSpec]
    inputs: Mapping[str, str] = field(default_factory=dict)
    outputs: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GraphSpec":
        nodes_raw = data.get("nodes", {})
        edges_raw = data.get("edges", [])
        inputs = dict(data.get("inputs", {}))
        outputs = dict(data.get("outputs", {}))
        metadata = dict(data.get("metadata", {}))

        nodes: Dict[str, NodeSpec] = {}
        for node_id, spec in nodes_raw.items():
            if isinstance(spec, NodeSpec):
                nodes[node_id] = spec
            else:
                nodes[node_id] = NodeSpec(
                    id=node_id,
                    operator=spec["operator"],
                    config=dict(spec.get("config", {})),
                    metadata=dict(spec.get("metadata", {})),
                )

        edges: List[EdgeSpec] = []
        for spec in edges_raw:
            if isinstance(spec, EdgeSpec):
                edges.append(spec)
            else:
                edges.append(EdgeSpec(src=spec["src"], dst=spec["dst"]))

        if not outputs:
            raise GraphBuildError("Graph specification requires at least one output")

        return cls(
            nodes=nodes,
            edges=edges,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )


@dataclass(frozen=True)
class GraphInputRef:
    """Reference to a named graph input."""

    name: str


@dataclass(frozen=True)
class NodeOutputRef:
    """Reference to a node output port."""

    node_id: str
    port: str

# ---------------------------------------------------------------------------
# Graph runtime
# ---------------------------------------------------------------------------


class NodeRuntime:
    """Runtime node holding instantiated operator."""

    def __init__(
        self,
        node_id: str,
        template: OperatorTemplate,
        runner: OperatorRunner,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        self.id = node_id
        self.template = template
        self.runner = runner
        self.metadata = dict(metadata or {})
        self.input_ports = dict(template.input_ports)
        self.output_ports = dict(template.output_ports)
        cache_flag = self.metadata.get("cache_enabled", True)
        if isinstance(cache_flag, bool):
            self.cache_enabled = cache_flag
        else:
            self.cache_enabled = bool(cache_flag)
        self._cache_valid = False
        self._cache_inputs: Optional[Dict[str, Any]] = None
        self._cache_outputs: Optional[Dict[str, Any]] = None

    def execute(
        self,
        inputs: Mapping[str, Any],
        *,
        use_cache: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        inputs_dict = dict(inputs)
        if self.cache_enabled and use_cache and not force:
            if self._cache_valid and self._cache_inputs == inputs_dict:
                return dict(self._cache_outputs or {})

        outputs = self.runner(**inputs_dict)
        outputs_dict = dict(outputs)

        if self.cache_enabled:
            self._cache_valid = True
            self._cache_inputs = inputs_dict
            self._cache_outputs = outputs_dict
        else:
            self.clear_cache()

        return dict(outputs_dict)

    def clear_cache(self) -> None:
        self._cache_valid = False
        self._cache_inputs = None
        self._cache_outputs = None


class GraphRuntime:
    """Executable graph produced from a GraphSpec."""

    def __init__(
        self,
        *,
        spec: GraphSpec,
        node_runtimes: Mapping[str, NodeRuntime],
        inbound: Mapping[str, Dict[str, Union[GraphInputRef, NodeOutputRef]]],
        graph_inputs: Mapping[str, Tuple[str, str]],
        graph_outputs: Mapping[str, NodeOutputRef],
        adjacency: Mapping[str, Set[str]],
        topo_order: Sequence[str],
    ):
        self.spec = spec
        self.node_runtimes = dict(node_runtimes)
        self._inbound = {
            node_id: dict(port_map) for node_id, port_map in inbound.items()
        }
        self.graph_inputs = dict(graph_inputs)
        self.graph_outputs = dict(graph_outputs)
        self._adjacency = {node: set(targets) for node, targets in adjacency.items()}
        self.topological_order = list(topo_order)

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
                f"Missing required graph inputs: {', '.join(sorted(missing_inputs))}"
            )

        requested = list(outputs) if outputs is not None else list(self.graph_outputs)
        undefined = [name for name in requested if name not in self.graph_outputs]
        if undefined:
            raise ExecutionError(
                f"Requested outputs not defined: {', '.join(sorted(undefined))}"
            )

        force_set = set(force_nodes or [])

        state = _ExecutionState(self, inputs, use_cache=use_cache, force_nodes=force_set)
        state.execute()

        results: Dict[str, Any] = {}
        for alias in requested:
            out_ref = self.graph_outputs[alias]
            node_outputs = state.node_values[out_ref.node_id]
            if out_ref.port not in node_outputs:
                raise ExecutionError(
                    f"Node '{out_ref.node_id}' did not produce output '{out_ref.port}'"
                )
            results[alias] = node_outputs[out_ref.port]
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
        """Return a serialisable view of the runtime for inspection/UI."""
        return {
            "nodes": {
                node_id: {
                    "operator": runtime.template.name,
                    "inputs": list(runtime.input_ports.keys()),
                    "outputs": list(runtime.output_ports.keys()),
                    "metadata": runtime.metadata,
                }
                for node_id, runtime in self.node_runtimes.items()
            },
            "edges": [
                {
                    "src": f"{ref.node_id}.{ref.port}",
                    "dst": f"{node_id}.{port}",
                }
                for node_id, port_map in self._inbound.items()
                for port, ref in port_map.items()
                if isinstance(ref, NodeOutputRef)
            ],
            "inputs": dict(self.graph_inputs),
            "outputs": {
                alias: f"{ref.node_id}.{ref.port}"
                for alias, ref in self.graph_outputs.items()
            },
        }


class _ExecutionState:
    """Per-run execution helper."""

    def __init__(
        self,
        runtime: GraphRuntime,
        external_inputs: Mapping[str, Any],
        *,
        use_cache: bool,
        force_nodes: Set[str],
    ):
        self.runtime = runtime
        self.external_inputs = dict(external_inputs)
        self.node_values: Dict[str, Dict[str, Any]] = {}
        self.use_cache = use_cache
        self.force_nodes = force_nodes

    def execute(self) -> None:
        for node_id in self.runtime.topological_order:
            runtime_node = self.runtime.node_runtimes[node_id]
            kwargs = self._collect_inputs(node_id, runtime_node)
            outputs = runtime_node.execute(
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
        inbound = self.runtime._inbound.get(node_id, {})
        kwargs: Dict[str, Any] = {}

        for port_name, ref in inbound.items():
            if isinstance(ref, GraphInputRef):
                kwargs[port_name] = self.external_inputs[ref.name]
            else:
                producer_outputs = self.node_values[ref.node_id]
                kwargs[port_name] = producer_outputs[ref.port]

        for port_name, port_def in runtime_node.input_ports.items():
            if port_name not in kwargs and not port_def.required:
                kwargs[port_name] = port_def.default

        return kwargs


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


class GraphBuilder:
    """Materialises GraphRuntime instances from specs."""

    def __init__(self, registry: Optional["OperatorRegistry"] = None):
        self.registry = registry or registry_default

    def build(self, spec: GraphSpec) -> GraphRuntime:
        if not spec.nodes:
            raise GraphBuildError("GraphSpec must contain at least one node")

        node_runtimes: Dict[str, NodeRuntime] = {}
        inbound: Dict[str, Dict[str, Union[GraphInputRef, NodeOutputRef]]] = defaultdict(dict)
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        in_degree: Dict[str, int] = defaultdict(int)

        for node_id, node_spec in spec.nodes.items():
            if node_id in node_runtimes:
                raise GraphBuildError(f"Duplicate node id '{node_id}' in GraphSpec")
            template = self.registry.get(node_spec.operator)
            runner = template.instantiate(config=node_spec.config, runtime_id=node_id)
            node_runtimes[node_id] = NodeRuntime(
                node_id=node_id,
                template=template,
                runner=runner,
                metadata=node_spec.metadata,
            )
            in_degree[node_id] = 0

        graph_inputs: Dict[str, Tuple[str, str]] = {}

        for alias, endpoint in spec.inputs.items():
            node_id, port = _parse_endpoint(endpoint)
            if node_id not in node_runtimes:
                raise GraphBuildError(
                    f"Graph input '{alias}' targets unknown node '{node_id}'"
                )
            node_runtime = node_runtimes[node_id]
            if port not in node_runtime.input_ports:
                raise GraphBuildError(
                    f"Graph input '{alias}' targets unknown input port "
                    f"'{port}' on node '{node_id}'"
                )
            if port in inbound[node_id]:
                raise GraphBuildError(
                    f"Input port '{port}' on node '{node_id}' already bound"
                )
            inbound[node_id][port] = GraphInputRef(alias)
            graph_inputs[alias] = (node_id, port)

        for edge in spec.edges:
            (src_node, src_port), (dst_node, dst_port) = edge.unpack()
            if src_node not in node_runtimes:
                raise GraphBuildError(f"Edge source node '{src_node}' not found")
            if dst_node not in node_runtimes:
                raise GraphBuildError(f"Edge target node '{dst_node}' not found")

            src_runtime = node_runtimes[src_node]
            dst_runtime = node_runtimes[dst_node]

            if src_port not in src_runtime.output_ports:
                raise GraphBuildError(
                    f"Edge references unknown output port '{src_port}' on node '{src_node}'"
                )
            if dst_port not in dst_runtime.input_ports:
                raise GraphBuildError(
                    f"Edge references unknown input port '{dst_port}' on node '{dst_node}'"
                )
            if dst_port in inbound[dst_node]:
                raise GraphBuildError(
                    f"Input port '{dst_port}' on node '{dst_node}' already bound"
                )

            inbound[dst_node][dst_port] = NodeOutputRef(src_node, src_port)
            adjacency[src_node].add(dst_node)
            in_degree[dst_node] += 1

        # Validate required ports
        for node_id, runtime in node_runtimes.items():
            for port_name, port_def in runtime.input_ports.items():
                if port_name not in inbound[node_id] and port_def.required:
                    raise GraphBuildError(
                        f"Required input '{port_name}' on node '{node_id}' "
                        "is not connected and has no default value"
                    )

        if not spec.outputs:
            raise GraphBuildError("GraphSpec must declare at least one output")

        graph_outputs: Dict[str, NodeOutputRef] = {}
        for alias, endpoint in spec.outputs.items():
            node_id, port = _parse_endpoint(endpoint)
            if node_id not in node_runtimes:
                raise GraphBuildError(
                    f"Graph output '{alias}' targets unknown node '{node_id}'"
                )
            node_runtime = node_runtimes[node_id]
            if port not in node_runtime.output_ports:
                raise GraphBuildError(
                    f"Graph output '{alias}' references unknown port '{port}' "
                    f"on node '{node_id}'"
                )
            graph_outputs[alias] = NodeOutputRef(node_id, port)

        topo_order = _topological_sort(node_runtimes.keys(), adjacency, in_degree)

        return GraphRuntime(
            spec=spec,
            node_runtimes=node_runtimes,
            inbound=inbound,
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
            adjacency=adjacency,
            topo_order=topo_order,
        )


# ---------------------------------------------------------------------------
# Operator registry and decorators
# ---------------------------------------------------------------------------


class OperatorRegistry:
    """Registry mapping operator names to templates."""

    def __init__(self):
        self._operators: Dict[str, OperatorTemplate] = {}

    def register(self, template: OperatorTemplate) -> OperatorTemplate:
        if template.name in self._operators:
            raise RegistrationError(
                f"Operator '{template.name}' already registered"
            )
        self._operators[template.name] = template
        return template

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

        template = OperatorTemplate.from_function(
            func, name=name, outputs=outputs, metadata=metadata
        )
        self.register(template)
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

        template = OperatorTemplate.from_class(
            cls,
            name=name,
            forward=forward,
            outputs=outputs,
            metadata=metadata,
        )
        self.register(template)
        return cls

    def register_graph(
        self,
        name: str,
        graph_spec: GraphSpec,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> OperatorTemplate:
        input_defs = {
            alias: PortDefinition(name=alias)
            for alias in graph_spec.inputs.keys()
        }
        output_defs = {}
        for alias in graph_spec.outputs.keys():
            output_defs[alias] = PortDefinition(name=alias)

        def factory(config: Mapping[str, Any], runtime_id: str) -> OperatorRunner:
            builder = GraphBuilder(self)
            runtime = builder.build(graph_spec)
            return GraphOperatorRunner(
                runtime=runtime,
                name=runtime_id or name,
                output_keys=list(graph_spec.outputs.keys()),
            )

        template = OperatorTemplate(
            name=name,
            create_runner=factory,
            input_ports=MappingProxyType(input_defs),
            output_ports=MappingProxyType(output_defs),
            metadata=MappingProxyType(dict(metadata or {})),
        )
        return self.register(template)

    def get(self, name: str) -> OperatorTemplate:
        try:
            return self._operators[name]
        except KeyError as exc:
            raise RegistrationError(f"Unknown operator '{name}'") from exc

    def __contains__(self, name: str) -> bool:  # pragma: no cover
        return name in self._operators

    def items(self) -> Iterable[Tuple[str, OperatorTemplate]]:  # pragma: no cover
        return self._operators.items()


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
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> OperatorTemplate:
    return registry_default.register_graph(
        name, graph_spec, metadata=metadata
    )


def build_graph(
    spec: GraphSpec,
    *,
    registry: Optional[OperatorRegistry] = None,
) -> GraphRuntime:
    builder = GraphBuilder(registry or registry_default)
    return builder.build(spec)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_input_ports(
    signature: inspect.Signature,
    *,
    skip_first: bool = False,
) -> Dict[str, PortDefinition]:
    parameters = list(signature.parameters.values())
    if skip_first and parameters:
        parameters = parameters[1:]

    ports: Dict[str, PortDefinition] = {}
    for param in parameters:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise RegistrationError(
                "Variable positional or keyword arguments are not supported "
                "in operator signatures"
            )
        annotation = param.annotation if param.annotation is not inspect._empty else Any
        default = param.default if param.default is not inspect._empty else _NO_DEFAULT
        ports[param.name] = PortDefinition(
            name=param.name,
            type=annotation,
            default=default,
        )
    return ports


def _infer_output_ports(
    obj: Callable[..., Any],
    *,
    explicit: Optional[Union[Sequence[str], Mapping[str, Any]]] = None,
) -> Tuple[Dict[str, PortDefinition], List[str]]:
    if explicit is not None:
        ports = _normalise_output_definitions(explicit)
        return ports, list(ports.keys())

    annotated = getattr(obj, "__dag_returns__", None)
    if annotated is not None:
        ports = _normalise_output_definitions(annotated)
        return ports, list(ports.keys())

    return {
        "_return": PortDefinition(name="_return"),
    }, ["_return"]


def _infer_default_kwargs(
    signature: inspect.Signature,
    *,
    skip_first: bool = False,
) -> Dict[str, Any]:
    parameters = list(signature.parameters.values())
    if skip_first and parameters:
        parameters = parameters[1:]

    defaults: Dict[str, Any] = {}
    for param in parameters:
        if param.default is not inspect._empty:
            defaults[param.name] = param.default
    return defaults


def _parse_endpoint(value: str) -> Tuple[str, str]:
    if "." not in value:
        raise GraphBuildError(
            f"Endpoint '{value}' must use '<node>.<port>' notation"
        )
    node_id, port = value.split(".", 1)
    if not node_id or not port:
        raise GraphBuildError(
            f"Endpoint '{value}' must include both node id and port"
        )
    return node_id, port


def _normalise_output_definitions(
    value: Union[Sequence[str], Mapping[str, Any]]
) -> Dict[str, PortDefinition]:
    if isinstance(value, Mapping):
        return {
            name: PortDefinition(
                name=name,
                type=type_hint if type_hint is not None else Any,
            )
            for name, type_hint in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {name: PortDefinition(name=name) for name in value}
    raise RegistrationError(
        "Output definition should be a mapping (name->type) or sequence of names"
    )


def _normalise_output(
    value: Any,
    output_keys: Sequence[str],
) -> Dict[str, Any]:
    keys = list(output_keys)
    if isinstance(value, Mapping):
        if keys and any(key not in value for key in keys):
            missing = [key for key in keys if key not in value]
            raise ExecutionError(
                f"Operator result missing keys: {', '.join(missing)}"
            )
        return dict(value)

    if hasattr(value, "_asdict"):
        mapped = value._asdict()  # type: ignore[attr-defined]
        return _normalise_output(mapped, keys)

    if len(keys) == 1:
        return {keys[0]: value}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != len(keys):
            raise ExecutionError(
                f"Expected {len(keys)} outputs, received {len(value)}"
            )
        return dict(zip(keys, value))

    raise ExecutionError(
        "Unable to normalise operator output; provide explicit "
        "returns_keys() metadata or return a mapping."
    )


def _ensure_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise RegistrationError(f"Expected mapping for '{label}' configuration")


def _topological_sort(
    nodes: Iterable[str],
    adjacency: Mapping[str, Set[str]],
    in_degree: MutableMapping[str, int],
) -> List[str]:
    node_list = list(nodes)
    queue = deque(node for node in node_list if in_degree[node] == 0)
    order: List[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in adjacency.get(node, set()):
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(order) != len(node_list):
        raise GraphBuildError("Graph contains a cycle and cannot be executed")
    return order


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
    return a + b


@register_function(name="multiplication")
@returns_keys(result=float)
def multiplication(a: float, b: float) -> float:
    return a * b


__all__ = [
    "Constant",
    "GraphBuilder",
    "GraphRuntime",
    "GraphSpec",
    "NodeSpec",
    "EdgeSpec",
    "OperatorRegistry",
    "OperatorTemplate",
    "PortDefinition",
    "build_graph",
    "register_class",
    "register_function",
    "register_graph",
    "returns_keys",
    "registry_default",
]
