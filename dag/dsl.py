"""Domain specific language for declaratively describing graph specs."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from .node import EdgeSpec, GraphSpec, NodeSpec, registry_default


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------


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
class GraphDecl:
    name: str
    inputs: List[str] = field(default_factory=list)
    outputs: List[OutputDecl] = field(default_factory=list)
    nodes: List[NodeDecl] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Operator helpers
# ---------------------------------------------------------------------------


@dataclass
class OperatorInvocation:
    operator: Any
    config: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass
class RefInvocation:
    graph_name: str
    config: Mapping[str, Any]
    metadata: Mapping[str, Any]


def op(
    operator: Any,
    *,
    config: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    init: Optional[Mapping[str, Any]] = None,
    call: Optional[Mapping[str, Any]] = None,
    **init_kwargs: Any,
) -> OperatorInvocation:
    config_map: Dict[str, Any] = {}
    if config:
        config_map.update(config)
    if init or init_kwargs:
        config_map.setdefault("init", {}).update(dict(init or {}))
        config_map["init"].update(init_kwargs)
    if call:
        config_map.setdefault("call", {}).update(dict(call))
    return OperatorInvocation(
        operator=operator,
        config=config_map,
        metadata=dict(metadata or {}),
    )


class OperatorsProxy:
    """Namespace providing registered operators as callables."""

    def __init__(self, registry: Mapping[str, Any]):
        self._registry = dict(registry)
        self._cache: Dict[str, Any] = {}

    def refresh(self, registry: Mapping[str, Any]) -> None:
        for name, operator in registry.items():
            if self._registry.get(name) is not operator:
                self._registry[name] = operator
                self._cache.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._registry

    def __getattr__(self, name: str):
        if name in self._cache:
            return self._cache[name]
        operator_ref = self._registry.get(name, name)

        def _call(**kwargs: Any) -> OperatorInvocation:
            return op(operator_ref, **kwargs)

        self._cache[name] = _call
        return _call


class _RefHandle:
    def __init__(self, owner: "DSLProgram", graph_name: str):
        self._owner = owner
        self._graph_name = graph_name

    def __call__(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        **config_kwargs: Any,
    ) -> RefInvocation:
        cfg: Dict[str, Any] = {}
        if config:
            cfg.update(config)
        if config_kwargs:
            cfg.update(config_kwargs)
        return RefInvocation(
            graph_name=self._graph_name,
            config=cfg,
            metadata=dict(metadata or {}),
        )


class _RefResolver:
    def __init__(self, owner: "DSLProgram"):
        self._owner = owner

    def __getattr__(self, name: str) -> _RefHandle:
        if name not in self._owner.graph_names:
            raise AttributeError(f"Unknown graph reference '{name}'")
        return _RefHandle(self._owner, name)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class DSLParseError(SyntaxError):
    pass


class _DSLParser:
    def __init__(self, text: str):
        self.lines = text.splitlines()
        self.index = 0
        self.length = len(self.lines)

    def parse(self) -> List[GraphDecl]:
        graphs: List[GraphDecl] = []
        while self.index < self.length:
            if not self._current_stripped():
                self.index += 1
                continue
            graph = self._parse_graph()
            graphs.append(graph)
        return graphs

    def _parse_graph(self) -> GraphDecl:
        line = self._current_line()
        stripped = line.strip()
        if self._indent_of(line) != 0 or not stripped.startswith("graph "):
            raise self._error("Expected 'graph <name>:'")
        if not stripped.endswith(":"):
            raise self._error("Graph declaration must end with ':'")
        payload = stripped[6:-1].strip()
        if not payload:
            raise self._error("Graph declaration missing name")
        metadata: Mapping[str, Any] = {}
        if "(" in payload:
            name_part, meta_part = payload.split("(", 1)
            name = name_part.strip()
            metadata = self._parse_metadata(meta_part.rstrip(")"), self.index + 1)
        else:
            name = payload
        self._validate_identifier(name, "graph name")
        graph = GraphDecl(name=name, metadata=metadata)
        self.index += 1
        block_indent = self._expect_indented_block()
        while self.index < self.length:
            line = self._current_line()
            stripped = line.strip()
            if not stripped:
                self.index += 1
                continue
            indent = self._indent_of(line)
            if indent < block_indent:
                break
            if indent > block_indent:
                raise self._error("Inconsistent indentation")
            if stripped.startswith("input "):
                self._parse_input(graph, stripped)
            elif stripped.startswith("output "):
                self._parse_output(graph, stripped)
            else:
                self._parse_node(graph, stripped)
            self.index += 1
        return graph

    def _parse_input(self, graph: GraphDecl, stripped: str) -> None:
        payload = stripped[len("input "):].strip()
        if not payload:
            raise self._error("Input declaration requires names")
        for alias in payload.replace(",", " ").split():
            self._validate_identifier(alias, "input")
            if alias in graph.inputs:
                raise self._error(f"Duplicate input '{alias}'")
            graph.inputs.append(alias)

    def _parse_output(self, graph: GraphDecl, stripped: str) -> None:
        payload = stripped[len("output "):].strip()
        if not payload:
            raise self._error("Output declaration requires a source")
        if "=" in payload:
            alias_part, source_part = payload.split("=", 1)
            alias = alias_part.strip()
            source = source_part.strip()
            self._validate_identifier(alias, "output alias")
        else:
            source = payload.strip()
            alias = source.split(".")[-1] if "." in source else source
            self._validate_identifier(alias, "output alias")
        if "." not in source:
            source = f"{source}._return"
        graph.outputs.append(OutputDecl(alias=alias, source=source, line=self.index + 1))

    def _parse_node(self, graph: GraphDecl, stripped: str) -> None:
        if "=" not in stripped:
            raise self._error("Expected assignment 'node = operator[...]'")
        lhs, rhs = stripped.split("=", 1)
        name = lhs.strip()
        self._validate_identifier(name, "node name")
        if any(node.name == name for node in graph.nodes):
            raise self._error(f"Duplicate node '{name}'")
        operator_expr, bindings_expr = self._split_operator_bindings(rhs.strip())
        bindings = self._parse_bindings(bindings_expr)
        graph.nodes.append(
            NodeDecl(
                name=name,
                operator_expr=operator_expr,
                bindings=bindings,
                metadata={},
                line=self.index + 1,
            )
        )

    def _split_operator_bindings(self, rhs: str) -> Tuple[str, str]:
        if "[" not in rhs:
            return self._normalise_operator_expr(rhs), ""
        if not rhs.endswith("]"):
            raise self._error("Bindings must end with ']'")
        operator_expr, bindings_expr = rhs[:-1].split("[", 1)
        return self._normalise_operator_expr(operator_expr.strip()), bindings_expr.strip()

    @staticmethod
    def _normalise_operator_expr(expr: str) -> str:
        return expr.replace("(*)", "()")

    def _parse_bindings(self, bindings_str: str) -> List[BindingDecl]:
        if not bindings_str:
            return []
        bindings: List[BindingDecl] = []
        for part in bindings_str.split(","):
            piece = part.strip()
            if not piece:
                continue
            if "=" not in piece:
                raise self._error("Binding must use 'port=source' syntax")
            port_part, value_part = piece.split("=", 1)
            port = port_part.strip()
            self._validate_identifier(port, "port")
            value = value_part.strip()
            default_expr = None
            source = None
            if ":" in value:
                source_section, default_section = value.split(":", 1)
                source = source_section.strip() or None
                default_expr = default_section.strip() or None
            else:
                source = value or None
            if source is None and default_expr is None:
                raise self._error("Binding must specify a source or default")
            bindings.append(
                BindingDecl(
                    port=port,
                    source=source,
                    default_expr=default_expr,
                    line=self.index + 1,
                )
            )
        return bindings

    def _parse_metadata(self, meta_str: str, line: int) -> Mapping[str, Any]:
        meta: Dict[str, Any] = {}
        for chunk in meta_str.split(","):
            piece = chunk.strip()
            if not piece:
                continue
            if "=" not in piece:
                raise DSLParseError(f"Invalid metadata pair '{piece}' (line {line})")
            key, value = piece.split("=", 1)
            key = key.strip()
            if not key:
                raise DSLParseError(f"Metadata key cannot be empty (line {line})")
            meta[key] = value.strip()
        return meta

    def _expect_indented_block(self) -> int:
        while self.index < self.length:
            line = self._current_line()
            if not line.strip():
                self.index += 1
                continue
            indent = self._indent_of(line)
            if indent == 0:
                raise self._error("Expected indented block")
            return indent
        raise self._error("Unexpected EOF while parsing graph body")

    def _current_line(self) -> str:
        return self.lines[self.index]

    def _current_stripped(self) -> str:
        return self.lines[self.index].strip()

    @staticmethod
    def _indent_of(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def _validate_identifier(self, name: str, label: str) -> None:
        if not name:
            raise self._error(f"{label} cannot be empty")
        if not (name[0].isalpha() or name[0] == "_") or not all(
            ch.isalnum() or ch == "_" for ch in name
        ):
            raise self._error(f"Invalid {label} '{name}'")

    def _error(self, message: str) -> DSLParseError:
        return DSLParseError(f"{message} (line {self.index + 1})")


# ---------------------------------------------------------------------------
# Program builder
# ---------------------------------------------------------------------------


class DSLEvaluationError(RuntimeError):
    pass


class DSLProgram:
    def __init__(
        self,
        graphs: Iterable[GraphDecl],
        *,
        globals_ctx: Optional[Mapping[str, Any]] = None,
        locals_ctx: Optional[Mapping[str, Any]] = None,
        operator_registry: Optional[Mapping[str, Any]] = None,
    ):
        self._graphs: Dict[str, GraphDecl] = {graph.name: graph for graph in graphs}
        self._globals = dict(globals_ctx or {})
        self._locals = dict(locals_ctx or {})
        registry_mapping = operator_registry or dict(registry_default.items())
        self._ops_namespace = OperatorsProxy(registry_mapping)
        self._uses_default_registry = operator_registry is None
        self._compiled: Dict[str, GraphSpec] = {}
        self._compile_stack: List[str] = []

    def _refresh_ops_namespace(self) -> None:
        if self._uses_default_registry:
            self._ops_namespace.refresh(dict(registry_default.items()))

    @property
    def graph_names(self) -> List[str]:
        return list(self._graphs.keys())

    def build(
        self,
        name: str,
        *,
        globals_ctx: Optional[Mapping[str, Any]] = None,
        locals_ctx: Optional[Mapping[str, Any]] = None,
    ) -> GraphSpec:
        if name in self._compiled:
            return self._compiled[name]
        if name not in self._graphs:
            raise KeyError(f"Unknown graph '{name}'")
        if name in self._compile_stack:
            chain = " -> ".join(self._compile_stack + [name])
            raise RuntimeError(f"Recursive graph reference detected: {chain}")
        self._compile_stack.append(name)
        try:
            combined_globals = dict(self._globals)
            if globals_ctx:
                combined_globals.update(globals_ctx)
            combined_globals.setdefault("__builtins__", __builtins__)
            combined_locals = dict(self._locals)
            if locals_ctx:
                combined_locals.update(locals_ctx)
            self._refresh_ops_namespace()
            graph = self._graphs[name]
            spec = self._compile_graph(graph, combined_globals, combined_locals)
            self._compiled[name] = spec
            return spec
        finally:
            self._compile_stack.pop()

    def _compile_graph(
        self,
        graph: GraphDecl,
        globals_ctx: Mapping[str, Any],
        locals_ctx: Mapping[str, Any],
    ) -> GraphSpec:
        nodes: Dict[str, NodeSpec] = {}
        edges: List[EdgeSpec] = []
        inputs_map: Dict[str, List[str]] = {alias: [] for alias in graph.inputs}
        outputs_map: Dict[str, str] = {}

        for output in graph.outputs:
            outputs_map[output.alias] = output.source

        for node in graph.nodes:
            operator_value = self._evaluate_operator_expr(
                node.operator_expr, globals_ctx, locals_ctx
            )
            operator_ref, node_config, node_metadata = self._normalise_operator(operator_value)
            config_copy = dict(node_config)
            for binding in node.bindings:
                self._handle_binding(
                    graph,
                    node,
                    binding,
                    inputs_map,
                    edges,
                    config_copy,
                    globals_ctx,
                    locals_ctx,
                )

            nodes[node.name] = NodeSpec(
                id=node.name,
                operator=operator_ref,
                config=config_copy,
                metadata=dict(node_metadata),
            )

        compact_inputs = self._finalise_inputs(graph, inputs_map)
        edges_spec = [
            EdgeSpec(src=f"{src_node}.{src_port}", dst=f"{dst_node}.{dst_port}")
            for (src_node, src_port, dst_node, dst_port) in edges
        ]
        outputs_spec = {alias: source for alias, source in outputs_map.items()}

        return GraphSpec(
            nodes=nodes,
            edges=edges_spec,
            inputs=compact_inputs,
            outputs=outputs_spec,
            metadata=dict(graph.metadata),
        )

    def _evaluate_operator_expr(
        self,
        expr: str,
        globals_ctx: Mapping[str, Any],
        locals_ctx: Mapping[str, Any],
    ) -> Any:
        self._refresh_ops_namespace()
        eval_globals = dict(globals_ctx)
        eval_locals = dict(locals_ctx)
        eval_locals["op"] = op
        eval_locals["ops"] = self._ops_namespace
        eval_locals["Ref"] = _RefResolver(self)
        prepared_expr = self._prepare_operator_expr(expr, eval_globals, eval_locals)
        try:
            return eval(prepared_expr, eval_globals, eval_locals)  # pylint: disable=eval-used
        except Exception as exc:
            raise DSLEvaluationError(f"Failed to evaluate '{expr}': {exc}") from exc

    def _normalise_operator(
        self,
        value: Any,
    ) -> Tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
        if isinstance(value, OperatorInvocation):
            return value.operator, value.config, value.metadata
        if isinstance(value, RefInvocation):
            nested = self.build(value.graph_name)
            return nested, value.config, value.metadata
        return value, {}, {}

    def _handle_binding(
        self,
        graph: GraphDecl,
        node: NodeDecl,
        binding: BindingDecl,
        inputs_map: MutableMapping[str, List[str]],
        edges: List[Tuple[str, str, str, str]],
        node_config: Dict[str, Any],
        globals_ctx: Mapping[str, Any],
        locals_ctx: Mapping[str, Any],
    ) -> None:
        target_port = binding.port
        if binding.source:
            source = binding.source
            if "." in source:
                src_node, src_port = source.split(".", 1)
                edges.append((src_node, src_port, node.name, target_port))
            else:
                if source not in inputs_map:
                    raise DSLEvaluationError(
                        f"Unknown graph input '{source}' referenced on line {binding.line}"
                    )
                inputs_map[source].append(f"{node.name}.{target_port}")
        if binding.default_expr is not None:
            default_value = self._evaluate_default(binding.default_expr, globals_ctx, locals_ctx)
            node_config.setdefault("call", {})[target_port] = default_value

    def _evaluate_default(
        self,
        expr: str,
        globals_ctx: Mapping[str, Any],
        locals_ctx: Mapping[str, Any],
    ) -> Any:
        eval_globals = dict(globals_ctx)
        eval_locals = dict(locals_ctx)
        eval_locals.setdefault("ops", self._ops_namespace)
        try:
            return eval(expr, eval_globals, eval_locals)  # pylint: disable=eval-used
        except Exception as exc:
            raise DSLEvaluationError(f"Failed to evaluate default expression '{expr}': {exc}") from exc

    def _prepare_operator_expr(
        self,
        expr: str,
        globals_ctx: Mapping[str, Any],
        locals_ctx: Mapping[str, Any],
    ) -> str:
        expression = expr.strip()
        if not expression:
            raise DSLEvaluationError("Empty operator expression")

        # Expand bare operator names into ops.<name>()
        if expression[0].isalpha() and "(" not in expression.split()[0]:
            # Extract leading identifier
            match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", expression)
            if match:
                name = match.group(1)
                if not self._ops_namespace.has(name):
                    raise DSLEvaluationError(
                        f"Unknown operator '{name}'. Use 'ops.{name}' or register it first."
                    )
                rest = expression[len(name):].lstrip()
                expression = f"ops.{name}{rest or '()'}"

        # Normalise empty parentheses/brackets
        expression = expression.replace("(*)", "()")
        expression = re.sub(r"\(\s*\)", "()", expression)
        expression = re.sub(r"\[\s*\]", "", expression)

        return expression

    def _finalise_inputs(
        self,
        graph: GraphDecl,
        inputs_map: Mapping[str, List[str]],
    ) -> Mapping[str, Any]:
        result: Dict[str, Any] = {}
        for alias, endpoints in inputs_map.items():
            if not endpoints:
                raise DSLEvaluationError(
                    f"Input '{alias}' in graph '{graph.name}' has no bindings"
                )
            if len(endpoints) == 1:
                result[alias] = endpoints[0]
            else:
                result[alias] = endpoints
        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_dsl(
    text: str,
    *,
    globals: Optional[Mapping[str, Any]] = None,
    locals: Optional[Mapping[str, Any]] = None,
) -> DSLProgram:
    parser = _DSLParser(text)
    graphs = parser.parse()
    return DSLProgram(
        graphs,
        globals_ctx=globals,
        locals_ctx=locals,
        operator_registry=dict(registry_default.items()),
    )


__all__ = [
    "DSLProgram",
    "DSLParseError",
    "DSLEvaluationError",
    "parse_dsl",
    "op",
]



