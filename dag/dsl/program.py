"""Program builder and evaluator for the DAG DSL."""

from __future__ import annotations

import inspect
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from ..core.ports import ParameterRefValue, ParameterSpec
from ..core.registry import registry_default
from ..core.specs import EdgeSpec, GraphSpec, NodeSpec
from .ast import BindingDecl, GraphDecl, NodeDecl
from .invocation import (
    OperatorInvocation,
    OperatorsProxy,
    RefInvocation,
    _RefResolver,
    op,
)
from .parser import _DSLParser


class DSLEvaluationError(RuntimeError):
    """Raised when runtime evaluation of DSL expressions fails."""


class _ParamNamespace:
    def __init__(self, graph: GraphDecl, defaults: Dict[str, Any]):
        self._graph = graph
        self._defaults = defaults

    def _ensure_declared(self, name: str) -> None:
        if not any(param.name == name for param in self._graph.parameters):
            raise DSLEvaluationError(
                f"Unknown parameter '{name}' referenced in graph '{self._graph.name}'"
            )

    def get(self, name: str) -> ParameterRefValue:
        self._ensure_declared(name)
        return ParameterRefValue(name=name)

    def with_default(self, name: str, value: Any) -> ParameterRefValue:
        self._ensure_declared(name)
        if name in self._defaults:
            existing = self._defaults[name]
            if existing != value:
                raise DSLEvaluationError(
                    f"Conflicting default for parameter '{name}' in graph '{self._graph.name}'"
                )
        else:
            self._defaults[name] = value
        return ParameterRefValue(name=name, default=value)

    def __getattr__(self, name: str) -> ParameterRefValue:
        return self.get(name)


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

        parameter_defaults: Dict[str, Any] = {}
        param_namespace = _ParamNamespace(graph, parameter_defaults)
        for param in graph.parameters:
            if param.default_expr is not None:
                parameter_defaults[param.name] = self._evaluate_default(
                    param.default_expr,
                    globals_ctx,
                    locals_ctx,
                    param_namespace=param_namespace,
                )

        for output in graph.outputs:
            outputs_map[output.alias] = output.source

        for node in graph.nodes:
            operator_value = self._evaluate_operator_expr(
                node.operator_expr, globals_ctx, locals_ctx, param_namespace=param_namespace
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
                    param_namespace=param_namespace,
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
        parameters_spec: Dict[str, ParameterSpec] = {}
        for param in graph.parameters:
            if param.name in parameter_defaults:
                parameters_spec[param.name] = ParameterSpec(
                    name=param.name,
                    default=parameter_defaults[param.name],
                )
            else:
                parameters_spec[param.name] = ParameterSpec(name=param.name)

        return GraphSpec(
            nodes=nodes,
            edges=edges_spec,
            parameters=parameters_spec,
            inputs=compact_inputs,
            outputs=outputs_spec,
            metadata=dict(graph.metadata),
        )

    def _evaluate_operator_expr(
        self,
        expr: str,
        globals_ctx: Mapping[str, Any],
        locals_ctx: Mapping[str, Any],
        *,
        param_namespace: Optional[_ParamNamespace] = None,
    ) -> Any:
        self._refresh_ops_namespace()
        eval_globals, eval_locals, _ = self._build_eval_environment(
            globals_ctx,
            locals_ctx,
            param_namespace=param_namespace,
            include_operator_helpers=True,
        )
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
            config_map = dict(value.config)
            if value.init_args:
                init_cfg = dict(config_map.get("init", {}))
                init_cfg.update(value.init_args)
                config_map["init"] = init_cfg
            return nested, config_map, value.metadata
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
        *,
        param_namespace: Optional[_ParamNamespace] = None,
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
            default_value = self._evaluate_default(
                binding.default_expr,
                globals_ctx,
                locals_ctx,
                param_namespace=param_namespace,
            )
            node_config.setdefault("call", {})[target_port] = default_value

    def _evaluate_default(
        self,
        expr: str,
        globals_ctx: Mapping[str, Any],
        locals_ctx: Mapping[str, Any],
        *,
        param_namespace: Optional[_ParamNamespace] = None,
    ) -> Any:
        eval_globals, eval_locals, _ = self._build_eval_environment(
            globals_ctx,
            locals_ctx,
            param_namespace=param_namespace,
            include_operator_helpers=False,
        )
        try:
            return eval(expr, eval_globals, eval_locals)  # pylint: disable=eval-used
        except Exception as exc:
            raise DSLEvaluationError(
                f"Failed to evaluate default expression '{expr}': {exc}"
            ) from exc

    def _prepare_operator_expr(
        self,
        expr: str,
        globals_ctx: Mapping[str, Any],
        locals_ctx: Mapping[str, Any],
    ) -> str:
        expression = expr.strip()
        if not expression:
            raise DSLEvaluationError("Empty operator expression")

        expression = _rewrite_parameter_placeholders(expression)
        expression = re.sub(r"(?<![=!<>])=\s*:", "=", expression)

        expression = expression.replace("(*)", "()")
        expression = re.sub(r"\(\s*\)", "()", expression)
        expression = re.sub(r"\[\s*\]", "", expression)

        token_match = re.match(r"([A-Za-z_][A-Za-z0-9_\.]*)", expression)
        if not token_match:
            raise DSLEvaluationError(f"Malformed operator expression '{expr}'")
        token = token_match.group(1)
        rest = expression[len(token) :]

        def _ensure_call_prefix(prefix: str, remainder: str) -> str:
            if re.match(r"\s*\(", remainder):
                return f"{prefix}{remainder}"
            return f"{prefix}(){remainder}"

        if token.startswith("Ref."):
            expression = _ensure_call_prefix(token, rest)
        elif token.startswith("ops."):
            op_name = token[len("ops.") :]
            if not op_name or not self._ops_namespace.has(op_name):
                raise DSLEvaluationError(
                    f"Unknown operator '{op_name or '<missing>'}'. Use 'ops.<name>' or register it first."
                )
            expression = _ensure_call_prefix(token, rest)
        else:
            name = token
            if not self._ops_namespace.has(name):
                raise DSLEvaluationError(
                    f"Unknown operator '{name}'. Use 'ops.{name}' or register it first."
                )
            expression = _ensure_call_prefix(f"ops.{name}", rest)

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

    def _build_eval_environment(
        self,
        globals_ctx: Mapping[str, Any],
        locals_ctx: Mapping[str, Any],
        *,
        param_namespace: Optional[_ParamNamespace] = None,
        include_operator_helpers: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], _ParamNamespace]:
        eval_globals = dict(globals_ctx)
        eval_locals = dict(locals_ctx)
        eval_locals["ops"] = self._ops_namespace
        if include_operator_helpers:
            eval_locals["op"] = op
            eval_locals["Ref"] = _RefResolver(self)
        if param_namespace is None:
            param_namespace = _ParamNamespace(GraphDecl(name="<anonymous>"), {})
        eval_locals.setdefault("Param", param_namespace)
        return eval_globals, eval_locals, param_namespace


def _rewrite_parameter_placeholders(expression: str) -> str:
    result: List[str] = []
    length = len(expression)
    i = 0
    while i < length:
        if expression.startswith("Param.", i):
            start = i + len("Param.")
            if start >= length or not (expression[start].isalpha() or expression[start] == "_"):
                result.append("Param.")
                i = start
                continue
            end = start
            while end < length and (expression[end].isalnum() or expression[end] == "_"):
                end += 1
            name = expression[start:end]
            idx = end
            while idx < length and expression[idx].isspace():
                idx += 1
            if idx < length and expression[idx] == ":":
                idx += 1
                default_expr, next_index = _extract_default_expression(expression, idx)
                result.append(f"Param.with_default('{name}', {default_expr})")
                i = next_index
                continue
            result.append(f"Param.get('{name}')")
            i = end
        else:
            result.append(expression[i])
            i += 1
    return "".join(result)


def _extract_default_expression(expression: str, start: int) -> Tuple[str, int]:
    depth_paren = depth_bracket = depth_brace = 0
    i = start
    length = len(expression)
    while i < length:
        ch = expression[i]
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            if depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
                break
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_bracket += 1
        elif ch == "]":
            if depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
                break
            depth_bracket = max(0, depth_bracket - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            if depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
                break
            depth_brace = max(0, depth_brace - 1)
        elif ch == "," and depth_paren == 0 and depth_bracket == 0 and depth_brace == 0:
            break
        i += 1
    segment = expression[start:i].strip()
    return segment, i


def parse_dsl(
    text: str,
    *,
    globals: Optional[Mapping[str, Any]] = None,
    locals: Optional[Mapping[str, Any]] = None,
) -> DSLProgram:
    if globals is None or locals is None:
        frame = inspect.currentframe()
        try:
            caller = frame.f_back if frame is not None else None
            if caller is not None:
                if globals is None:
                    globals = caller.f_globals
                if locals is None:
                    locals = caller.f_locals
        finally:
            del frame
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
    "DSLEvaluationError",
    "parse_dsl",
    "op",
]
