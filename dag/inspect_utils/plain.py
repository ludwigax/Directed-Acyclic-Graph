"""Console-friendly inspection helpers for graph specs and execution plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Tuple

from ..node import ExecutionPlan
from ..core.ports import ParameterSpec
from ..core.specs import GraphSpec, NodeSpec

Serializable = Dict[str, Any]
StateLiteral = Literal["computed", "pending", "partial", "unknown"]

STATE_LABELS = {
    "computed": "computed",
    "pending": "pending",
    "partial": "partial",
    "unknown": "unknown",
}


@dataclass
class SpecTreeNode:
    """Tree node representing either a graph or a runnable node."""

    name: str
    kind: Literal["graph", "node"]
    path: Tuple[str, ...]
    display_path: str
    runtime_id: Optional[str]
    node_spec: Optional[NodeSpec]
    graph_spec: Optional[GraphSpec]
    operator_ref: Any
    node_config: Mapping[str, Any]
    node_metadata: Mapping[str, Any]
    graph_metadata: Mapping[str, Any]
    parameters: Mapping[str, ParameterSpec]
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]
    operator_name: Optional[str]
    state: StateLiteral = "pending"
    children: List["SpecTreeNode"] = field(default_factory=list)


class SpecInspector:
    """Interactive helper for exploring graph specs in text form."""

    def __init__(
        self,
        spec: GraphSpec,
        *,
        plan: Optional[ExecutionPlan] = None,
        root_name: Optional[str] = None,
    ) -> None:
        self.spec = spec
        self.plan = plan
        self.root_name = root_name or spec.metadata.get("name") or "Graph"
        self.root = self._build_graph_node(
            spec=spec,
            prefix=(),
            node_id=None,
            node_spec=None,
        )
        self._nodes_by_display: Dict[str, SpecTreeNode] = {}
        self._nodes_by_relative: Dict[str, SpecTreeNode] = {}
        self._nodes_by_runtime: Dict[str, SpecTreeNode] = {}
        self._register_node(self.root)
        self._compute_states(self.root)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def render_tree(
        self,
        *,
        show_runtime_ids: bool = True,
    ) -> str:
        """Render the spec tree as a text block."""
        lines: List[str] = []
        lines.append(self._format_node_label(self.root, show_runtime_ids))
        for index, child in enumerate(self.root.children):
            self._render_tree(
                node=child,
                lines=lines,
                prefix="",
                is_last=index == len(self.root.children) - 1,
                show_runtime_ids=show_runtime_ids,
            )
        return "\n".join(lines)

    def print_tree(
        self,
        *,
        show_runtime_ids: bool = True,
    ) -> None:
        """Pretty-print the spec tree to stdout."""
        print(self.render_tree(show_runtime_ids=show_runtime_ids))

    def list_paths(self) -> List[str]:
        """Return all known spec paths (relative to the root)."""
        paths = [
            path for path in self._nodes_by_relative.keys() if path
        ]
        return sorted(paths)

    def render_node(self, path: str) -> str:
        """Render node details for the given path."""
        node = self._locate(path)
        return "\n".join(self._format_node_details(node))

    def print_node(self, path: str) -> None:
        """Pretty-print node details for the given path."""
        print(self.render_node(path))

    # ------------------------------------------------------------------ #
    # Tree building and bookkeeping
    # ------------------------------------------------------------------ #

    def _build_graph_node(
        self,
        *,
        spec: GraphSpec,
        prefix: Tuple[str, ...],
        node_id: Optional[str],
        node_spec: Optional[NodeSpec],
    ) -> SpecTreeNode:
        if node_id is None:
            path: Tuple[str, ...] = ()
            name = self.root_name
            display_path = self.root_name
            node_metadata: Mapping[str, Any] = {}
        else:
            path = prefix + (node_id,)
            name = node_id
            display_path = self._format_display_path(path)
            node_metadata = dict(node_spec.metadata) if node_spec else {}

        node = SpecTreeNode(
            name=name,
            kind="graph",
            path=path,
            display_path=display_path,
            runtime_id=None,
            node_spec=node_spec,
            graph_spec=spec,
            operator_ref=spec,
            node_config=dict(node_spec.config) if node_spec else {},
            node_metadata=node_metadata,
            graph_metadata=dict(spec.metadata),
            parameters=dict(spec.parameters),
            inputs=dict(spec.inputs),
            outputs=dict(spec.outputs),
            operator_name=_resolve_graph_name(name, spec, node_spec),
        )

        for child_id in sorted(spec.nodes.keys()):
            child_spec = spec.nodes[child_id]
            operator_ref = child_spec.operator
            if isinstance(operator_ref, GraphSpec):
                child_node = self._build_graph_node(
                    spec=operator_ref,
                    prefix=path,
                    node_id=child_id,
                    node_spec=child_spec,
                )
            else:
                child_node = self._build_leaf_node(
                    node_spec=child_spec,
                    path=path + (child_id,),
                )
            node.children.append(child_node)

        return node

    def _build_leaf_node(
        self,
        *,
        node_spec: NodeSpec,
        path: Tuple[str, ...],
    ) -> SpecTreeNode:
        display_path = self._format_display_path(path)
        runtime_id = "__".join(path)
        operator_ref = node_spec.operator
        return SpecTreeNode(
            name=path[-1],
            kind="node",
            path=path,
            display_path=display_path,
            runtime_id=runtime_id,
            node_spec=node_spec,
            graph_spec=None,
            operator_ref=operator_ref,
            node_config=dict(node_spec.config),
            node_metadata=dict(node_spec.metadata),
            graph_metadata={},
            parameters={},
            inputs={},
            outputs={},
            operator_name=_resolve_operator_name(operator_ref),
        )

    def _register_node(self, node: SpecTreeNode) -> None:
        self._nodes_by_display[node.display_path] = node
        if node is self.root:
            self._nodes_by_display[self.root_name] = node
            self._nodes_by_relative[""] = node
        else:
            relative = ".".join(node.path)
            self._nodes_by_relative[relative] = node
        if node.runtime_id:
            self._nodes_by_runtime[node.runtime_id] = node
        for child in node.children:
            self._register_node(child)

    # ------------------------------------------------------------------ #
    # State computation
    # ------------------------------------------------------------------ #

    def _compute_states(self, node: SpecTreeNode) -> StateLiteral:
        if node.kind == "node":
            node.state = self._leaf_state(node)
            return node.state

        child_states = [self._compute_states(child) for child in node.children]
        if not child_states:
            node.state = "pending"
            return node.state

        if all(state == "computed" for state in child_states):
            node.state = "computed"
        elif all(state == "pending" for state in child_states):
            node.state = "pending"
        elif any(state == "computed" for state in child_states) or any(
            state == "partial" for state in child_states
        ):
            node.state = "partial"
        else:
            node.state = "unknown"
        return node.state

    def _leaf_state(self, node: SpecTreeNode) -> StateLiteral:
        if not self.plan or not node.runtime_id:
            return "pending"
        runtime = self.plan.node_runtimes.get(node.runtime_id)
        if runtime is None:
            return "pending"
        cached = runtime.get_cached()
        if cached is not None:
            return "computed"
        if not runtime.cache_enabled:
            return "partial"
        return "pending"

    # ------------------------------------------------------------------ #
    # Rendering helpers
    # ------------------------------------------------------------------ #

    def _render_tree(
        self,
        *,
        node: SpecTreeNode,
        lines: List[str],
        prefix: str,
        is_last: bool,
        show_runtime_ids: bool,
    ) -> None:
        branch = "└── " if is_last else "├── "
        lines.append(
            f"{prefix}{branch}{self._format_node_label(node, show_runtime_ids)}"
        )
        child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
        for index, child in enumerate(node.children):
            self._render_tree(
                node=child,
                lines=lines,
                prefix=child_prefix,
                is_last=index == len(node.children) - 1,
                show_runtime_ids=show_runtime_ids,
            )

    def _format_node_label(
        self,
        node: SpecTreeNode,
        show_runtime_ids: bool,
    ) -> str:
        state_label = STATE_LABELS.get(node.state, node.state)
        if node.kind == "graph":
            label = f"{node.display_path} (graph) [{state_label}]"
        else:
            label = f"{node.display_path} [{state_label}]"
            if show_runtime_ids and node.runtime_id:
                label += f" <{node.runtime_id}>"
            if node.operator_name:
                label += f" :: {node.operator_name}"
        return label

    def _format_node_details(self, node: SpecTreeNode) -> List[str]:
        state_label = STATE_LABELS.get(node.state, node.state)
        lines: List[str] = []
        if node.kind == "graph":
            lines.append(f"Graph {node.display_path}")
            lines.append(f"State: {state_label}")
            if node.operator_name and node.operator_name != node.name:
                lines.append(f"Display name: {node.operator_name}")
            if node.node_config:
                lines.append("Node configuration:")
                lines.extend(_indent_lines(_format_mapping_lines(node.node_config)))
            if node.node_metadata:
                lines.append("Node metadata:")
                lines.extend(_indent_lines(_format_mapping_lines(node.node_metadata)))
            if node.graph_metadata:
                lines.append("Graph metadata:")
                lines.extend(_indent_lines(_format_mapping_lines(node.graph_metadata)))
            if node.parameters:
                lines.append("Parameters:")
                lines.extend(_indent_lines(_format_parameters(node.parameters)))
            if node.inputs:
                lines.append("Inputs (outline):")
                lines.extend(_indent_lines(_format_endpoints(node.inputs)))
            if node.outputs:
                lines.append("Outputs (outline):")
                lines.extend(_indent_lines(_format_outputs(node.outputs)))
            if node.children:
                lines.append("Children:")
                for child in node.children:
                    child_state = STATE_LABELS.get(child.state, child.state)
                    suffix = ""
                    if child.kind == "node" and child.operator_name:
                        suffix = f" :: {child.operator_name}"
                    lines.append(
                        f"  - {child.display_path} ({'graph' if child.kind == 'graph' else 'node'}) [{child_state}]{suffix}"
                    )
        else:
            lines.append(f"Node {node.display_path}")
            lines.append(f"State: {state_label}")
            if node.runtime_id:
                lines.append(f"Runtime id: {node.runtime_id}")
            if node.operator_name:
                lines.append(f"Operator: {node.operator_name}")
            if node.operator_ref is not None:
                lines.append(f"Operator ref: {_describe_operator_ref(node.operator_ref)}")
            if node.node_config:
                lines.append("Config:")
                lines.extend(_indent_lines(_format_mapping_lines(node.node_config)))
            if node.node_metadata:
                lines.append("Metadata:")
                lines.extend(_indent_lines(_format_mapping_lines(node.node_metadata)))
            if self.plan and node.runtime_id:
                runtime = self.plan.node_runtimes.get(node.runtime_id)
                if runtime:
                    stats = runtime.runner.get_stats()
                    lines.append(
                        "Execution stats: "
                        f"calls={stats.get('call_count', 0)}, "
                        f"total={stats.get('execution_time', 0.0):.6f}s, "
                        f"avg={stats.get('avg_time', 0.0):.6f}s"
                    )
                    cache_state = runtime.get_cached()
                    cache_label = "valid" if cache_state is not None else "empty"
                    lines.append(
                        f"Cache: {cache_label} (enabled: {'yes' if runtime.cache_enabled else 'no'})"
                    )
        return lines

    # ------------------------------------------------------------------ #
    # Lookup helpers
    # ------------------------------------------------------------------ #

    def _locate(self, path: str) -> SpecTreeNode:
        key = (path or "").strip()
        if not key:
            return self.root

        # Direct display path
        node = self._nodes_by_display.get(key)
        if node:
            return node

        # Relative dotted path
        rel = self._normalize_relative_path(key)
        if rel is not None:
            node = self._nodes_by_relative.get(rel)
            if node:
                return node

        # Runtime ID for leaves
        node = self._nodes_by_runtime.get(key)
        if node:
            return node

        raise KeyError(f"Unknown spec path '{path}'")

    def _normalize_relative_path(self, value: str) -> Optional[str]:
        if "__" in value:
            # runtime ids are handled elsewhere
            return None
        parts = tuple(part for part in value.split(".") if part)
        if not parts:
            return ""
        if parts[0] == self.root_name:
            parts = parts[1:]
        return ".".join(parts)

    def _format_display_path(self, path: Tuple[str, ...]) -> str:
        if not path:
            return self.root_name
        return f"{self.root_name}." + ".".join(path)


# --------------------------------------------------------------------------- #
# Convenience API
# --------------------------------------------------------------------------- #


def build_spec_inspector(
    spec: GraphSpec,
    *,
    plan: Optional[ExecutionPlan] = None,
    root_name: Optional[str] = None,
) -> SpecInspector:
    """Construct a SpecInspector instance."""
    return SpecInspector(spec, plan=plan, root_name=root_name)


def render_spec_tree(
    spec: GraphSpec,
    *,
    plan: Optional[ExecutionPlan] = None,
    root_name: Optional[str] = None,
    show_runtime_ids: bool = True,
) -> str:
    inspector = build_spec_inspector(spec, plan=plan, root_name=root_name)
    return inspector.render_tree(show_runtime_ids=show_runtime_ids)


def print_spec_tree(
    spec: GraphSpec,
    *,
    plan: Optional[ExecutionPlan] = None,
    root_name: Optional[str] = None,
    show_runtime_ids: bool = True,
) -> None:
    print(
        render_spec_tree(
            spec,
            plan=plan,
            root_name=root_name,
            show_runtime_ids=show_runtime_ids,
        )
    )


def render_spec_node(
    spec: GraphSpec,
    path: str,
    *,
    plan: Optional[ExecutionPlan] = None,
    root_name: Optional[str] = None,
) -> str:
    inspector = build_spec_inspector(spec, plan=plan, root_name=root_name)
    return inspector.render_node(path)


def print_spec_node(
    spec: GraphSpec,
    path: str,
    *,
    plan: Optional[ExecutionPlan] = None,
    root_name: Optional[str] = None,
) -> None:
    print(render_spec_node(spec, path, plan=plan, root_name=root_name))


# --------------------------------------------------------------------------- #
# Legacy runtime helpers (unchanged)
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #


def _resolve_operator_name(operator_ref: Any) -> Optional[str]:
    if operator_ref is None:
        return None
    if hasattr(operator_ref, "name"):
        return getattr(operator_ref, "name")
    if hasattr(operator_ref, "__name__"):
        return operator_ref.__name__
    if isinstance(operator_ref, str):
        return operator_ref
    return type(operator_ref).__name__


def _resolve_graph_name(
    fallback: str,
    spec: GraphSpec,
    node_spec: Optional[NodeSpec],
) -> str:
    if node_spec and node_spec.metadata.get("name"):
        return str(node_spec.metadata["name"])
    if spec.metadata.get("name"):
        return str(spec.metadata["name"])
    return fallback


def _describe_operator_ref(operator_ref: Any) -> str:
    if operator_ref is None:
        return "<none>"
    if isinstance(operator_ref, GraphSpec):
        name = operator_ref.metadata.get("name") or "<graph>"
        return f"GraphSpec(name={name})"
    if hasattr(operator_ref, "name"):
        return f"{type(operator_ref).__name__}(name={operator_ref.name})"
    if hasattr(operator_ref, "__name__"):
        return f"{type(operator_ref).__name__}::{operator_ref.__name__}"
    return repr(operator_ref)


def _format_mapping(mapping: Mapping[str, Any]) -> str:
    pairs = [f"{key}={repr(value)}" for key, value in sorted(mapping.items())]
    return ", ".join(pairs) if pairs else "{}"


def _format_mapping_lines(mapping: Mapping[str, Any]) -> List[str]:
    if not mapping:
        return ["<empty>"]
    return [f"{key}: {repr(value)}" for key, value in sorted(mapping.items())]


def _format_parameters(parameters: Mapping[str, ParameterSpec]) -> List[str]:
    if not parameters:
        return ["<none>"]
    lines: List[str] = []
    for name in sorted(parameters.keys()):
        spec = parameters[name]
        if spec.required:
            lines.append(f"{name}: <required>")
        else:
            lines.append(f"{name}: default={repr(spec.default)}")
    return lines


def _format_endpoints(inputs: Mapping[str, Any]) -> List[str]:
    if not inputs:
        return ["<none>"]
    lines: List[str] = []
    for alias, endpoints in sorted(inputs.items()):
        if isinstance(endpoints, (list, tuple, set)):
            resolved = ", ".join(str(item) for item in endpoints)
        else:
            resolved = str(endpoints)
        lines.append(f"{alias}: {resolved}")
    return lines


def _format_outputs(outputs: Mapping[str, Any]) -> List[str]:
    if not outputs:
        return ["<none>"]
    lines: List[str] = []
    for alias, endpoint in sorted(outputs.items()):
        lines.append(f"{alias}: {endpoint}")
    return lines


def _indent_lines(lines: Iterable[str], *, indent: str = "  ") -> List[str]:
    return [f"{indent}{line}" for line in lines]


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
                _format_node_runtime(
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


def _format_node_runtime(
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


__all__ = [
    "SpecInspector",
    "build_spec_inspector",
    "render_spec_tree",
    "print_spec_tree",
    "render_spec_node",
    "print_spec_node",
    "runtime_to_dict",
    "render_runtime_text",
    "print_runtime",
]
