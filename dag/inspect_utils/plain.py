"""Human-friendly console inspection for DAG plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..node import ExecutionPlan
from ..core.nodes import GraphTemplate, NodeShell
from ..core.specs import GraphSpec, NodeSpec

TreePath = Tuple[str, ...]
PlanSource = Union[ExecutionPlan, GraphTemplate, GraphSpec]

ASCII_BRANCH_LAST = "+-- "
ASCII_BRANCH_MID = "|-- "
ASCII_PIPE_LAST = "    "
ASCII_PIPE_MID = "|   "
RUNTIME_SEP = "/"


@dataclass
class TreeNode:
    name: str
    path: TreePath
    kind: str  # "graph" | "node"
    display_path: str
    runtime_id: Optional[str]
    node_spec: Optional[NodeSpec] = None
    node_shell: Optional[NodeShell] = None
    graph_spec: Optional[GraphSpec] = None
    operator_ref: Any = None
    config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    graph_metadata: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    inputs: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    operator_name: Optional[str] = None
    state: str = "pending"
    children: List["TreeNode"] = field(default_factory=list)


class SpecInspector:
    """Compact inspector for plans/specs with optional runtime state."""

    def __init__(
        self,
        source: PlanSource,
        *,
        plan: Optional[ExecutionPlan] = None,
        root_name: Optional[str] = None,
    ) -> None:
        if isinstance(source, ExecutionPlan):
            self.plan = source
            self.template = source.template
            self.spec = None
            metadata = dict(self.template.metadata)
        elif isinstance(source, GraphTemplate):
            self.plan = plan
            self.template = source
            self.spec = None
            metadata = dict(self.template.metadata)
        else:
            self.plan = plan
            self.template = None
            self.spec = source
            metadata = dict(source.metadata)

        self.root_name = root_name or metadata.get("name") or "Graph"
        if self.template is not None:
            self.root = self._build_template_tree(self.template)
        else:
            assert self.spec is not None
            self.root = self._build_graph_tree(self.spec, prefix=())

        self._display_index: Dict[str, TreeNode] = {}
        self._relative_index: Dict[str, TreeNode] = {}
        self._runtime_index: Dict[str, TreeNode] = {}
        self._register(self.root)
        self._annotate_states(self.root)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def render_tree(self, *, show_runtime_ids: bool = True) -> str:
        lines: List[str] = [self._format_node_label(self.root, show_runtime_ids)]
        for idx, child in enumerate(self.root.children):
            self._render_subtree(
                node=child,
                lines=lines,
                prefix="",
                is_last=idx == len(self.root.children) - 1,
                show_runtime_ids=show_runtime_ids,
            )
        return "\n".join(lines)

    def render_node(self, path: str) -> str:
        node = self._resolve_path(path)
        return "\n".join(self._format_node_details(node))

    def resolve_paths(self) -> Sequence[str]:
        return sorted(key for key in self._relative_index if key)

    # ------------------------------------------------------------------ #
    # Tree building helpers
    # ------------------------------------------------------------------ #

    def _build_graph_tree(
        self,
        spec: GraphSpec,
        *,
        prefix: TreePath,
        node_id: Optional[str] = None,
        node_spec: Optional[NodeSpec] = None,
    ) -> TreeNode:
        path = prefix + ((node_id,) if node_id else ())
        display_path = self._format_display_path(path)
        node = TreeNode(
            name=node_id or self.root_name,
            path=path,
            kind="graph",
            display_path=display_path,
            runtime_id=None,
            node_spec=node_spec,
            graph_spec=spec,
            operator_ref=spec,
            config=dict(node_spec.config) if node_spec else {},
            metadata=dict(node_spec.metadata) if node_spec else {},
            graph_metadata=dict(spec.metadata),
            parameters=dict(spec.parameters),
            inputs=dict(spec.inputs),
            outputs=dict(spec.outputs),
            operator_name=self._resolve_graph_name(node_id or self.root_name, spec, node_spec),
        )
        for child_id in sorted(spec.nodes.keys()):
            child_spec = spec.nodes[child_id]
            if isinstance(child_spec.operator, GraphSpec):
                node.children.append(
                    self._build_graph_tree(
                        child_spec.operator,
                        prefix=path,
                        node_id=child_id,
                        node_spec=child_spec,
                    )
                )
            else:
                node.children.append(self._build_leaf_from_spec(child_spec, path + (child_id,)))
        return node

    def _build_leaf_from_spec(self, node_spec: NodeSpec, path: TreePath) -> TreeNode:
        runtime_id = RUNTIME_SEP.join(path)
        return TreeNode(
            name=path[-1],
            path=path,
            kind="node",
            display_path=self._format_display_path(path),
            runtime_id=runtime_id,
            node_spec=node_spec,
            operator_ref=node_spec.operator,
            config=dict(node_spec.config),
            metadata=dict(node_spec.metadata),
            operator_name=self._resolve_operator_name(node_spec.operator),
        )

    def _build_template_tree(self, template: GraphTemplate) -> TreeNode:
        root = TreeNode(
            name=self.root_name,
            path=(),
            kind="graph",
            display_path=self.root_name,
            runtime_id=None,
            graph_metadata=dict(template.metadata),
            parameters=dict(template.parameters),
            inputs={alias: list(endpoints) for alias, endpoints in template.inputs.items()},
            outputs=dict(template.outputs),
        )

        graph_nodes: Dict[str, TreeNode] = {"": root}
        for path_key in sorted(template.shell_index.keys()):
            parts = tuple(filter(None, path_key.split(".")))
            if not parts:
                continue
            runtime_ids = template.shell_index[path_key]
            prefix_id = RUNTIME_SEP.join(parts)
            has_nested = any(rid != prefix_id for rid in runtime_ids)
            if not has_nested:
                continue
            parent_key = ".".join(parts[:-1])
            parent = graph_nodes.get(parent_key, root)
            node_path = parent.path + (parts[-1],)
            graph_node = TreeNode(
                name=parts[-1],
                path=node_path,
                kind="graph",
                display_path=self._format_display_path(node_path),
                runtime_id=None,
            )
            parent.children.append(graph_node)
            graph_nodes[path_key] = graph_node

        for runtime_id, shell in sorted(template.nodes.items()):
            segments = tuple(filter(None, runtime_id.split(RUNTIME_SEP)))
            parent = graph_nodes.get(".".join(segments[:-1]), root)
            node_path = parent.path + (segments[-1],)
            parent.children.append(
                TreeNode(
                    name=segments[-1],
                    path=node_path,
                    kind="node",
                    display_path=self._format_display_path(node_path),
                    runtime_id=runtime_id,
                    node_shell=shell,
                    operator_ref=shell.template,
                    config=dict(shell.config),
                    metadata=dict(shell.metadata),
                    operator_name=self._resolve_operator_name(shell.template),
                )
            )
        return root

    # ------------------------------------------------------------------ #
    # State computation
    # ------------------------------------------------------------------ #

    def _annotate_states(self, node: TreeNode) -> str:
        if node.kind == "node":
            node.state = self._leaf_state(node)
            return node.state
        child_states = [self._annotate_states(child) for child in node.children]
        if not child_states:
            node.state = "pending"
        elif all(state == "computed" for state in child_states):
            node.state = "computed"
        elif all(state == "pending" for state in child_states):
            node.state = "pending"
        elif any(state == "computed" for state in child_states):
            node.state = "partial"
        else:
            node.state = "partial"
        return node.state

    def _leaf_state(self, node: TreeNode) -> str:
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
    # Rendering
    # ------------------------------------------------------------------ #

    def _render_subtree(
        self,
        *,
        node: TreeNode,
        lines: List[str],
        prefix: str,
        is_last: bool,
        show_runtime_ids: bool,
    ) -> None:
        branch = ASCII_BRANCH_LAST if is_last else ASCII_BRANCH_MID
        lines.append(f"{prefix}{branch}{self._format_node_label(node, show_runtime_ids)}")
        child_prefix = f"{prefix}{ASCII_PIPE_LAST if is_last else ASCII_PIPE_MID}"
        for idx, child in enumerate(node.children):
            self._render_subtree(
                node=child,
                lines=lines,
                prefix=child_prefix,
                is_last=idx == len(node.children) - 1,
                show_runtime_ids=show_runtime_ids,
            )

    def _format_node_label(self, node: TreeNode, show_runtime_ids: bool) -> str:
        state = node.state
        if node.kind == "graph":
            label = f"{node.display_path} (graph) [{state}]"
        else:
            label = f"{node.display_path} [{state}]"
            if show_runtime_ids and node.runtime_id:
                label += f" <{node.runtime_id}>"
            if node.operator_name:
                label += f" :: {node.operator_name}"
        return label

    def _format_node_details(self, node: TreeNode) -> List[str]:
        state = node.state
        lines: List[str] = []
        if node.kind == "graph":
            lines.append(f"Graph {node.display_path}")
            lines.append(f"State: {state}")
            if node.node_spec and node.operator_name:
                lines.append(f"Name: {node.operator_name}")
            if node.config:
                lines.append("Node config:")
                lines.extend(self._indent_lines(self._format_mapping(node.config)))
            if node.metadata:
                lines.append("Node metadata:")
                lines.extend(self._indent_lines(self._format_mapping(node.metadata)))
            if node.graph_metadata:
                lines.append("Graph metadata:")
                lines.extend(self._indent_lines(self._format_mapping(node.graph_metadata)))
            if node.parameters:
                lines.append("Parameters:")
                lines.extend(self._indent_lines(self._format_parameters(node.parameters)))
            if node.inputs:
                lines.append("Inputs:")
                lines.extend(self._indent_lines(self._format_multi_endpoint(node.inputs)))
            if node.outputs:
                lines.append("Outputs:")
                lines.extend(self._indent_lines(self._format_mapping(node.outputs)))
            if node.children:
                lines.append("Children:")
                for child in node.children:
                    suffix = f" :: {child.operator_name}" if (child.kind == "node" and child.operator_name) else ""
                    lines.append(f"  - {child.display_path} ({child.kind}) [{child.state}]{suffix}")
        else:
            lines.append(f"Node {node.display_path}")
            lines.append(f"State: {state}")
            if node.runtime_id:
                lines.append(f"Runtime id: {node.runtime_id}")
            if node.operator_name:
                lines.append(f"Operator: {node.operator_name}")
            if node.operator_ref is not None:
                lines.append(f"Operator ref: {self._describe_operator(node.operator_ref)}")
            if node.config:
                lines.append("Config:")
                lines.extend(self._indent_lines(self._format_mapping(node.config)))
            if node.metadata:
                lines.append("Metadata:")
                lines.extend(self._indent_lines(self._format_mapping(node.metadata)))
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
                    lines.append(f"Cache: {cache_label} (enabled: {'yes' if runtime.cache_enabled else 'no'})")
        return lines

    # ------------------------------------------------------------------ #
    # Indexing / resolution
    # ------------------------------------------------------------------ #

    def _register(self, node: TreeNode) -> None:
        self._display_index[node.display_path] = node
        relative = ".".join(node.path)
        self._relative_index[relative] = node
        if node.runtime_id:
            self._runtime_index[node.runtime_id] = node
        for child in node.children:
            self._register(child)

    def _resolve_path(self, path: str) -> TreeNode:
        query = path.strip()
        if not query:
            return self.root
        if query in self._display_index:
            return self._display_index[query]
        if query in self._runtime_index:
            return self._runtime_index[query]
        parts = [part for part in query.split(".") if part]
        if parts and parts[0] == self.root_name:
            parts = parts[1:]
        normalized = ".".join(parts)
        if normalized in self._relative_index:
            return self._relative_index[normalized]
        raise KeyError(f"Unknown node path '{path}'")

    def _format_display_path(self, path: TreePath) -> str:
        if not path:
            return self.root_name
        return f"{self.root_name}." + ".".join(path)

    # ------------------------------------------------------------------ #
    # Formatting helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_mapping(mapping: Mapping[str, Any]) -> List[str]:
        if not mapping:
            return ["<none>"]
        return [f"{key}: {repr(value)}" for key, value in sorted(mapping.items())]

    @staticmethod
    def _format_multi_endpoint(mapping: Mapping[str, Any]) -> List[str]:
        if not mapping:
            return ["<none>"]
        lines: List[str] = []
        for key, value in sorted(mapping.items()):
            if isinstance(value, (list, tuple, set)):
                rendered = ", ".join(str(item) for item in value)
            else:
                rendered = str(value)
            lines.append(f"{key}: {rendered}")
        return lines

    @staticmethod
    def _format_parameters(parameters: Mapping[str, Any]) -> List[str]:
        lines: List[str] = []
        for name in sorted(parameters):
            spec = parameters[name]
            default = getattr(spec, "default", None)
            required = getattr(spec, "required", False)
            if required:
                lines.append(f"{name}: <required>")
            else:
                lines.append(f"{name}: default={repr(default)}")
        return lines or ["<none>"]

    @staticmethod
    def _indent_lines(lines: Iterable[str], indent: str = "  ") -> List[str]:
        return [f"{indent}{line}" for line in lines]

    @staticmethod
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

    @staticmethod
    def _resolve_graph_name(name: str, spec: GraphSpec, node_spec: Optional[NodeSpec]) -> str:
        if node_spec and node_spec.metadata.get("name"):
            return str(node_spec.metadata["name"])
        if spec.metadata.get("name"):
            return str(spec.metadata["name"])
        return name

    @staticmethod
    def _describe_operator(operator_ref: Any) -> str:
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


# --------------------------------------------------------------------------- #
# Convenience entry points
# --------------------------------------------------------------------------- #


def inspect_plan(
    source: PlanSource,
    *,
    plan: Optional[ExecutionPlan] = None,
    root_name: Optional[str] = None,
) -> SpecInspector:
    """Build an inspector for a plan/spec/template."""
    if isinstance(source, ExecutionPlan):
        return SpecInspector(source, root_name=root_name)
    return SpecInspector(source, plan=plan, root_name=root_name)


def render_plan_tree(
    source: PlanSource,
    *,
    plan: Optional[ExecutionPlan] = None,
    root_name: Optional[str] = None,
    show_runtime_ids: bool = True,
) -> str:
    return inspect_plan(source, plan=plan, root_name=root_name).render_tree(
        show_runtime_ids=show_runtime_ids
    )


def print_plan_tree(
    source: PlanSource,
    *,
    plan: Optional[ExecutionPlan] = None,
    root_name: Optional[str] = None,
    show_runtime_ids: bool = True,
) -> None:
    print(
        render_plan_tree(
            source,
            plan=plan,
            root_name=root_name,
            show_runtime_ids=show_runtime_ids,
        )
    )


def render_plan_node(
    source: PlanSource,
    path: str,
    *,
    plan: Optional[ExecutionPlan] = None,
    root_name: Optional[str] = None,
) -> str:
    return inspect_plan(source, plan=plan, root_name=root_name).render_node(path)


def print_plan_node(
    source: PlanSource,
    path: str,
    *,
    plan: Optional[ExecutionPlan] = None,
    root_name: Optional[str] = None,
) -> None:
    print(render_plan_node(source, path, plan=plan, root_name=root_name))


__all__ = [
    "SpecInspector",
    "inspect_plan",
    "render_plan_tree",
    "print_plan_tree",
    "render_plan_node",
    "print_plan_node",
]
