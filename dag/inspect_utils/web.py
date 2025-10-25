"""FastAPI/Plotly visualisation for declarative graph runtimes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections import defaultdict, deque
import shutil
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from ..node import GraphRuntime, GraphSpec, NodeOutputRef, OperatorTemplate

# Optional dependency placeholders; populated lazily.
FastAPI = None  # type: ignore
HTTPException = None  # type: ignore
HTMLResponse = None  # type: ignore
JSONResponse = None  # type: ignore
go = None  # type: ignore
PlotlyJSONEncoder = None  # type: ignore
nx = None  # type: ignore
uvicorn = None  # type: ignore
graphviz = None  # type: ignore
_EXECUTABLE_NOT_FOUND = ()  # type: ignore


NODE_CATEGORIES = {
    "graph_input": {"label": "Graph Input", "color": "#1f77b4"},
    "graph_output": {"label": "Graph Output", "color": "#ff7f0e"},
    "nested": {"label": "Nested Graph", "color": "#9467bd"},
    "operator": {"label": "Operator", "color": "#2ca02c"},
}

EDGE_CATEGORIES = {
    "input": {"label": "Input Bindings", "color": "#1f77b4"},
    "internal": {"label": "Internal Flow", "color": "#6c757d"},
    "output": {"label": "Graph Outputs", "color": "#ff7f0e"},
}


def _ensure_optional_deps() -> None:
    """Import optional dependencies on demand."""
    global FastAPI, HTTPException, HTMLResponse, JSONResponse
    global go, PlotlyJSONEncoder, nx, uvicorn, graphviz

    if FastAPI is not None:
        return

    missing: List[str] = []
    try:
        from fastapi import FastAPI as _FastAPI, HTTPException as _HTTPException
        from fastapi.responses import HTMLResponse as _HTMLResponse, JSONResponse as _JSONResponse
    except ImportError:
        missing.append("fastapi")
    else:
        FastAPI = _FastAPI  # type: ignore[assignment]
        HTTPException = _HTTPException  # type: ignore[assignment]
        HTMLResponse = _HTMLResponse  # type: ignore[assignment]
        JSONResponse = _JSONResponse  # type: ignore[assignment]

    try:
        import networkx as _nx
    except ImportError:
        missing.append("networkx")
    else:
        nx = _nx  # type: ignore[assignment]

    try:
        import plotly.graph_objects as _go
        import plotly.utils as _plotly_utils
    except ImportError:
        missing.append("plotly")
    else:
        go = _go  # type: ignore[assignment]
        PlotlyJSONEncoder = _plotly_utils.PlotlyJSONEncoder  # type: ignore[assignment]

    try:
        import uvicorn as _uvicorn
    except ImportError:
        missing.append("uvicorn")
    else:
        uvicorn = _uvicorn  # type: ignore[assignment]

    try:
        import graphviz as _graphviz
    except ImportError:
        graphviz = None  # type: ignore[assignment]
    else:
        graphviz = _graphviz  # type: ignore[assignment]
        try:
            from graphviz.backend import ExecutableNotFound as _ExecutableNotFound
        except ImportError:  # pragma: no cover - version dependent
            pass
        else:
            global _EXECUTABLE_NOT_FOUND  # noqa: PLW0603
            _EXECUTABLE_NOT_FOUND = (_ExecutableNotFound,)  # type: ignore[assignment]

    if missing and FastAPI is None:
        raise ImportError(
            "Graph visualisation requires optional packages: "
            + ", ".join(sorted(missing))
            + ". Install with `pip install dag_pkg[visualizer]` (adjust name as needed)."
        )


def _stringify_mapping(mapping: Mapping[str, Any]) -> Dict[str, str]:
    return {str(k): repr(v) for k, v in mapping.items()}


def _operator_descriptor(operator: Any, template: OperatorTemplate) -> Tuple[str, str]:
    if isinstance(operator, GraphSpec):
        return "inline_graph", template.name
    if isinstance(operator, str):
        return "registered", operator
    if isinstance(operator, OperatorTemplate):
        return "template", operator.name
    return type(operator).__name__, repr(operator)


def _node_category(op_type: str, has_nested: bool) -> str:
    if has_nested:
        return "nested"
    return "operator"


def _make_node_id(kind: str, name: str) -> str:
    return f"{kind}::{name}"


def _compute_layers(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> Dict[str, int]:
    node_ids = [node["id"] for node in nodes]
    category = {node["id"]: node["category"] for node in nodes}

    predecessors: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
    successors: Dict[str, List[str]] = {node_id: [] for node_id in node_ids}
    indegree: Dict[str, int] = {node_id: 0 for node_id in node_ids}

    for edge in edges:
        src = edge["source"]
        dst = edge["target"]
        if dst not in predecessors:
            continue
        predecessors[dst].append(src)
        successors.setdefault(src, []).append(dst)
        indegree[dst] += 1
        successors.setdefault(dst, [])

    depth: Dict[str, int] = {}
    queue: deque[str] = deque()

    for node_id in node_ids:
        if indegree[node_id] == 0:
            depth[node_id] = 0 if category[node_id] == "graph_input" else 0
            queue.append(node_id)

    while queue:
        node_id = queue.popleft()
        base_depth = depth.get(node_id, 0)
        for successor in successors.get(node_id, []):
            candidate = base_depth if category.get(successor) == "graph_input" else base_depth + 1
            existing = depth.get(successor)
            if existing is None or candidate > existing:
                depth[successor] = candidate
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)

    if depth:
        max_internal = max(
            (value for node, value in depth.items() if category.get(node) != "graph_output"),
            default=0,
        )
    else:
        max_internal = 0

    for node_id in node_ids:
        if node_id not in depth:
            depth[node_id] = 0
        if category.get(node_id) == "graph_output":
            depth[node_id] = max_internal + 1

    return depth


def _build_graph_elements(
    runtime: GraphRuntime,
    child_index: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # Inputs as synthetic nodes
    for alias in sorted(runtime.graph_inputs.keys()):
        node_id = _make_node_id("input", alias)
        nodes.append(
            {
                "id": node_id,
                "raw_id": alias,
                "label": alias,
                "category": "graph_input",
                "operator": "graph_input",
                "metadata": {},
                "config": {},
                "child_graph_id": None,
            }
        )

    # Outputs as synthetic nodes
    for alias in sorted(runtime.graph_outputs.keys()):
        node_id = _make_node_id("output", alias)
        nodes.append(
            {
                "id": node_id,
                "raw_id": alias,
                "label": alias,
                "category": "graph_output",
                "operator": "graph_output",
                "metadata": {},
                "config": {},
                "child_graph_id": None,
            }
        )

    # Actual runtime nodes
    for node_id in runtime.topological_order:
        spec_node = runtime.spec.nodes[node_id]
        runtime_node = runtime.node_runtimes[node_id]
        op_type, op_name = _operator_descriptor(spec_node.operator, runtime_node.template)
        child_graph_id = child_index.get(node_id)
        category = _node_category(op_type, child_graph_id is not None)
        nodes.append(
            {
                "id": _make_node_id("node", node_id),
                "raw_id": node_id,
                "label": node_id,
                "category": category,
                "operator": f"{op_type}:{op_name}",
                "metadata": _stringify_mapping(spec_node.metadata),
                "config": _stringify_mapping(spec_node.config),
                "child_graph_id": child_graph_id,
            }
        )

    # Input edges
    for alias, endpoints in runtime.graph_inputs.items():
        source_id = _make_node_id("input", alias)
        for node_id, port in endpoints:
            target_id = _make_node_id("node", node_id)
            edges.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "label": f"{alias} → {port}",
                    "kind": "input",
                }
            )

    # Internal edges
    inbound = runtime._inbound  # pylint: disable=protected-access
    for dst_node, port_map in inbound.items():
        for dst_port, ref in port_map.items():
            if isinstance(ref, NodeOutputRef):
                source_id = _make_node_id("node", ref.node_id)
                target_id = _make_node_id("node", dst_node)
                label = f"{ref.port} → {dst_port}"
                edges.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "label": label,
                        "kind": "internal",
                    }
                )

    # Output edges
    for alias, ref in runtime.graph_outputs.items():
        source_id = _make_node_id("node", ref.node_id)
        target_id = _make_node_id("output", alias)
        edges.append(
            {
                "source": source_id,
                "target": target_id,
                "label": f"{ref.port} → {alias}",
                "kind": "output",
            }
        )

    return nodes, edges


def _compute_positions(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> Dict[str, Tuple[float, float]]:
    layers = _compute_layers(nodes, edges)
    layer_groups: Dict[int, List[str]] = defaultdict(list)
    for node_id, depth in layers.items():
        layer_groups[depth].append(node_id)

    # Try Graphviz first for stable layouts; fall back to NetworkX spring layout.
    if graphviz is not None and shutil.which("dot"):
        max_layer_size = max((len(group) for group in layer_groups.values()), default=1)
        ranksep = max(1.0, min(2.5, 1.0 + len(layer_groups) * 0.2))
        nodesep = max(0.6, min(1.5, 0.6 + max_layer_size * 0.15))

        graph = graphviz.Digraph(
            engine="dot",
            graph_attr={
                "rankdir": "LR",
                "nodesep": str(nodesep),
                "ranksep": str(ranksep),
                "splines": "ortho",
                "overlap": "false",
                "concentrate": "false",
                "pack": "true",
                "packmode": "clust",
                "sep": "+20",
                "esep": "+10",
                "mclimit": "2.0",
                "nslimit": "2.0",
                "remincross": "true",
            },
        )

        for depth in sorted(layer_groups.keys()):
            with graph.subgraph() as sub:
                sub.attr(rank="same")
                for node_id in sorted(layer_groups[depth]):
                    sub.node(node_id)

        for edge in edges:
            graph.edge(edge["source"], edge["target"])

        try:
            plain_lines = graph.pipe(format="plain").decode("utf-8").splitlines()
            positions: Dict[str, Tuple[float, float]] = {}
            width = height = 1.0
            for line in plain_lines:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "graph" and len(parts) >= 4:
                    width = float(parts[2])
                    height = float(parts[3])
                elif parts[0] == "node" and len(parts) >= 4:
                    node_id = parts[1]
                    x = float(parts[2]) / width * 6.0 if width else float(parts[2])
                    y = float(parts[3]) / height * 6.0 if height else float(parts[3])
                    positions[node_id] = (x - 3.0, y - 3.0)
            if len(positions) == len(nodes):
                return positions
        except Exception:  # pragma: no cover - fallback safety
            pass

    assert nx is not None  # populated by _ensure_optional_deps
    graph_nx = nx.DiGraph()
    for node in nodes:
        graph_nx.add_node(node["id"])
    for edge in edges:
        graph_nx.add_edge(edge["source"], edge["target"])

    if graph_nx.number_of_nodes() == 1:
        node_id = next(iter(graph_nx.nodes))
        return {node_id: (0.0, 0.0)}

    max_layer = max(layers.values()) if layers else 0
    layer_positions: Dict[int, Tuple[float, float]] = {}
    for depth, group in layer_groups.items():
        num = len(group)
        if num == 0:
            continue
        x = -3.0 + (6.0 / max(1, max_layer)) * depth if max_layer else 0.0
        start_y = -((num - 1) * 0.8) / 2.0
        for idx, node_id in enumerate(sorted(group)):
            layer_positions[node_id] = (x, start_y + idx * 0.8)
    if len(layer_positions) == len(nodes):
        return layer_positions

    pos = nx.spring_layout(
        graph_nx,
        k=1.2,
        iterations=200,
        seed=42,
    )
    return {node_id: (float(x), float(y)) for node_id, (x, y) in pos.items()}


def _build_plotly_figure(
    graph_label: str,
    graph_depth: int,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    positions: Mapping[str, Tuple[float, float]],
) -> Dict[str, Any]:
    assert go is not None and PlotlyJSONEncoder is not None

    annotations = []
    traces = []

    # Edge traces grouped by kind
    for kind, spec in EDGE_CATEGORIES.items():
        group = [edge for edge in edges if edge["kind"] == kind]
        if not group:
            continue
        edge_x: List[float] = []
        edge_y: List[float] = []
        for edge in group:
            x0, y0 = positions[edge["source"]]
            x1, y1 = positions[edge["target"]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            if edge["label"]:
                annotations.append(
                    dict(
                        x=(x0 + x1) / 2.0,
                        y=(y0 + y1) / 2.0,
                        text=edge["label"],
                        showarrow=False,
                        font=dict(size=10, color="#555555"),
                    )
                )
        traces.append(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1.8, color=spec["color"]),
                hoverinfo="none",
                showlegend=True,
                name=spec["label"],
            )
        )

    # Node traces grouped by category for legend
    for category, spec in NODE_CATEGORIES.items():
        group = [node for node in nodes if node["category"] == category]
        if not group:
            continue
        traces.append(
            go.Scatter(
                x=[positions[node["id"]][0] for node in group],
                y=[positions[node["id"]][1] for node in group],
                mode="markers+text",
                marker=dict(
                    size=22 if category in {"graph_input", "graph_output"} else 18,
                    color=spec["color"],
                    line=dict(width=1.5, color="#2f2f2f"),
                ),
                text=[node["label"] for node in group],
                textposition="top center",
                hoverinfo="text",
                hovertext=[
                    f"{node['label']}<br>{node['operator']}"
                    for node in group
                ],
                name=spec["label"],
            )
        )

    layout = go.Layout(
        title=f"{graph_label} (depth {graph_depth})",
        showlegend=True,
        hovermode="closest",
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=annotations,
        height=720,
        width=1100,
    )

    figure = go.Figure(data=traces, layout=layout)
    return json.loads(figure.to_json())


@dataclass
class _GraphEntry:
    graph_id: str
    runtime: GraphRuntime
    label: str
    depth: int
    path: Tuple[str, ...]
    parent: Optional[str]
    children: Dict[str, str]


class _RuntimeHierarchy:
    def __init__(self, runtime: GraphRuntime):
        self._order: List[str] = []
        self._entries: Dict[str, _GraphEntry] = {}
        self._build(runtime, path=(), parent=None)

    def _build(self, runtime: GraphRuntime, path: Tuple[str, ...], parent: Optional[str]) -> str:
        graph_id = f"g{len(self._order)}"
        label = runtime.spec.metadata.get("name") or ("root" if not path else path[-1])
        entry = _GraphEntry(
            graph_id=graph_id,
            runtime=runtime,
            label=label,
            depth=len(path),
            path=path,
            parent=parent,
            children={},
        )
        self._entries[graph_id] = entry
        self._order.append(graph_id)

        for node_id in runtime.topological_order:
            runtime_node = runtime.node_runtimes[node_id]
            child_runtime = getattr(runtime_node.runner, "_runtime", None)
            spec_node = runtime.spec.nodes[node_id]
            if child_runtime and isinstance(spec_node.operator, GraphSpec):
                child_id = self._build(child_runtime, path + (node_id,), parent=graph_id)
                entry.children[node_id] = child_id

        return graph_id

    def list_graphs(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": entry.graph_id,
                "label": entry.label,
                "depth": entry.depth,
                "path": list(entry.path),
                "parent": entry.parent,
                "children": entry.children,
            }
            for entry in (self._entries[gid] for gid in self._order)
        ]

    def get_entry(self, graph_id: str) -> _GraphEntry:
        try:
            return self._entries[graph_id]
        except KeyError as exc:
            raise KeyError(f"Unknown graph id '{graph_id}'") from exc


def create_runtime_app(runtime: GraphRuntime) -> Any:
    """Create a FastAPI app that serves an interactive Plotly visualisation."""
    _ensure_optional_deps()
    hierarchy = _RuntimeHierarchy(runtime)
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)  # type: ignore[misc]
    def index() -> str:
        return _INDEX_HTML

    @app.get("/graphs", response_class=JSONResponse)  # type: ignore[misc]
    def list_graphs() -> List[Dict[str, Any]]:
        return hierarchy.list_graphs()

    @app.get("/graph/{graph_id}", response_class=JSONResponse)  # type: ignore[misc]
    def get_graph(graph_id: str) -> Dict[str, Any]:
        try:
            entry = hierarchy.get_entry(graph_id)
        except KeyError as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=404, detail=str(exc)) from exc  # type: ignore[arg-type]

        nodes, edges = _build_graph_elements(entry.runtime, entry.children)
        positions = _compute_positions(nodes, edges)
        figure = _build_plotly_figure(entry.label, entry.depth, nodes, edges, positions)

        legend = {
            "nodes": [
                {"category": key, **NODE_CATEGORIES[key]}
                for key in NODE_CATEGORIES
            ],
            "edges": [
                {"kind": key, **EDGE_CATEGORIES[key]}
                for key in EDGE_CATEGORIES
            ],
        }

        return {
            "graph": {
                "id": entry.graph_id,
                "label": entry.label,
                "depth": entry.depth,
                "path": list(entry.path),
                "parent": entry.parent,
            },
            "figure": figure,
            "nodes": nodes,
            "edges": edges,
            "legend": legend,
        }

    return app


def launch_runtime_visualizer(
    runtime: GraphRuntime,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "info",
) -> None:
    """Launch the FastAPI visualiser using uvicorn.run."""
    _ensure_optional_deps()
    app = create_runtime_app(runtime)
    uvicorn.run(app, host=host, port=port, log_level=log_level)  # type: ignore[arg-type]


def visualize_runtime(
    runtime: GraphRuntime,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "info",
) -> None:
    """Alias for launch_runtime_visualizer."""
    launch_runtime_visualizer(runtime, host=host, port=port, log_level=log_level)


_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>DAG Runtime Visualiser</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background-color: #f6f8fb; }
    header { background-color: #1f2933; color: white; padding: 1.2rem 2rem; }
    main { padding: 1.5rem 2rem; }
    h1 { margin: 0; font-size: 1.6rem; }
    .controls { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
    select { padding: 0.4rem 0.6rem; font-size: 1rem; }
    #plot { width: 100%; height: 72vh; background-color: white; border: 1px solid #d1d9e6; border-radius: 0.5rem; }
    .info-panel { margin-top: 1rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.2rem; }
    .card { background: white; border: 1px solid #d1d9e6; border-radius: 0.6rem; padding: 1rem; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08); }
    .card h2 { margin-top: 0; font-size: 1.1rem; color: #1f2933; }
    .node-item { margin-bottom: 0.4rem; }
    .node-item strong { color: #111a2b; }
    .badge { display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.15rem 0.45rem; border-radius: 0.5rem; font-size: 0.8rem; color: white; }
    .legend-list { list-style: none; padding-left: 0; margin: 0; }
    .legend-list li { margin-bottom: 0.35rem; display: flex; align-items: center; gap: 0.5rem; }
    .legend-color { width: 0.9rem; height: 0.9rem; border-radius: 50%; border: 1px solid rgba(17, 26, 43, 0.3); }
    button.link { background: none; border: none; color: #2563eb; cursor: pointer; padding: 0; font-size: 0.9rem; }
    button.link:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <header>
    <h1>DAG Runtime Visualiser</h1>
  </header>
  <main>
    <div class="controls">
      <label for="graphSelect"><strong>Select graph layer:</strong></label>
      <select id="graphSelect"></select>
      <div id="graphMeta"></div>
    </div>
    <div id="plot"></div>
    <div class="info-panel">
      <section class="card">
        <h2>Nodes</h2>
        <div id="nodeList"></div>
      </section>
      <section class="card">
        <h2>Legend</h2>
        <div>
          <h3>Nodes</h3>
          <ul class="legend-list" id="nodeLegend"></ul>
        </div>
        <div style="margin-top: 0.6rem;">
          <h3>Edges</h3>
          <ul class="legend-list" id="edgeLegend"></ul>
        </div>
      </section>
    </div>
  </main>
  <script>
    const graphSelect = document.getElementById('graphSelect');
    const graphMeta = document.getElementById('graphMeta');
    const nodeList = document.getElementById('nodeList');
    const nodeLegend = document.getElementById('nodeLegend');
    const edgeLegend = document.getElementById('edgeLegend');

    async function loadGraphs() {
      const response = await fetch('/graphs');
      if (!response.ok) {
        graphMeta.textContent = 'Failed to load available graphs.';
        return;
      }
      const graphs = await response.json();
      graphSelect.innerHTML = '';
      graphs.forEach((graph) => {
        const option = document.createElement('option');
        option.value = graph.id;
        const pathLabel = graph.path.length ? graph.path.join(' / ') : 'root';
        option.textContent = `[depth ${graph.depth}] ${graph.label} (${pathLabel})`;
        graphSelect.appendChild(option);
      });
      if (graphs.length) {
        await loadGraph(graphs[0].id);
      }
    }

    async function loadGraph(graphId) {
      const response = await fetch(`/graph/${graphId}`);
      if (!response.ok) {
        graphMeta.textContent = 'Failed to load graph.';
        return;
      }
      const data = await response.json();
      Plotly.newPlot('plot', data.figure.data, data.figure.layout, {responsive: true});

      const pathLabel = data.graph.path.length ? data.graph.path.join(' / ') : 'root';
      graphMeta.textContent = `Name: ${data.graph.label} | Depth: ${data.graph.depth} | Path: ${pathLabel}`;

      nodeList.innerHTML = '';
      data.nodes
        .filter((node) => node.category === 'operator' || node.category === 'nested')
        .forEach((node) => {
          const div = document.createElement('div');
          div.className = 'node-item';
          const metaItems = Object.entries(node.metadata)
            .map(([key, value]) => `${key}: ${value}`).join('; ');
          const configItems = Object.entries(node.config)
            .map(([key, value]) => `${key}: ${value}`).join('; ');
          div.innerHTML = `<strong>${node.label}</strong> <span class="badge" style="background:${categoryColor(node.category)}">${node.category}</span><br/>
                           <em>${node.operator}</em><br/>
                           <small>metadata: ${metaItems || '∅'}</small><br/>
                           <small>config: ${configItems || '∅'}</small>`;
          if (node.child_graph_id) {
            const button = document.createElement('button');
            button.className = 'link';
            button.textContent = 'Open nested graph';
            button.addEventListener('click', async () => {
              graphSelect.value = node.child_graph_id;
              await loadGraph(node.child_graph_id);
            });
            div.appendChild(document.createElement('br'));
            div.appendChild(button);
          }
          nodeList.appendChild(div);
        });

      renderLegend(nodeLegend, data.legend.nodes, 'category');
      renderLegend(edgeLegend, data.legend.edges, 'kind');
    }

    function renderLegend(container, items, key) {
      container.innerHTML = '';
      items.forEach((item) => {
        const li = document.createElement('li');
        const color = document.createElement('span');
        color.className = 'legend-color';
        color.style.background = item.color;
        li.appendChild(color);
        const text = document.createTextNode(item.label);
        li.appendChild(text);
        container.appendChild(li);
      });
    }

    function categoryColor(category) {
      switch (category) {
        case 'graph_input': return '#1f77b4';
        case 'graph_output': return '#ff7f0e';
        case 'nested': return '#9467bd';
        default: return '#2ca02c';
      }
    }

    graphSelect.addEventListener('change', async (event) => {
      await loadGraph(event.target.value);
    });

    loadGraphs();
  </script>
</body>
</html>
"""
