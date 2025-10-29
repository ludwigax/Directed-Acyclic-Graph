# DAG: Declarative Graph Runtime

`dag` is a compact toolkit for building and executing deterministic computation graphs. It combines:

- A runtime that instantiates registered operators, manages caching, and exposes debug stats.
- Immutable `GraphSpec` definitions that can be generated from Python or the DSL and serialised back to plain data.
- A readable DSL for sketching graphs, wiring parameters, and composing reusable templates.
- Introspection tooling that keeps the original hierarchy intact so specs can be inspected without flattening.

Use it for research pipelines, feature transforms, retrieval chains, or any workload that benefits from explicit, inspectable DAGs.

---

## Installation

```bash
pip install dag
```

Optional extras: `pip install fastapi uvicorn plotly networkx graphviz` (web visualiser currently being rebuilt after the runtime refactor).

---

## 1. Register Operators

The registry accepts plain functions, classes, or previously defined graphs. Decorators infer ports automatically.

```python
from dag.node import register_function, register_class, returns_keys

@register_function(name="add_pair", outputs={"result": float})
def add_pair(left: float, right: float) -> float:
    return left + right

@register_class(name="ScaleAndBias", forward="compute")
class ScaleAndBias:
    def __init__(self, scale: float = 1.0, bias: float = 0.0) -> None:
        self.scale = scale
        self.bias = bias

    @returns_keys(output=float)
    def compute(self, value: float) -> float:
        return {"output": self.scale * value + self.bias}
```

Once registered, the operators appear under `ops.*` inside the DSL and can be instantiated from Python via `registry_default`.

---

## 2. Author Graphs with the DSL

The DSL allows parameters, nested graph templates, and binding defaults. Parentheses carry constructor/parameter overrides; brackets wire runtime inputs.

```text
GRAPH SCALE_CORE(version="1.0"):
    PARAMETER scale = 2.0
    INPUT value
    OUTPUT scaled = node.output

    node = ScaleAndBias(scale=Param.scale)[value=value]

GRAPH PIPELINE:
    PARAMETER bias = 0.5
    INPUT x, y

    helper = add_pair[left=x, right=y]
    scaled = Ref.SCALE_CORE(scale=3.0)[value=helper.result]
    out = add_pair[left=scaled.scaled, right=: Param.bias]

    OUTPUT total = out.result
```

Compile and register:

```python
from dag.dsl import parse_dsl
from dag.node import register_graph

program = parse_dsl(DSL_TEXT)
scale_spec = program.build("SCALE_CORE")
register_graph("SCALE_CORE", scale_spec)
pipeline_spec = program.build("PIPELINE")
```

The compiler automatically resolves globals/locals when evaluating inline expressions, so values such as `Param.bias` or Python constants work without extra plumbing.

---

## 3. Build and Run Runtimes

```python
from dag.node import build_graph

runtime = build_graph(pipeline_spec, parameters={"bias": 2.0})
print(runtime.run({"x": 2.0, "y": 1.5}))  # {'total': 12.5}

# run selected outputs, disable cache, force individual nodes, or cap concurrency
runtime.run({"x": 1.0, "y": 1.0}, outputs=["total"], use_cache=False, max_workers=2)
```

Runtime highlights:

- `run(outputs=[...], force_nodes={...}, use_cache=False)` for selective execution.
- Parallel scheduling via `max_workers` with per-node caches guarded by thread-safe runners.
- Pre/post hooks (`ExecutionPlan.register_hook`) execute inside worker threads, enabling custom cache tweaks or result normalisation.
- Debug hooks (`dag.dbg.DebuggingContext`) collect per-node timing/metrics without modifying operators.

---

## 4. GraphSpec Serialisation

`GraphSpec` objects can be emitted to plain dicts (and back) without losing parameter refs or nested graphs.

```python
spec_dict = pipeline_spec.to_dict()
restored = GraphSpec.from_dict(spec_dict)
assert restored.to_dict() == spec_dict
```

The encoding preserves `ParameterRefValue` placeholders (`{"__dag_param__": "bias"}`), tuples, and sets so you can store specs in JSON, ship them across processes, or generate them in other languages.

---

## 5. Tooling & Examples

- `docs/dag_dsl_snapshot.md` - deep dive into the architecture with layered examples.
- `examples/getting_started.py` - mirrors the workflow, demonstrates parallel execution, hooks, and renders spec trees.
- `dag.inspect_utils` - text/tree summaries (see below); FastAPI/Plotly visualiser coming back in a future release.

### Inspect specs without flattening

The text inspector preserves the hierarchy and annotates each node with its runtime state when a plan is available:

```python
from dag.inspect_utils import render_spec_tree, render_spec_node

print(render_spec_tree(pipeline_spec, plan=runtime, root_name="PIPELINE"))
print(render_spec_node(pipeline_spec, "scaled.node", plan=runtime, root_name="PIPELINE"))
```

Example output:

```
PIPELINE (graph) [computed]
├── PIPELINE.helper [computed] <helper> :: add_pair
├── PIPELINE.out [computed] <out> :: add_pair
└── PIPELINE.scaled (graph) [computed]
    └── PIPELINE.scaled.node [computed] <scaled__node> :: ScaleAndBias
```

States are inferred from the plan's cache: `computed` (cache populated), `pending` (never evaluated), `partial` (cache disabled/cleared), `unknown` (mixed children). Use `render_spec_node(...)` to drill into configs, metadata, execution stats, and cache status for any path or runtime id.

---

## License

MIT (c) Ludwig
