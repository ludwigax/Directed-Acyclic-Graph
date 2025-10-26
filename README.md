# DAG â€?Declarative Graph Runtime

`dag` is a light-weight toolkit for describing, executing, and inspecting directed acyclic computation graphs in Python. It offers:

- A **GraphSpec** API to declare nodes, inputs/outputs, and edges in pure data structures.
- A **GraphRuntime** that instantiates registered operators, handles caching, and records execution stats.
- **Inspect utilities** for text-based summaries and a FastAPI/Plotly web explorer.
- A compact **DSL** that compiles into GraphSpecs for rapid prototyping or review.

The project is ideal for research pipelines, RAG workflows, feature-engineering DAGs, or anywhere you need deterministic, inspectable execution graphs without pulling in a heavyweight orchestrator.

---

## Installation

```bash
pip install dag
```

Optional extras:

- `pip install "fastapi uvicorn plotly networkx graphviz"` to enable the interactive visualizer.
- Example scripts (under `examples/`) assume Python 3.10+.

---

## Registering Operators

Graph nodes reference operators registered on the global registry. Operators can be plain functions, classes (with `forward`), or even previously-defined graphs.

```python
from dag.node import register_function, returns_keys

@register_function(name="normalize")
@returns_keys(result=float)
def normalize(value: float, mean: float = 0.0, std: float = 1.0) -> float:
    return (value - mean) / std
```

`returns_keys` records the output mapping so downstream nodes can refer to `normalize.result`.

---

## Creating Graph Specs

Graph specs are pure data. Each node points to a registered operator (by name) or embeds another spec. Inputs and outputs use `<node>.<port>` notation.

```python
from dag.node import GraphSpec

normalize_workflow = GraphSpec.from_dict(
    {
        "nodes": {
            "input": {"operator": "constant", "config": {"value": 42}},
            "stats": {"operator": "constant", "config": {"value": {"mean": 40, "std": 4}}},
            "norm": {"operator": "normalize"},
        },
        "edges": [
            {"src": "input._return", "dst": "norm.value"},
            {"src": "stats._return", "dst": "norm.mean"},
            {"src": "stats._return", "dst": "norm.std"},
        ],
        "inputs": {},
        "outputs": {"result": "norm.result"},
        "metadata": {"name": "normalize_workflow"},
    }
)
```

`GraphSpec.from_dict` performs validation (duplicate nodes, missing ports, cycles, etc.) so specs fail fast before execution.

---

## Parameterised Graphs

Expose knobs for callers by declaring parameters. Operators read the parameter values via `ParameterRefValue`, and `build_graph` accepts overrides when instantiating the runtime.

```python
from dag.node import GraphSpec, ParameterRefValue, build_graph

scaler_spec = GraphSpec.from_dict(
    {
        "parameters": {"scale": {"default": 2.0}},
        "nodes": {
            "mul": {
                "operator": "multiplication",
                "config": {"call": {"b": ParameterRefValue("scale")}},
            }
        },
        "edges": [],
        "inputs": {"value": "mul.a"},
        "outputs": {"result": "mul.result"},
    }
)

runtime = build_graph(scaler_spec, parameters={"scale": 5})
print(runtime.run(inputs={"value": 4})["result"])  # 20
```

In the DSL the `parameter` keyword creates placeholders and `Param.<name>` injects them into operator configs (with optional defaults):

```text
GRAPH scaler:\r\n    PARAMETER scale=3\r\n    INPUT value

    mul = ops.multiplication()[a=value, b=: Param.scale]

    OUTPUT result = mul.result
```

Nested graphs receive overrides via `Ref.scaler(scale=7)` or `Ref.scaler(parameters={"scale": 7})`.

---

## Building & Running Graphs

```python
from dag.node import build_graph
from dag.dbg import DebuggingContext

runtime = build_graph(normalize_workflow)

with DebuggingContext(True):  # tracks execution time per node
    outputs = runtime.run(inputs={})

print(outputs["result"])  # 0.5
```

### Execution Controls

- `GraphRuntime.run(outputs=[...])` restricts which graph outputs are materialized.
- `force_nodes={"norm"}` recomputes specific nodes even when cache inputs match.
- `runtime.clear_cache()` clears all node caches; pass a node id to target one.

### Debug stats

Every operator inherits from `Debug`. When debug mode is on, `runtime.node_runtimes["norm"].runner.get_stats()` reports total/average execution time and call counts.

---

## Inspecting Graphs

Text mode:

```python
from dag.inspect_utils import render_runtime_text

print(render_runtime_text(runtime))
```

This prints nodes, inputs, outputs, and nested graphs in a readable treeâ€”handy for code reviews.

Web visualizer (requires optional deps):

```python
from dag.inspect_utils import visualize_runtime

visualize_runtime(runtime, port=5001, open_browser=True)
```

The FastAPI/Plotly app shows DAG layers, allows drilling into nested graphs, and exposes node metadata/config. Use it when debugging complex pipelines or showcasing architecture.

---

## Declarative DSL

For quick authoring or reviews, the DSL turns indented text into GraphSpecs. Features include:

- `input`/`output` declarations.
- Node assignments with concise operator syntax (`add = ops.addition()[a=x, b=y]`).
- Default values via `port=:expression`.
- `Ref.other_graph()` to reuse previously-defined graphs.

```python
from dag.dsl import parse_dsl
from dag.node import build_graph

dsl_text = """
GRAPH scoring:
    INPUT x, y

    add = ops.addition()[a=x, b=y]
    scale = ops.multiplication()[a=add.result, b=:0.5]

    OUTPUT score = scale.result
"""

program = parse_dsl(dsl_text, globals=globals(), locals=locals())
spec = program.build("scoring")
runtime = build_graph(spec)
print(runtime.run(inputs={"x": 2, "y": 4}))
```

You can attach parameters in DSL just as in Python:

```text
GRAPH tuned_scoring:\r\n    PARAMETER weight\r\n    INPUT x, y

    base = Ref.scoring()[x=x, y=y]
    adjust = ops.multiplication()[a=y, b=: Param.weight]

    OUTPUT score = ops.addition()[a=base.score, b=adjust.result]
```

The DSL shares the same registry as the Python API, so any decorated function/class is instantly available in `ops.*`. The generated `GraphSpec` can also be serialized, visualized, or embedded in other specs.

---

## Examples & Next Steps

- `examples/declarative_math_example.py` â€?fully programmatic specs.
- `examples/dsl_equation_example.py` â€?nested graphs via DSL.
- `examples/inspect_*` â€?console and web inspection utilities.
- `examples/parameter_override_demo.py` â€?parameter overrides for registered graphs and DSL `Ref` usage.

Ideas for extending your graph stack:

1. **Package reusable graphs** with `register_graph("name", spec)` and treat them like first-class operators.
2. **Add observability** by inserting logging or metric operators; caching + debug stats make it easy to profile nodes.
3. **Integrate with RAG / feature stores** by wrapping domain-specific logic in functions, registering them, and using DSL files as architecture blueprints.

---

## License

MIT Â© Ludwig.



