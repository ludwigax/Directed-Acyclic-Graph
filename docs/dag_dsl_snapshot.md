# DAG & DSL Snapshot

This note gives a practical, example-driven tour of the `dag` package. Each section builds on the previous one so a new user can move from registering operators, to writing DSL graphs, to executing graph runtimes, and finally to understanding the underlying `GraphSpec` structures produced by the compiler.

## 1. Registering Functions and Classes as Operators

Operators are created by decorating Python callables or classes. Registration inspects signatures to infer input ports and output keys automatically.

```python
from dag.node import register_function, register_class, returns_keys

@register_function(name="add_pair")
def add_pair(left: float, right: float):
    return {"result": left + right}

@register_class(name="ScaleAndBias", forward="compute")
class ScaleAndBias:
    def __init__(self, scale: float = 1.0, bias: float = 0.0):
        self.scale = scale
        self.bias = bias

    @returns_keys(output=float)
    def compute(self, value: float):
        return {"output": self.scale * value + self.bias}
```

- `register_function` creates a template that defaults to the function name (`add_pair`) and exposes all parameters as input ports. Returning a mapping lets you name outputs explicitly.
- `register_class` inspects both `__init__` and the chosen forward method (`compute`). Constructor defaults surface as `config["init"]` overrides, while forward parameters become runtime inputs.

These registered names (e.g. `add_pair`, `ScaleAndBias`) now appear in the DSL’s `ops` namespace and can be referenced without qualifying their module.

## 2. Authoring DSL Graphs and Templates

The DSL combines registered operators, temporary graph templates, and nested graph reuse. Key syntax rules:
- `GRAPH/INPUT/OUTPUT/PARAMETER` are uppercase keywords.
- Parentheses pass constructor/parameter overrides; brackets bind runtime inputs or defaults.
- Omit `ops.` unless you want to force a different namespace: `adder = add_pair[left=a, right=b]`.
- Use `Ref.<GraphName>(...)` to instantiate a previously defined graph within another graph. Bare `Ref.NAME` defaults to the graph’s own parameter defaults.

```dsl
GRAPH SCALE_CORE(version="1.0"):
    PARAMETER scale = 2.0
    INPUT value
    OUTPUT scaled = mul.result

    mul = ScaleAndBias(scale=Param.scale, bias=0.0)[value=value]

GRAPH PIPELINE:
    PARAMETER bias = 0.5
    INPUT x, y
    OUTPUT total = out.result

    # Temporary helper inside this file; not registered globally.
    helper = add_pair[left=x, right=y]

    scaled = Ref.SCALE_CORE(scale=3.0)[value=helper.result]
    out = add_pair[left=scaled.scaled, right=: Param.bias]
```

- `ScaleAndBias(scale=Param.scale)` passes the DSL parameter through the parentheses, so the nested class sees it as an `init` override.
- `Ref.SCALE_CORE(scale=3.0)` instantiates the registered graph template. The brackets `[value=...]` connect runtime inputs.
- Leaving the source blank (`right=: Param.bias`) attaches a default to the port while keeping the parameter overrideable via `GraphBuilder.build(..., parameters={"bias": ...})`.

Once parsed via `parse_dsl(text)`, you may call `.build("PIPELINE")` to obtain a `GraphSpec`, or `register_graph("PIPELINE", spec)` to expose the template to other DSL files.

## 3. Building and Inspecting Runtimes

`GraphSpec` records the declarative structure, while `GraphBuilder` produces an executable `GraphRuntime`.

```python
from dag.dsl import parse_dsl
from dag.node import build_graph

program = parse_dsl(DSL_TEXT)
pipeline_spec = program.build("PIPELINE")

# Materialise with optional parameter overrides.
runtime = build_graph(pipeline_spec, parameters={"bias": 2.0})

# Execute by supplying graph inputs.
result = runtime.run({"x": 2.0, "y": 1.5})
assert result == {"total": 2.0 * 3.0 + 2.0}  # (x + y) scaled + custom bias

# The runtime can be inspected or partially evaluated.
runtime.describe()          # serialisable overview (nodes, edges, ports, parameters)
runtime.run({"x": 1.0}, outputs=["total"])  # request a subset of outputs
runtime.clear_cache()       # reset memoised node outputs if needed
```

- `build_graph` resolves graph parameters, instantiates each node (including nested graphs), wires dependencies, and returns a runtime ready for repeated execution.
- `GraphRuntime.run()` accepts input mappings, optional `outputs`, `force_nodes`, or `use_cache` flags to control evaluation.
- `describe()` is useful for debugging or feeding visualisation utilities; it mirrors the logical structure without exposing internal runner state.

## 4. Understanding `GraphSpec` and Serialisation

Every DSL graph compiles into a `GraphSpec`. The spec is immutable, easy to serialise, and can be round-tripped between Python and plain dicts.

```python
spec_dict = pipeline_spec.to_dict()

# Structure highlights
spec_dict["parameters"]      # {"bias": {"default": 0.5}}
spec_dict["inputs"]          # {"x": "helper.left", "y": "helper.right"}
spec_dict["nodes"]["scaled"] # nested operator resolved to a dict (GraphSpec)

# Reconstructing from a serialised form
from dag.node import GraphSpec

restored_spec = GraphSpec.from_dict(spec_dict)
assert restored_spec.to_dict() == spec_dict
```

- `GraphSpec.nodes` stores `NodeSpec` entries; `operator` may be a registered template name, a nested `GraphSpec`, or a mapping that can rebuild one.
- `ParameterRefValue` placeholders are preserved inside configs. The serialised form uses `{ "__dag_param__": "bias", "default": ... }` so defaults survive across environments.
- Because `GraphSpec` is declarative, you can generate it via DSL, Python APIs, or external tooling, then feed it back into `build_graph` or `register_graph`.

Taken together, these four stages—operator registration, DSL authoring, runtime execution, and spec inspection—cover the essential workflow of the project. Use them as a reference when extending the DSL, debugging graph templates, or integrating the runtime into larger systems.
