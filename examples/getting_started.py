"""End-to-end walkthrough for the DAG package.

This script mirrors the documentation in ``docs/dag_dsl_snapshot.md``:

1. Register plain Python callables and classes as operators.
2. Author graphs with the DSL, including graph templates and registered operators.
3. Materialise a runtime, execute it, and inspect the results.
4. Serialise the resulting ``GraphSpec`` to a plain dict for storage or transport.
"""

from __future__ import annotations

from pprint import pprint

from dag.dsl import parse_dsl
from dag.node import (
    GraphSpec,
    build_graph,
    register_class,
    register_function,
    register_graph,
    returns_keys,
)


# ---------------------------------------------------------------------------
# 1. Register functions and classes as operators
# ---------------------------------------------------------------------------


@register_function(name="add_pair", outputs={"result": float})
def add_pair(left: float, right: float) -> float:
    """Return the sum of two floats."""
    return left + right


@register_class(name="ScaleAndBias", forward="compute")
class ScaleAndBias:
    """Simple affine transform node."""

    def __init__(self, scale: float = 1.0, bias: float = 0.0) -> None:
        self.scale = scale
        self.bias = bias

    @returns_keys(output=float)
    def compute(self, value: float) -> float:
        return {"output": self.scale * value + self.bias}


# ---------------------------------------------------------------------------
# 2. Author DSL graphs and register templates
# ---------------------------------------------------------------------------

DSL_TEXT = """
GRAPH SCALE_CORE(version="1.0"):
    PARAMETER scale = 2.0
    INPUT value
    OUTPUT scaled = node.output

    node = ops.ScaleAndBias(scale=Param.scale, bias=0.0)[value=value]

GRAPH PIPELINE:
    PARAMETER bias = 0.5
    INPUT x, y

    helper = add_pair[left=x, right=y]
    scaled = Ref.SCALE_CORE(scale=3.0)[value=helper.result]
    out = add_pair[left=scaled.scaled, right=: Param.bias]

    OUTPUT total = out.result
"""

program = parse_dsl(DSL_TEXT)
scale_spec = program.build("SCALE_CORE")
register_graph("SCALE_CORE", scale_spec)
pipeline_spec = program.build("PIPELINE")



# ---------------------------------------------------------------------------
# 3. Build and execute a runtime
# ---------------------------------------------------------------------------

runtime = build_graph(pipeline_spec, parameters={"bias": 2.0})
runtime_outputs = runtime.run({"x": 2.0, "y": 1.5})


# ---------------------------------------------------------------------------
# 4. Serialise the GraphSpec
# ---------------------------------------------------------------------------

spec_dict = pipeline_spec.to_dict()
roundtrip_spec = GraphSpec.from_dict(spec_dict)


def main() -> None:
    print("Registered operators: add_pair, ScaleAndBias")
    print("\nPIPELINE nodes:")
    for node_id in pipeline_spec.nodes:
        print(f"  - {node_id}")

    print("\nRuntime result with bias override=2.0, inputs x=2.0, y=1.5:")
    pprint(runtime_outputs)

    print("\nSerialised GraphSpec (partial view):")
    pprint(
        {
            "parameters": spec_dict["parameters"],
            "inputs": spec_dict["inputs"],
            "nodes": list(spec_dict["nodes"].keys()),
        }
    )

    assert roundtrip_spec.to_dict() == spec_dict, "GraphSpec round-trip failed"
    print("\nRound-trip GraphSpec successful.")


if __name__ == "__main__":
    main()
