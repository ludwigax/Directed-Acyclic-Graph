"""
Tiny validation that graph-level parameters remain overridable after
registration and through the DSL `Ref.graph(...)` syntax.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dag.dsl import parse_dsl
from dag.node import (
    GraphSpec,
    ParameterRefValue,
    build_graph,
    register_function,
    register_graph,
    returns_keys,
)


@register_function(name="scale_mul")
@returns_keys(result=float)
def multiply(value: float, scale: float) -> float:
    return value * scale


def main() -> None:
    # ------------------------------------------------------------------ #
    # Register a graph template whose scale factor remains configurable. #
    # ------------------------------------------------------------------ #
    scaler_spec = GraphSpec.from_dict(
        {
            "parameters": {"scale": {"default": 2.0}},
            "nodes": {
                "mul": {
                    "operator": "scale_mul",
                    "config": {"call": {"scale": ParameterRefValue("scale")}},
                }
            },
            "edges": [],
            "inputs": {"value": "mul.value"},
            "outputs": {"result": "mul.result"},
            "metadata": {"name": "registered_scaler"},
        }
    )

    register_graph("scaler_template", scaler_spec)

    pipeline_spec = GraphSpec.from_dict(
        {
            "nodes": {
                "default": {"operator": "scaler_template"},
                "custom": {
                    "operator": "scaler_template",
                    "config": {"parameters": {"scale": 5.0}},
                },
            },
            "edges": [],
            "inputs": {"value": ["default.value", "custom.value"]},
            "outputs": {
                "default": "default.result",
                "custom": "custom.result",
            },
        }
    )

    pipeline_runtime = build_graph(pipeline_spec)
    pipeline_output = pipeline_runtime.run(inputs={"value": 3.0})

    print("registered graph default :", pipeline_output["default"])
    print("registered graph override:", pipeline_output["custom"])

    assert pipeline_output["default"] == 6.0
    assert pipeline_output["custom"] == 15.0

    # --------------------------------------------------------------- #
    # Same idea expressed in the DSL using Ref.graph(parameter=value) #
    # --------------------------------------------------------------- #
    user_scale = 7.0
    dsl_text = """
graph scaler:
    parameter scale
    input value

    node = ops.scale_mul(call={"scale": Param.scale : 2})[value=value]

    output result = node.result

graph wrapper:
    input value

    base = Ref.scaler()[value=value]
    tuned = Ref.scaler(scale=user_scale)[value=value]

    output base = base.result
    output tuned = tuned.result
"""

    program = parse_dsl(dsl_text, globals=globals(), locals=locals())
    wrapper_spec = program.build("wrapper")
    wrapper_runtime = build_graph(wrapper_spec)
    wrapper_output = wrapper_runtime.run(inputs={"value": 3.0})

    print("dsl wrapper base :", wrapper_output["base"])
    print("dsl wrapper tuned:", wrapper_output["tuned"])

    assert wrapper_output["base"] == 6.0
    assert wrapper_output["tuned"] == 21.0


if __name__ == "__main__":
    main()
