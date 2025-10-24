"""
Example script building a declarative DAG that evaluates:

    y! + 6 * 10 - x / 3

using the new GraphSpec/GraphRuntime API.
"""

import math

from dag.dbg import DebuggingContext
from dag.node import (
    GraphSpec,
    build_graph,
    register_function,
    register_graph,
    returns_keys,
)


@register_function(name="factorial")
@returns_keys(result=int)
def factorial(n: int) -> int:
    return math.factorial(n)


@register_function(name="divide")
@returns_keys(result=float)
def divide(dividend: float, divisor: float) -> float:
    return dividend / divisor


@register_function(name="subtract")
@returns_keys(result=float)
def subtract(a: float, b: float) -> float:
    return a - b


spec_dict = {
    "nodes": {
        "fact": {"operator": "factorial"},
        "const_six": {"operator": "constant", "config": {"value": 6}},
        "const_ten": {"operator": "constant", "config": {"value": 10}},
        "mul60": {"operator": "multiplication"},
        "const_three": {"operator": "constant", "config": {"value": 3}},
        "div_x": {"operator": "divide"},
        "sum": {"operator": "addition"},
        "sub": {"operator": "subtract"},
    },
    "edges": [
        {"src": "const_six._return", "dst": "mul60.a"},
        {"src": "const_ten._return", "dst": "mul60.b"},
        {"src": "fact.result", "dst": "sum.a"},
        {"src": "mul60.result", "dst": "sum.b"},
        {"src": "const_three._return", "dst": "div_x.divisor"},
        {"src": "sum.result", "dst": "sub.a"},
        {"src": "div_x.result", "dst": "sub.b"},
    ],
    "inputs": {
        "x": "div_x.dividend",
        "y": "fact.n",
    },
    "outputs": {
        "result": "sub.result",
    },
}

graph_runtime = build_graph(GraphSpec.from_dict(spec_dict))

inputs = {"x": 2, "y": 5}
with DebuggingContext(True):
    outputs = graph_runtime.run(inputs=inputs)

print(f"Base graph inputs: {inputs}")
print(f"Base graph outputs: {outputs}")
print(
    "Factorial node stats:",
    graph_runtime.node_runtimes["fact"].runner.get_stats(),
)

# Register the base graph as a reusable operator and invoke it from a wrapper graph
register_graph("math_expression", GraphSpec.from_dict(spec_dict))

outer_spec = GraphSpec.from_dict(
    {
        "nodes": {
            "inner": {"operator": "math_expression"},
        },
        "edges": [],
        "inputs": {"x": "inner.x", "y": "inner.y"},
        "outputs": {"result": "inner.result"},
    }
)

outer_runtime = build_graph(outer_spec)
with DebuggingContext(True):
    outer_outputs = outer_runtime.run(inputs=inputs)

print(f"Outer graph outputs: {outer_outputs}")
print(
    "Outer graph inner node stats:",
    outer_runtime.node_runtimes["inner"].runner.get_stats(),
)
