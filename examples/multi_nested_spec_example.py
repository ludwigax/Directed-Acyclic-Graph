"""
Multi-level nested graph specification example.

Final expression (for x, y inputs):

    final_result = outer_mid_result + (inner_expression(x, y) * outer_inner_expression(x, y))

Where:
    inner_expression(x, y) = (factorial(y) + 6 * 10) - x / 3
    outer_inner_expression(x, y) = inner_expression(x, y) + x^2
    outer_mid_result = (x * 5) - (y - 2)
"""

import math

from dag.dbg import DebuggingContext
from dag.node import GraphSpec, build_graph, register_function, returns_keys


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


@register_function(name="power2")
@returns_keys(result=float)
def power2(a: float) -> float:
    return a**2


@register_function(name="multiply3")
@returns_keys(result=float)
def multiply3(a: float, b: float, c: float) -> float:
    return a * b * c


# innermost expression: (factorial(y) + 6 * 10) - x / 3
inner_spec = {
    "nodes": {
        "fact": {"operator": "factorial"},
        "const_six": {"operator": "constant", "config": {"value": 6}},
        "const_ten": {"operator": "constant", "config": {"value": 10}},
        "mul60": {"operator": "multiplication"},
        "sum": {"operator": "addition"},
        "const_three": {"operator": "constant", "config": {"value": 3}},
        "div": {"operator": "divide"},
        "sub": {"operator": "subtract"},
    },
    "edges": [
        {"src": "const_six._return", "dst": "mul60.a"},
        {"src": "const_ten._return", "dst": "mul60.b"},
        {"src": "fact.result", "dst": "sum.a"},
        {"src": "mul60.result", "dst": "sum.b"},
        {"src": "sum.result", "dst": "sub.a"},
        {"src": "div.result", "dst": "sub.b"},
        {"src": "const_three._return", "dst": "div.divisor"},
    ],
    "inputs": {"x": "div.dividend", "y": "fact.n"},
    "outputs": {"inner_result": "sub.result"},
    "metadata": {"name": "inner_expression"},
}

# middle expression: inner_expression + x^2
middle_spec = {
    "nodes": {
        "inner": {"operator": inner_spec, "metadata": {"name": "inner"}},
        "square": {"operator": "power2"},
        "sum": {"operator": "addition"},
    },
    "edges": [
        {"src": "inner.inner_result", "dst": "sum.a"},
        {"src": "square.result", "dst": "sum.b"},
    ],
    "inputs": {"x": ["inner.x", "square.a"], "y": "inner.y"},
    "outputs": {"middle_result": "sum.result"},
    "metadata": {"name": "middle_expression"},
}

# outer expression combines middle_spec result with other terms
outer_spec = GraphSpec.from_dict(
    {
        "nodes": {
            "middle": {"operator": middle_spec, "metadata": {"name": "middle"}},
            "inner_again": {
                "operator": inner_spec,
                "metadata": {"name": "inner_again"},
            },
            "multiplier": {"operator": "multiply3"},
            "const_five": {"operator": "constant", "config": {"value": 5}},
            "mul5": {"operator": "multiplication"},
            "const_two": {"operator": "constant", "config": {"value": 2}},
            "sub_y2": {"operator": "subtract"},
            "outer_sum": {"operator": "addition"},
        },
        "edges": [
            {"src": "const_five._return", "dst": "mul5.a"},
            {"src": "middle.middle_result", "dst": "multiplier.a"},
            {"src": "inner_again.inner_result", "dst": "multiplier.b"},
            {"src": "mul5.result", "dst": "multiplier.c"},
            {"src": "const_two._return", "dst": "sub_y2.b"},
            {"src": "multiplier.result", "dst": "outer_sum.a"},
            {"src": "sub_y2.result", "dst": "outer_sum.b"},
        ],
        "inputs": {
            "x": ["middle.x", "inner_again.x", "mul5.b"],
            "y": ["middle.y", "inner_again.y", "sub_y2.a"],
        },
        "outputs": {"result": "outer_sum.result"},
        "metadata": {"name": "outer_expression"},
    }
)

runtime = build_graph(outer_spec)

inputs = {
    "x": 2,
    "y": 5,
}

with DebuggingContext(True):
    outputs = runtime.run(inputs=inputs)

print(f"Inputs: {inputs}")
print(f"Outputs: {outputs}")
for node_id, node_runtime in runtime.node_runtimes.items():
    print(node_id, node_runtime.runner.get_stats())
