"""
Demonstrate the plain-text inspection utilities on a nested graph runtime.

Run with::

    python examples/inspect_plain_demo.py

It will print the graph structure using the new plain-text inspector and execute
the graph to show the computed outputs.
"""

import json
import math

from dag.inspect_utils import render_runtime_text, runtime_to_dict
from dag.node import GraphSpec, build_graph, register_function, returns_keys


@register_function(name="demo_factorial")
@returns_keys(result=int)
def factorial(n: int) -> int:
    return math.factorial(n)


@register_function(name="demo_divide")
@returns_keys(result=float)
def divide(dividend: float, divisor: float) -> float:
    return dividend / divisor


@register_function(name="demo_subtract")
@returns_keys(result=float)
def subtract(a: float, b: float) -> float:
    return a - b


@register_function(name="demo_power2")
@returns_keys(result=float)
def power2(a: float) -> float:
    return a**2


@register_function(name="demo_multiply3")
@returns_keys(result=float)
def multiply3(a: float, b: float, c: float) -> float:
    return a * b * c


def build_nested_runtime() -> GraphSpec:
    """Create a multi-level nested specification."""
    inner_spec = {
        "nodes": {
            "fact": {"operator": "demo_factorial"},
            "const_six": {"operator": "constant", "config": {"value": 6}},
            "const_ten": {"operator": "constant", "config": {"value": 10}},
            "mul60": {"operator": "multiplication"},
            "sum": {"operator": "addition"},
            "const_three": {"operator": "constant", "config": {"value": 3}},
            "div": {"operator": "demo_divide"},
            "sub": {"operator": "demo_subtract"},
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

    middle_spec = {
        "nodes": {
            "inner": {"operator": inner_spec, "metadata": {"name": "inner"}},
            "square": {"operator": "demo_power2"},
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

    return GraphSpec.from_dict(
        {
            "nodes": {
                "middle": {"operator": middle_spec, "metadata": {"name": "middle"}},
                "inner_again": {
                    "operator": inner_spec,
                    "metadata": {"name": "inner_again"},
                },
                "multiplier": {"operator": "demo_multiply3"},
                "const_five": {"operator": "constant", "config": {"value": 5}},
                "mul5": {"operator": "multiplication"},
                "const_two": {"operator": "constant", "config": {"value": 2}},
                "sub_y2": {"operator": "demo_subtract"},
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


def main() -> None:
    spec = build_nested_runtime()
    runtime = build_graph(spec)

    print("=== Plain Inspector ===")
    print(render_runtime_text(runtime, indent_step=4))

    print("\n=== Serialized Form ===")
    print(json.dumps(runtime_to_dict(runtime), indent=2))

    inputs = {"x": 2, "y": 5}
    outputs = runtime.run(inputs=inputs)
    print("\n=== Execution ===")
    print(f"inputs:  {inputs}")
    print(f"outputs: {outputs}")


if __name__ == "__main__":
    main()
