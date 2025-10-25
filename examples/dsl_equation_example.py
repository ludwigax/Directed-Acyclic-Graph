"""
Demonstrate building GraphSpecs from the DSL.

This example mirrors the earlier factorial/multiplication pipeline using the DSL
syntax. Run with::

    python examples/dsl_equation_example.py
"""

import math

from dag.dsl import parse_dsl
from dag.node import build_graph, register_function, returns_keys


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


dsl_text = """
graph inner_expression:
    input x, y

    fact = ops.factorial()[n=y]
    six = ops.constant(value=6)
    mul6x = ops.multiplication()[a=six._return, b=x]
    sum = ops.addition()[a=fact.result, b=mul6x.result]
    three = ops.constant(value=3)
    div = ops.divide()[dividend=sum.result, divisor=three._return]
    sub = ops.subtract()[a=sum.result, b=div.result]

    output result = sub.result

graph middle_expression:
    input x, y

    inner = Ref.inner_expression()[x=x, y=y]
    square = ops.power2()[a=x]
    sum = ops.addition()[a=inner.result, b=square.result]

    output result = sum.result

graph outer_expression:
    input x, y

    middle = Ref.middle_expression()[x=x, y=y]
    inner = Ref.inner_expression()[x=x, y=y]
    const_five = ops.constant(value=5)
    mul5 = ops.multiplication()[a=const_five._return, b=x]
    multiplier = ops.multiply3()[a=middle.result, b=inner.result, c=mul5.result]
    const_two = ops.constant(value=2)
    sub_y2 = ops.subtract()[a=y, b=const_two._return]
    outer_sum = ops.addition()[a=multiplier.result, b=sub_y2.result]

    output result = outer_sum.result
"""


def main() -> None:
    program = parse_dsl(dsl_text, globals=globals(), locals=locals())
    outer_spec = program.build("outer_expression")

    runtime = build_graph(outer_spec)
    inputs = {"x": 2, "y": 5}
    outputs = runtime.run(inputs=inputs)

    print("Inputs :", inputs)
    print("Outputs:", outputs)


if __name__ == "__main__":
    main()
