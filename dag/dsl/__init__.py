"""Public entrypoints for the DAG DSL package."""

from __future__ import annotations

from .parser import DSLParseError
from .program import DSLProgram, DSLEvaluationError, parse_dsl
from .invocation import op

__all__ = [
    "DSLProgram",
    "DSLParseError",
    "DSLEvaluationError",
    "op",
    "parse_dsl",
]
