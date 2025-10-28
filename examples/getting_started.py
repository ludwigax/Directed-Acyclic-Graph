"""End-to-end walkthrough and stress test for the DAG package.

This script now covers two complementary scenarios:

1. A lightweight walkthrough that mirrors the introduction in
   ``docs/dag_dsl_snapshot.md`` (registering operators, compiling DSL graphs,
   running a plan, and round-tripping specifications).
2. A NumPy-heavy benchmark that compares the threaded execution plan with plain
   serial Python, including hook-based normalisation applied inside worker
   threads.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from pprint import pprint
from typing import Mapping

import sys
import numpy as np

if __package__ in (None, ""):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from dag.dsl import parse_dsl
from dag.inspect_utils import render_spec_tree
from dag.node import (
    GraphSpec,
    RegistrationError,
    build_graph,
    register_class,
    register_function,
    register_graph,
    returns_keys,
)


# ---------------------------------------------------------------------------
# Basic operators used in the introductory pipeline
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
# NumPy-heavy operators for the parallel execution demo
# ---------------------------------------------------------------------------


@register_function(name="generate_matrix", outputs={"matrix": "ndarray"})
def generate_matrix(seed: int, size: int) -> np.ndarray:
    """Create a symmetric positive-definite matrix for stable inversion."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(size, size))
    matrix = base @ base.T + np.eye(size) * (size * 1e-3)
    return matrix


@register_function(name="invert_matrix", outputs={"inverse": "ndarray"})
def invert_matrix(matrix: np.ndarray) -> np.ndarray:
    """Compute the matrix inverse."""
    return np.linalg.inv(matrix)


@register_function(name="combine_inverses", outputs={"metric": float})
def combine_inverses(left: np.ndarray, right: np.ndarray) -> float:
    """Fuse two inverses and return the spectral radius of their product."""
    product = left @ right
    eigenvalues = np.linalg.eigvals(product)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    return spectral_radius


# ---------------------------------------------------------------------------
# DSL programs
# ---------------------------------------------------------------------------

BASIC_DSL = """
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


NUMERIC_DSL = """
GRAPH NUMERIC_PIPELINE:
    PARAMETER size = 320
    INPUT seed_a, seed_b

    matrix_a = ops.generate_matrix(size=Param.size)[seed=seed_a]
    matrix_b = ops.generate_matrix(size=Param.size)[seed=seed_b]

    inverse_a = ops.invert_matrix[matrix=matrix_a.matrix]
    inverse_b = ops.invert_matrix[matrix=matrix_b.matrix]

    fused = ops.combine_inverses[left=inverse_a.inverse, right=inverse_b.inverse]

    OUTPUT fused_metric = fused.metric
"""


# ---------------------------------------------------------------------------
# Helper utilities shared by the advanced demo and hooks
# ---------------------------------------------------------------------------


def _standardise(matrix: np.ndarray) -> np.ndarray:
    """Centre and scale a matrix, keeping it invertible."""
    mean = float(matrix.mean())
    std = float(matrix.std())
    if std == 0.0:
        std = 1.0
    normalised = (matrix - mean) / std
    return normalised + np.eye(matrix.shape[0]) * 1e-3


def _squash_metric(value: float) -> float:
    """Logistic squashing to keep metrics within (0, 1)."""
    return float(1.0 / (1.0 + np.exp(-value)))


def _matrix_normalisation_hook(
    *,
    node_id: str,
    inputs: Mapping[str, np.ndarray],
    outputs: Mapping[str, np.ndarray],
    plan,
) -> Mapping[str, np.ndarray]:
    normalised = _standardise(outputs["matrix"])
    print(f"[hook:{node_id}] normalised matrix on {threading.current_thread().name}")
    return {"matrix": normalised}


def _metric_tuning_hook(
    *,
    node_id: str,
    inputs: Mapping[str, np.ndarray],
    outputs: Mapping[str, float],
    plan,
) -> Mapping[str, float]:
    tuned = _squash_metric(float(outputs["metric"]))
    print(f"[hook:{node_id}] squashed metric on {threading.current_thread().name}")
    return {"metric": tuned}


def _serial_numpy_baseline(*, seed_a: int, seed_b: int, size: int) -> float:
    """Reference implementation using plain Python + NumPy."""
    matrix_a = generate_matrix(seed=seed_a, size=size)
    matrix_b = generate_matrix(seed=seed_b, size=size)
    norm_a = _standardise(matrix_a)
    norm_b = _standardise(matrix_b)

    inverse_a = invert_matrix(norm_a)
    inverse_b = invert_matrix(norm_b)
    metric = combine_inverses(left=inverse_a, right=inverse_b)
    return _squash_metric(metric)


# ---------------------------------------------------------------------------
# Basic walkthrough
# ---------------------------------------------------------------------------


def run_basic_demo() -> None:
    program = parse_dsl(BASIC_DSL)
    scale_spec = program.build("SCALE_CORE")
    try:
        register_graph("SCALE_CORE", scale_spec)
    except RegistrationError:
        # Allow re-execution in the same interpreter session.
        pass
    pipeline_spec = program.build("PIPELINE")
    runtime = build_graph(pipeline_spec, parameters={"bias": 2.0})
    runtime_outputs = runtime.run({"x": 2.0, "y": 1.5})

    spec_dict = pipeline_spec.to_dict()
    roundtrip_spec = GraphSpec.from_dict(spec_dict)

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

    print("\nSpec tree for PIPELINE:")
    print(render_spec_tree(pipeline_spec, plan=runtime, root_name="PIPELINE"))


# ---------------------------------------------------------------------------
# Parallel NumPy workload
# ---------------------------------------------------------------------------


def run_parallel_numpy_demo() -> None:
    program = parse_dsl(NUMERIC_DSL)
    numeric_spec = program.build("NUMERIC_PIPELINE")
    plan = build_graph(numeric_spec, parameters={"size": numeric_spec.parameters["size"].default})

    # Hooks normalise matrices before inversion and squash the fused metric.
    plan.register_hook(["matrix_a", "matrix_b"], when="post", hook=_matrix_normalisation_hook)
    plan.register_hook("fused", when="post", hook=_metric_tuning_hook)

    seeds = {"seed_a": 11, "seed_b": 23}
    size = int(numeric_spec.parameters["size"].default or 320)

    print("\nComparing serial NumPy with ExecutionPlan (max_workers=1 vs 2)...")

    serial_start = time.perf_counter()
    serial_metric = _serial_numpy_baseline(seed_a=seeds["seed_a"], seed_b=seeds["seed_b"], size=size)
    serial_elapsed = time.perf_counter() - serial_start

    plan.clear_cache()
    sequential_start = time.perf_counter()
    sequential_outputs = plan.run(seeds, use_cache=False, max_workers=1)
    sequential_elapsed = time.perf_counter() - sequential_start

    plan.clear_cache()
    parallel_start = time.perf_counter()
    parallel_outputs = plan.run(seeds, use_cache=False, max_workers=2)
    parallel_elapsed = time.perf_counter() - parallel_start

    fused_seq = sequential_outputs["fused_metric"]
    fused_par = parallel_outputs["fused_metric"]

    assert np.isclose(fused_seq, serial_metric, atol=1e-9, rtol=1e-9), "Sequential plan mismatch"
    assert np.isclose(fused_par, serial_metric, atol=1e-9, rtol=1e-9), "Parallel plan mismatch"

    print(
        f"Serial Python:   {serial_elapsed:.3f}s  -> metric={serial_metric:.6f}\n"
        f"Plan max_workers=1: {sequential_elapsed:.3f}s\n"
        f"Plan max_workers=2: {parallel_elapsed:.3f}s"
    )
    if parallel_elapsed > 0:
        speedup = sequential_elapsed / parallel_elapsed
        print(f"Observed speed-up (plan): {speedup:.2f}x")

    print("\nSpec tree for NUMERIC_PIPELINE:")
    print(render_spec_tree(numeric_spec, plan=plan, root_name="NUMERIC_PIPELINE"))


def main() -> None:
    run_basic_demo()
    run_parallel_numpy_demo()


if __name__ == "__main__":
    main()
