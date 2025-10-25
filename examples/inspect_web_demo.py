"""
Launch the optional FastAPI/Plotly runtime visualiser.

This example reuses the nested graph specification from ``inspect_plain_demo``
and spins up the interactive viewer (if the optional dependencies are
installed). Run with::

    python examples/inspect_web_demo.py

Install extras first if needed::

    pip install fastapi uvicorn plotly networkx
"""

from inspect_plain_demo import build_nested_runtime

from dag.inspect_utils import visualize_runtime
from dag.node import build_graph


def main() -> None:
    spec = build_nested_runtime()
    runtime = build_graph(spec)
    print("Starting runtime visualiser on http://127.0.0.1:8000 ...")
    try:
        visualize_runtime(runtime)
    except ImportError as exc:
        print(
            "Runtime visualiser dependencies are missing.\n"
            "Install them with:\n"
            "    pip install fastapi uvicorn plotly networkx\n"
            f"Details: {exc}"
        )


if __name__ == "__main__":
    main()
