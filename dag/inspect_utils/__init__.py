"""DAG inspection and visualization utilities."""

from .plain import (
    print_module,
    print_runtime,
    render_runtime_text,
    runtime_to_dict,
)

# Interactive visualization (legacy Module-based implementation).
try:  # pragma: no cover - legacy path
    from .flask_app import DAGVisualizationApp, create_app
    from .launcher import launch_visualizer, visualize_module
except Exception:  # pylint: disable=broad-except
    DAGVisualizationApp = None  # type: ignore[assignment]

    def create_app(*args, **kwargs):  # type: ignore[unused-arg]
        raise ImportError(
            "Interactive inspector is unavailable; legacy Module API was removed."
        )

    def launch_visualizer(*args, **kwargs):  # type: ignore[unused-arg]
        raise ImportError(
            "Interactive inspector is unavailable; legacy Module API was removed."
        )

    def visualize_module(*args, **kwargs):  # type: ignore[unused-arg]
        raise ImportError(
            "Interactive inspector is unavailable; legacy Module API was removed."
        )
