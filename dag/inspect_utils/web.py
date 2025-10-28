"""
Web visualisation utilities.

The previous implementation depended on the old nested-runtime architecture and has
been removed during the refactor. A refreshed visualiser built on top of execution
plans will arrive in a later iteration.
"""

from __future__ import annotations

from typing import Any

from ..node import ExecutionPlan


def create_runtime_app(runtime: ExecutionPlan) -> Any:  # pragma: no cover - placeholder
    raise NotImplementedError(
        "The FastAPI/Plotly visualiser is temporarily unavailable after the runtime "
        "refactor. Track the progress in the refactor plan and feel free to contribute "
        "an updated implementation."
    )


__all__ = ["create_runtime_app"]
