"""
Node templates, shells, and runtime wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from ..dbg import Debug, get_debug_state
from .inspect import (
    gather_callable_metadata,
    gather_class_metadata,
)
from .ports import (
    PortDefinition,
    ensure_mapping,
    ParameterSpec,
)


class NodeError(Exception):
    """Base class for node-related failures."""


class NodeInstantiationError(NodeError):
    """Raised when a node template cannot be instantiated."""


class CallableRunner(Debug):
    """Runtime wrapper around an operator implementation."""

    def __init__(self, *, name: str):
        super().__init__()
        self.name = name

    def compute(self, **kwargs) -> Mapping[str, Any]:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, **kwargs) -> Dict[str, Any]:
        if get_debug_state():
            self._start_timer()
        result = self.compute(**kwargs)
        if get_debug_state():
            self._stop_timer()
        if not isinstance(result, Mapping):
            raise NodeError(
                f"Operator '{self.name}' returned non-mapping output: {type(result)!r}"
            )
        return dict(result)


class FunctionRunner(CallableRunner):
    """Runner for plain Python callables."""

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str,
        output_keys: Sequence[str],
        call_defaults: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(name=name)
        self._func = func
        self._output_keys = list(output_keys)
        self._call_defaults = dict(call_defaults or {})

    def compute(self, **kwargs) -> Mapping[str, Any]:
        call_kwargs = {**self._call_defaults, **kwargs}
        result = self._func(**call_kwargs)
        return normalise_output(result, self._output_keys)


class ClassRunner(CallableRunner):
    """Runner for class instances exposing a forward-like method."""

    def __init__(
        self,
        *,
        instance: Any,
        forward: Callable[..., Any],
        name: str,
        output_keys: Sequence[str],
        call_defaults: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__(name=name)
        self._instance = instance
        self._forward = forward
        self._output_keys = list(output_keys)
        self._call_defaults = dict(call_defaults or {})

    def compute(self, **kwargs) -> Mapping[str, Any]:
        call_kwargs = {**self._call_defaults, **kwargs}
        result = self._forward(**call_kwargs)
        return normalise_output(result, self._output_keys)


@dataclass(frozen=True)
class NodeTemplate:
    """Factory descriptor for runnable operator instances."""

    name: str
    create_runner: Callable[[Mapping[str, Any], str], CallableRunner]
    input_ports: Mapping[str, PortDefinition] = field(default_factory=dict)
    output_ports: Mapping[str, PortDefinition] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def instantiate(
        self,
        *,
        config: Optional[Mapping[str, Any]] = None,
        runtime_id: Optional[str] = None,
    ) -> CallableRunner:
        runner = self.create_runner(dict(config or {}), runtime_id or self.name)
        if not isinstance(runner, CallableRunner):
            raise NodeInstantiationError(
                f"Node template '{self.name}' produced an invalid runner: "
                f"{type(runner)!r}"
            )
        return runner

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
        outputs: Optional[Sequence[str] | Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "NodeTemplate":
        if not callable(func):
            raise NodeInstantiationError(f"Expected callable, got {type(func)!r}")

        input_ports, output_keys, output_ports = gather_callable_metadata(
            func, explicit_outputs=outputs
        )

        def factory(config: Mapping[str, Any], runtime_id: str) -> CallableRunner:
            cfg = dict(config or {})
            call_defaults: Dict[str, Any] = {}

            if "call" in cfg:
                call_defaults.update(
                    ensure_mapping(cfg.pop("call"), "call defaults")
                )

            if "init" in cfg:
                init_cfg = ensure_mapping(cfg.pop("init"), "function init")
                for key, value in init_cfg.items():
                    if key in call_defaults:
                        raise NodeInstantiationError(
                            f"Function operator '{func.__name__}' received "
                            f"conflicting init/call default for '{key}'"
                        )
                    call_defaults[key] = value

            for key, value in list(cfg.items()):
                call_defaults.setdefault(key, value)

            return FunctionRunner(
                func,
                name=runtime_id or func.__name__,
                output_keys=output_keys,
                call_defaults=call_defaults,
            )

        return cls(
            name=name or func.__name__,
            create_runner=factory,
            input_ports=dict(input_ports),
            output_ports=dict(output_ports),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_class(
        cls,
        operator_cls: type,
        *,
        name: Optional[str] = None,
        forward: str = "forward",
        outputs: Optional[Sequence[str] | Mapping[str, Any]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "NodeTemplate":
        input_ports, output_keys, output_ports, init_defaults = gather_class_metadata(
            operator_cls,
            forward=forward,
            explicit_outputs=outputs,
        )

        def factory(config: Mapping[str, Any], runtime_id: str) -> CallableRunner:
            cfg = dict(config or {})
            init_kwargs = dict(init_defaults)
            call_defaults: Dict[str, Any] = {}

            if "init" in cfg:
                init_kwargs.update(ensure_mapping(cfg.pop("init"), "init"))
            if "call" in cfg:
                call_defaults.update(ensure_mapping(cfg.pop("call"), "call"))

            init_kwargs.update(cfg)

            instance = operator_cls(**init_kwargs)
            forward_method = getattr(instance, forward)
            return ClassRunner(
                instance=instance,
                forward=forward_method,
                name=runtime_id or operator_cls.__name__,
                output_keys=output_keys,
                call_defaults=call_defaults,
            )

        return cls(
            name=name or operator_cls.__name__,
            create_runner=factory,
            input_ports=dict(input_ports),
            output_ports=dict(output_ports),
            metadata=dict(metadata or {}),
        )


@dataclass
class NodeShell:
    """Template instance awaiting runtime materialisation."""

    id: str
    template: NodeTemplate
    config: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def instantiate(self, runtime_id: Optional[str] = None) -> "NodeRuntime":
        runner = self.template.instantiate(
            config=self.config,
            runtime_id=runtime_id or self.id,
        )
        return NodeRuntime(
            node_id=runtime_id or self.id,
            template=self.template,
            runner=runner,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class GraphTemplate:
    """Intermediate structure bridging specs and runtime execution."""

    nodes: Mapping[str, NodeShell]
    edges: Sequence[tuple[str, str, str, str]]
    inputs: Mapping[str, Sequence[tuple[str, str]]]
    outputs: Mapping[str, tuple[str, str]]
    parameters: Mapping[str, "ParameterSpec"]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    shell_index: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


class NodeRuntime:
    """Runtime node holding instantiated operator."""

    def __init__(
        self,
        *,
        node_id: str,
        template: NodeTemplate,
        runner: CallableRunner,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        self.id = node_id
        self.template = template
        self.runner = runner
        self.metadata = dict(metadata or {})
        self.input_ports = dict(template.input_ports)
        self.output_ports = dict(template.output_ports)
        cache_flag = self.metadata.get("cache_enabled", True)
        self.cache_enabled = bool(cache_flag) if isinstance(cache_flag, bool) else bool(cache_flag)
        self._cache_valid = False
        self._cache_inputs: Optional[Dict[str, Any]] = None
        self._cache_outputs: Optional[Dict[str, Any]] = None
        self._cache_lock = threading.Lock()

    def run(
        self,
        inputs: Mapping[str, Any],
        *,
        use_cache: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        inputs_dict = dict(inputs)

        if self.cache_enabled and use_cache and not force:
            with self._cache_lock:
                if self._cache_valid and self._cache_inputs == inputs_dict:
                    return dict(self._cache_outputs or {})

        outputs = self.runner(**inputs_dict)
        outputs_dict = dict(outputs)

        if self.cache_enabled:
            with self._cache_lock:
                self._cache_valid = True
                self._cache_inputs = inputs_dict
                self._cache_outputs = outputs_dict
        else:
            self.clear_cache()

        return dict(outputs_dict)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache_valid = False
            self._cache_inputs = None
            self._cache_outputs = None

    def get_cached(self) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        with self._cache_lock:
            if not self._cache_valid:
                return None
            inputs = dict(self._cache_inputs or {})
            outputs = dict(self._cache_outputs or {})
            return inputs, outputs

    def update_cache(self, inputs: Mapping[str, Any], outputs: Mapping[str, Any]) -> None:
        with self._cache_lock:
            if self.cache_enabled:
                self._cache_valid = True
                self._cache_inputs = dict(inputs)
                self._cache_outputs = dict(outputs)
            else:
                self._cache_valid = False
                self._cache_inputs = None
                self._cache_outputs = None

    def set_cache(self, inputs: Mapping[str, Any], outputs: Mapping[str, Any]) -> None:
        self.update_cache(inputs, outputs)


def normalise_output(
    value: Any,
    output_keys: Sequence[str],
) -> Dict[str, Any]:
    keys = list(output_keys)
    if isinstance(value, Mapping):
        if keys and any(key not in value for key in keys):
            missing = [key for key in keys if key not in value]
            raise NodeError(
                f"Operator result missing keys: {', '.join(missing)}"
            )
        return dict(value)

    if hasattr(value, "_asdict"):
        mapped = value._asdict()  # type: ignore[attr-defined]
        return normalise_output(mapped, keys)

    if len(keys) == 1:
        return {keys[0]: value}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != len(keys):
            raise NodeError(
                f"Expected {len(keys)} outputs, received {len(value)}"
            )
        return dict(zip(keys, value))

    raise NodeError(
        "Unable to normalise operator output; provide explicit "
        "returns_keys() metadata or return a mapping."
    )


__all__ = [
    "NodeError",
    "NodeInstantiationError",
    "CallableRunner",
    "FunctionRunner",
    "ClassRunner",
    "NodeTemplate",
    "NodeShell",
    "GraphTemplate",
    "NodeRuntime",
    "normalise_output",
]
