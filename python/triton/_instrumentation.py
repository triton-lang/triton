from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

_callbacks: dict[tuple[str, str, str], Callable[..., None]] = {}


def _mode_names(options: Any) -> set[str]:
    if options is None:
        return set()
    if isinstance(options, Mapping):
        # Raw compiler options and runtime metadata store the mode as a mapping key.
        mode = options.get("instrumentation_mode", "")
    else:
        # Parsed backend options (for example, CUDAOptions) store it as an attribute.
        mode = getattr(options, "instrumentation_mode", "")
    return {name.strip() for name in mode.split(",") if name.strip()}


def is_enabled(options: Any, name: str) -> bool:
    return name in _mode_names(options)


def register_instrumentation(*, name: str, point: str, backend: str, callback: Callable[..., None]) -> None:
    key = (name, point, backend)
    if key in _callbacks:
        raise RuntimeError(f"Instrumentation already registered: {key}")
    _callbacks[key] = callback


def unregister_instrumentation(*, name: str, point: str, backend: str) -> None:
    _callbacks.pop((name, point, backend), None)


def instrument(*args: Any, name: str, point: str, backend: str, options: Any = None) -> bool:
    if not is_enabled(options, name):
        return False

    key = (name, point, backend)
    callback = _callbacks.get(key)
    if callback is None:
        return False

    callback(*args, options)
    return True
