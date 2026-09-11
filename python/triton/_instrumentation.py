from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

_callbacks: dict[tuple[str, str], Callable[..., None]] = {}


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


def register_instrumentation(*, point: str, backend: str, callback: Callable[..., None]) -> None:
    key = (point, backend)
    if key in _callbacks:
        raise RuntimeError(f"Instrumentation already registered: {key}")
    _callbacks[key] = callback


def unregister_instrumentation(*, point: str, backend: str) -> None:
    _callbacks.pop((point, backend), None)


def instrument(*args: Any, point: str, backend: str, context: Any = None) -> bool:
    callback = _callbacks.get((point, backend))
    if callback is None:
        return False

    load_dialects = _callbacks.get(("load-dialects", backend))
    if point != "load-dialects" and load_dialects is not None:
        load_dialects(context)

    callback(*args)
    return True
