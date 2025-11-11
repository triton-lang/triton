from __future__ import annotations
from typing import Optional

class warp_pipeline_stage:
    __slots__ = ("label", "_semantic")

    def __init__(self, label: Optional[str] = None, **_internal):
        self.label = label
        self._semantic = _internal.get("_semantic", None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            return False
        if self._semantic is None:
            return False
        self._semantic.builder.create_warp_pipeline_border()
        return False
