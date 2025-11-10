from __future__ import annotations
from typing import Optional

class warp_pipeline_stage:
    __slots__ = ("label", "_semantic")

    def __init__(self, label: Optional[str] = None, **_internal_kwargs):
        self.label = label
        self._semantic = _internal_kwargs.pop("_semantic", None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            return False
        try:
            from . import split_warp_pipeline
            try:
                split_warp_pipeline(_semantic=self._semantic)
            except TypeError:
                split_warp_pipeline()
        except Exception:
            pass
        return False