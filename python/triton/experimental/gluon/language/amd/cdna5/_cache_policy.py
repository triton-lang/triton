"""Cache policy hints for gfx1250 (CDNA5) memory accesses."""

from __future__ import annotations
from dataclasses import dataclass

from triton.language.core import constexpr_type
from ..._core import _unwrap_if_constexpr

__all__ = ["CachePolicy"]

_TEMPORAL_HINTS = ("regular", "nontemporal", "high_temporal", "last_use")


@dataclass(frozen=True)
class CachePolicy:
    """
    Cache hints for gfx1250 (CDNA5) buffer accesses.

    The hints are performance-only and never affect coherence. The coherence
    scope of an access comes from the op's ``scope=`` argument instead.

    Args:
        temporal_hint (str): How long the data is expected to stay in the caches (TH):
            ``"regular"``, ``"nontemporal"`` (not expected to be reused), ``"high_temporal"``
            (expected to be reused) or, for loads only, ``"last_use"`` (the last read of the
            data, which the caches can drop). ``"last_use"`` can't be combined with the
            ``"sys"`` scope.
    """

    temporal_hint: str = "regular"

    def __post_init__(self):
        temporal_hint = _unwrap_if_constexpr(self.temporal_hint)
        if temporal_hint not in _TEMPORAL_HINTS:
            raise ValueError(f"temporal_hint must be one of {_TEMPORAL_HINTS}, got {temporal_hint!r}")
        object.__setattr__(self, "temporal_hint", temporal_hint)

    @property
    def type(self):
        return constexpr_type(self)

    def mangle(self) -> str:
        return f"AMDCP_gfx1250_{self.temporal_hint}_AMDCP"

    def _to_ir(self, builder):
        return builder.get_amd_cache_policy_gfx1250(self.temporal_hint)
