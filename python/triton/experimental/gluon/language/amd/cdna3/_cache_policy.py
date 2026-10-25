"""Cache policy hints for gfx9 (CDNA3/CDNA4) memory accesses."""

from __future__ import annotations
from dataclasses import dataclass

from triton.language.core import constexpr_type
from ..._core import _unwrap_if_constexpr

__all__ = ["CachePolicy"]


@dataclass(frozen=True)
class CachePolicy:
    """
    Cache hints for gfx9 (CDNA3/CDNA4) buffer accesses.

    The hints are performance-only and never affect coherence. The coherence
    scope of an access comes from the op's ``scope=`` argument instead.

    Args:
        nontemporal (bool): Set NT: the data is not expected to be reused.
    """

    nontemporal: bool = False

    def __post_init__(self):
        nontemporal = _unwrap_if_constexpr(self.nontemporal)
        if not isinstance(nontemporal, bool):
            raise TypeError("nontemporal must be a bool")
        object.__setattr__(self, "nontemporal", nontemporal)

    @property
    def type(self):
        return constexpr_type(self)

    def mangle(self) -> str:
        return f"AMDCP_gfx9_{int(self.nontemporal)}_AMDCP"

    def _to_ir(self, builder):
        return builder.get_amd_cache_policy_gfx9(self.nontemporal)
