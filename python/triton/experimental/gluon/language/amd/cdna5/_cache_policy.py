"""Cache policy hints for CDNA5 memory accesses."""

from __future__ import annotations
from dataclasses import dataclass

from triton.language.core import constexpr_type
from ..._core import _unwrap_if_constexpr
from ..._semantic import _check

__all__ = ["CachePolicy"]

_TEMPORAL_HINTS = ("regular", "none", "high", "last_use")


@dataclass(frozen=True)
class CachePolicy:
    """
    Cache hints for CDNA5 memory accesses, taken by the ``buffer``, ``async_copy``
    and ``tdm`` ops.

    The hints are performance-only and never affect coherence. The coherence
    scope of an access comes from the op's ``scope=`` argument instead.

    Args:
        temporal (str): How long the data is expected to stay in the caches (TH): ``"regular"``,
            ``"none"`` (not expected to be reused), ``"high"`` (expected to be reused) or, for loads
            only, ``"last_use"`` (the last read of the data, which the caches can drop).
            ``"last_use"`` can't be combined with the ``"sys"`` scope.
    """

    temporal: str = "regular"

    def __post_init__(self):
        temporal = _unwrap_if_constexpr(self.temporal)
        if temporal not in _TEMPORAL_HINTS:
            raise ValueError(f"temporal must be one of {_TEMPORAL_HINTS}, got {temporal!r}")
        object.__setattr__(self, "temporal", temporal)

    @property
    def type(self):
        return constexpr_type(self)

    def mangle(self) -> str:
        return f"AMDCP_gfx12_{self.temporal}_AMDCP"

    def _to_ir(self, builder):
        return builder.get_amd_cache_policy_gfx12(self.temporal)


def _policy_and_scope(cache_policy, scope, cache_modifier, semantic):
    """Return the `cachePolicy` attribute, or None, and the optional scope of an access.

    A legacy ``cache_modifier`` implies its own scope, so it can't be combined with either.
    """
    cache_policy = _unwrap_if_constexpr(cache_policy)
    scope = _unwrap_if_constexpr(scope)
    cache_modifier = _unwrap_if_constexpr(cache_modifier)
    _check(not cache_modifier or (cache_policy is None and scope is None),
           lambda: "cache_modifier can't be combined with cache_policy or scope")
    if cache_policy is not None:
        _check(isinstance(cache_policy, CachePolicy),
               lambda: f"cache_policy must be a {CachePolicy.__qualname__}, got {cache_policy!r}")
        cache_policy = cache_policy._to_ir(semantic.builder)
    if scope is not None:
        _check(scope in ("cta", "gpu", "sys"), lambda: f"scope must be 'cta', 'gpu' or 'sys', got {scope!r}")
        scope = semantic._str_to_scope(scope)
    return cache_policy, scope
