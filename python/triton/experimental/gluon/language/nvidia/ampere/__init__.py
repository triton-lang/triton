from __future__ import annotations

from dataclasses import dataclass
import math

from triton import knobs
from triton.language.core import constexpr_type
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._layouts import DotOperandLayout, NVMMADistributedLayout
from ..._core import builtin, _unwrap_if_constexpr
from . import async_copy, mbarrier

__all__ = ["CachePolicy", "FractionalEvictionPolicy", "async_copy", "mbarrier", "mma_v2"]

_EVICTION_PRIORITIES = {
    "evict_normal",
    "evict_unchanged",
    "evict_first",
    "evict_last",
}
_L1_EVICTION_PRIORITIES = _EVICTION_PRIORITIES | {"no_allocate"}
_L2_SECONDARY_EVICTION_PRIORITIES = {"evict_unchanged", "evict_first"}
_CACHE_MODIFIERS = {"ca", "cg", "wb", "cs", "wt", "cv"}


@dataclass(frozen=True)
class FractionalEvictionPolicy:
    """A fractional NVIDIA L2 eviction policy.

    ``fraction`` is the fraction of cache lines assigned ``primary``. Remaining
    lines receive ``secondary``.
    """

    primary: str
    fraction: float
    secondary: str = "evict_unchanged"

    def __post_init__(self):
        object.__setattr__(self, "primary", _unwrap_if_constexpr(self.primary))
        object.__setattr__(self, "fraction", _unwrap_if_constexpr(self.fraction))
        object.__setattr__(self, "secondary", _unwrap_if_constexpr(self.secondary))
        if self.primary not in _EVICTION_PRIORITIES:
            raise ValueError(f"Unsupported L2 primary eviction priority {self.primary!r}")
        if self.secondary not in _L2_SECONDARY_EVICTION_PRIORITIES:
            raise ValueError(f"Unsupported L2 secondary eviction priority {self.secondary!r}")
        if not isinstance(self.fraction, (int, float)) or not math.isfinite(self.fraction):
            raise ValueError("L2 eviction fraction must be finite")
        if not 0 < self.fraction <= 1:
            raise ValueError("L2 eviction fraction must be in (0, 1]")

    @property
    def type(self):
        return constexpr_type(self)

    def mangle(self) -> str:
        fraction = float(self.fraction).hex().replace(".", "d").replace("-", "m").replace("+", "p")
        return f"FEP_{self.primary}_{fraction}_{self.secondary}_FEP"


@dataclass(frozen=True)
class CachePolicy:
    """Detailed NVIDIA cache policy for Gluon memory operations."""

    cache_modifier: str | None = None
    l1: str | None = None
    l2: FractionalEvictionPolicy | None = None
    l2_prefetch_size: int | None = None

    def __post_init__(self):
        cache_modifier = _unwrap_if_constexpr(self.cache_modifier)
        if cache_modifier is not None:
            cache_modifier = cache_modifier.removeprefix(".")
        object.__setattr__(self, "cache_modifier", cache_modifier)
        object.__setattr__(self, "l1", _unwrap_if_constexpr(self.l1))
        object.__setattr__(self, "l2", _unwrap_if_constexpr(self.l2))
        object.__setattr__(self, "l2_prefetch_size", _unwrap_if_constexpr(self.l2_prefetch_size))
        if self.cache_modifier is not None and self.cache_modifier not in _CACHE_MODIFIERS:
            raise ValueError(f"Unsupported cache modifier {self.cache_modifier!r}")
        if self.l1 is not None and self.l1 not in _L1_EVICTION_PRIORITIES:
            raise ValueError(f"Unsupported L1 eviction priority {self.l1!r}")
        if self.l2 is not None and not isinstance(self.l2, FractionalEvictionPolicy):
            raise TypeError("l2 must be a FractionalEvictionPolicy")
        if self.l2_prefetch_size is not None and (not isinstance(self.l2_prefetch_size, int)
                                                  or self.l2_prefetch_size not in (64, 128, 256)):
            raise ValueError("L2 prefetch size must be 64, 128, or 256")
        if (self.cache_modifier is None and self.l1 is None and self.l2 is None and self.l2_prefetch_size is None):
            raise ValueError("CachePolicy requires a cache modifier, L1 policy, L2 policy, or L2 prefetch size")

    @property
    def type(self):
        return constexpr_type(self)

    def mangle(self) -> str:
        cache_modifier = self.cache_modifier or "none"
        l1 = self.l1 or "none"
        l2 = self.l2.mangle() if self.l2 is not None else "none"
        l2_prefetch_size = self.l2_prefetch_size or "none"
        return f"CP_{cache_modifier}_{l1}_{l2}_{l2_prefetch_size}_CP"

    def _to_ir(self, builder):
        if self.l2 is None:
            return builder.get_nvidia_cache_policy(self.cache_modifier, self.l1, None, None, None,
                                                   self.l2_prefetch_size)
        return builder.get_nvidia_cache_policy(self.cache_modifier, self.l1, self.l2.primary, self.l2.secondary,
                                               float(self.l2.fraction), self.l2_prefetch_size)


@builtin
def mma_v2(a, b, acc, input_precision=None, _semantic=None):
    input_precision = _unwrap_if_constexpr(input_precision)
    assert isinstance(a, ttgl.tensor), "a must be a tensor"
    assert isinstance(b, ttgl.tensor), "b must be a tensor"
    assert isinstance(acc, ttgl.tensor), "acc must be a tensor"

    mma_layout = acc.type.layout
    assert isinstance(mma_layout, NVMMADistributedLayout), "acc must have an NVMMADistributedLayout"
    assert mma_layout.version == [2, 0], "MMA layout must have version 2.0"

    assert isinstance(a.type.layout, DotOperandLayout), "a must have a DotOperandLayout"
    assert isinstance(b.type.layout, DotOperandLayout), "b must have a DotOperandLayout"
    assert a.type.layout.parent == mma_layout, "a's parent layout must be the same as acc's layout"
    assert b.type.layout.parent == mma_layout, "b's parent layout must be the same as acc's layout"
    assert a.type.layout.operand_index == 0, "a's operand index must be 0"
    assert b.type.layout.operand_index == 1, "b's operand index must be 1"

    handle = _semantic.dot(a, b, acc, input_precision=input_precision, max_num_imprecise_acc=None,
                           out_dtype=acc.dtype).handle
    return ttgl.tensor(handle, acc.type)
