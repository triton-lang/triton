from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

from triton import knobs
from triton.language.core import constexpr_type
from triton.experimental.gluon.language import _core as ttgl
from triton._C.libtriton import ir
from ..._core import builtin, int8, uint8, _unwrap_if_constexpr
from .._layouts import AMDMFMALayout
from .._ops import _scaled_upcast, get_scaled_upcast_fp4_scale_layout
from ..._semantic import _check

if TYPE_CHECKING:
    from ..._semantic import GluonSemantic

__all__ = [
    "CachePolicy", "buffer_descriptor", "buffer_descriptor_type", "make_buffer_descriptor", "buffer_atomic_add",
    "buffer_atomic_and", "buffer_atomic_min", "buffer_atomic_max", "buffer_atomic_or", "buffer_atomic_xor",
    "buffer_atomic_xor", "buffer_load", "buffer_store", "mfma", "scaled_upcast", "get_scaled_upcast_fp4_scale_layout"
]

_atomic_op_str_to_op = {
    "smax": ir.ATOMIC_OP.MAX, "smin": ir.ATOMIC_OP.MIN, "umax": ir.ATOMIC_OP.UMAX, "umin": ir.ATOMIC_OP.UMIN, "fadd":
    ir.ATOMIC_OP.FADD, "iadd": ir.ATOMIC_OP.ADD, "and": ir.ATOMIC_OP.AND, "or": ir.ATOMIC_OP.OR, "xor":
    ir.ATOMIC_OP.XOR, "xchg": ir.ATOMIC_OP.XCHG
}


@dataclass(eq=True)
class buffer_descriptor_type(ttgl.base_type):
    """The type of a buffer resource descriptor accessing ``element_ty`` elements."""

    element_ty: ttgl.dtype

    def __str__(self) -> str:
        return f"buffer_descriptor<{self.element_ty}>"

    def _to_ir(self, builder: ir.builder) -> ir.type:
        return builder.get_buffer_desc_type(self.element_ty.to_ir(builder))

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[buffer_descriptor, int]:
        return buffer_descriptor(handles[cursor], self), cursor + 1

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self._to_ir(builder))

    def mangle(self) -> str:
        return f"BD{self.element_ty.mangle()}BD"


@dataclass
class buffer_descriptor(ttgl.base_value):
    """
    A buffer resource descriptor (V#), made by :func:`make_buffer_descriptor`.

    It is a scalar, program-uniform value held in SGPRs. Pass it as ``ptr`` to
    :func:`buffer_load`, :func:`buffer_store` and
    :func:`~triton.experimental.gluon.language.amd.cdna4.async_copy.buffer_load_to_shared`.
    """

    handle: ir.value
    type: buffer_descriptor_type

    def _set_name(self, builder: ir.builder, name: str) -> None:
        self.handle.set_loc(builder.create_name_loc(name, self.handle.get_loc()))

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    @property
    def dtype(self):
        return self.type.element_ty


@builtin
def make_buffer_descriptor(base, num_records, stride=0, _semantic=None) -> buffer_descriptor:
    """
    Make a buffer resource descriptor from the hardware fields the program controls.

    Buffer loads and stores through the descriptor address
    ``base + soffset + offsets[i]`` (plus ``indices[i] * stride`` when they take
    ``indices``), and the hardware checks bounds from ``num_records``:

    * ``stride == 0``: an access is out of bounds when its byte offset from
      ``base``, including ``soffset``, is at least ``num_records``.
    * ``stride != 0``: an access without ``indices`` is out of bounds when its
      per-lane byte offset (excluding ``soffset``) is at least ``num_records``;
      an access with ``indices`` is out of bounds when its record index is at
      least ``num_records``, whatever its offsets.

    Out-of-bounds loads return zero and out-of-bounds stores are dropped, so a
    descriptor with ``num_records=0`` disables every access. Swizzling,
    thread-ID addressing, data formats and the out-of-bounds mode are chosen by
    the compiler and not exposed.

    Args:
        base (pointer to scalar): Global memory base pointer. Its pointee type is
            the element type of the descriptor; offsets count these elements.
        num_records (int32 scalar): Bytes accessible from ``base`` (``stride == 0``),
            or the number of records (``stride != 0``). At most 2**31 - 1.
        stride (int32 scalar): Record size in bytes for indexed accesses, in
            [0, 16383] on CDNA3/CDNA4. Defaults to 0.

    Returns:
        buffer_descriptor: the descriptor.
    """
    _check(base.type.is_ptr() and not base.type.is_block(), lambda: f"base must be a scalar pointer, got {base.type}")
    num_records = _buffer_scalar_operand(num_records, "num_records", _semantic)
    stride = _buffer_scalar_operand(stride, "stride", _semantic)
    handle = _semantic.builder.create_make_buffer_desc(base.handle, num_records, stride)
    return buffer_descriptor(handle, buffer_descriptor_type(base.type.element_ty))


@dataclass(frozen=True)
class CachePolicy:
    """
    Cache hints for gfx9 (CDNA3/CDNA4) buffer loads and stores.

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


def _buffer_cache_policy(cache, cache_policy, scope, is_load, _semantic):
    """Return the `cachePolicy` attribute and the optional scope of a buffer load/store."""
    cache = _unwrap_if_constexpr(cache)
    cache_policy = _unwrap_if_constexpr(cache_policy)
    scope = _unwrap_if_constexpr(scope)
    if cache:
        _check(cache_policy is None, lambda: "cache and cache_policy are mutually exclusive")
        # The legacy cache modifiers imply an architecture-specific scope.
        _check(scope is None, lambda: "cache and scope are mutually exclusive")
        # Validate the modifier for the direction of the access.
        if is_load:
            _semantic._str_to_load_cache_modifier(cache)
        else:
            _semantic._str_to_store_cache_modifier(cache)
        policy = _semantic.builder.get_cache_policy(cache.removeprefix("."), "evict_normal")
    elif cache_policy is not None:
        _check(isinstance(cache_policy, CachePolicy),
               lambda: f"cache_policy must be an amd.cdna3/cdna4 CachePolicy, got {cache_policy!r}")
        policy = cache_policy._to_ir(_semantic.builder)
    else:
        policy = _semantic.builder.get_cache_policy("none", "evict_normal")
    if scope is not None:
        _check(scope in ("cta", "gpu", "sys"), lambda: f"scope must be 'cta', 'gpu' or 'sys', got {scope!r}")
        scope = _semantic._str_to_scope(scope)
    return policy, scope


def _buffer_scalar_operand(value, name, _semantic):
    """Convert an optional scalar integer operand to an i32 handle."""
    value = _unwrap_if_constexpr(value)
    if value is None:
        return ir.value()
    value = _semantic.to_tensor(value)
    _check(not value.type.is_block(), lambda: f"{name} must be a scalar, got {value.type}")
    _check(value.dtype.is_int() and value.dtype.primitive_bitwidth <= 32,
           lambda: f"{name} must be an integer of at most 32 bits, got {value.dtype}")
    return _semantic.cast(value, ttgl.int32).handle


def _buffer_element_type(ptr):
    """The element type accessed through a buffer op's scalar pointer or descriptor."""
    if isinstance(ptr, buffer_descriptor):
        return ptr.dtype
    return ptr.type.scalar.element_ty


def _buffer_indices(indices, offsets, ptr, _semantic):
    """Broadcast optional record indices to the shape of the offsets."""
    indices = _unwrap_if_constexpr(indices)
    if indices is None:
        return offsets, ir.value()
    _check(isinstance(ptr, buffer_descriptor), lambda: "indices require a buffer descriptor as ptr")
    indices = _semantic.to_tensor(indices)
    _check(indices.dtype.is_int32() or indices.dtype.is_uint32(),
           lambda: f"indices element type must be int32 or uint32, got {indices.dtype}")
    offsets, indices = _semantic.broadcast_impl_value(offsets, indices)
    return offsets, indices.handle


def _verify_buffer_ops(ptr, offsets, mask=None, other=None, allow_descriptor=False):
    if allow_descriptor and isinstance(ptr, buffer_descriptor):
        pass
    else:
        assert ptr.type.is_ptr(), "ptr must be a scalar pointer type"

    assert isinstance(offsets.type, ttgl.distributed_type), "expected offsets type to be a distributed_type"
    assert offsets.dtype.is_int32() or offsets.dtype.is_uint32(), "offsets element type must be int32 or uint32"

    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)
    if other is not None:
        assert mask is not None, "when other is not None, mask should not be None"


def _verify_element_type_and_dispatch_op(op, elem_type, arch):
    supported_types = [
        ttgl.float16, ttgl.float32, ttgl.bfloat16, ttgl.float64, ttgl.int32, ttgl.int64, ttgl.uint32, ttgl.uint64
    ]
    assert elem_type in supported_types, f"{elem_type} is not supported in buffer atomic on {arch}."

    if op in ['and', 'or', 'xor', 'xchg']:
        assert elem_type in [ttgl.int32, ttgl.int64], f"{op} with {elem_type} is not supported on CDNA3 or CDNA4"
        return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]

    if op in ['max', 'min']:
        if elem_type in [ttgl.int32, ttgl.int64, ttgl.float64]:
            op = 's' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        elif elem_type in [ttgl.uint32, ttgl.uint64]:
            op = 'u' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        else:
            raise ValueError(f"{op} with {elem_type} is not supported on CDNA3 and CDNA4")

    if op == 'add':
        if elem_type in [ttgl.uint32, ttgl.uint64]:
            op = 'i' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        elif elem_type in [ttgl.float16, ttgl.float32, ttgl.float64]:
            op = 'f' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        elif elem_type is ttgl.bfloat16:
            assert arch == "cdna4", "Buffer atomic fadd with bf16 is only supported on CDNA4 for now."
            op = 'f' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        else:
            raise ValueError(f"{op} with {elem_type} is not supported on CDNA3 and CDNA4")

    raise ValueError(f"Unknown {op} on CDNA3 or CDNA4")


def _buffer_atomic_rmw_impl(op, ptr, offsets, value, arch, mask, sem, scope, _semantic):
    _verify_buffer_ops(ptr, offsets, mask)

    op = _verify_element_type_and_dispatch_op(op, ptr.type.scalar.element_ty, arch)

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
        mask = _semantic.cast(mask, ttgl.int1)
        _, mask = _semantic.broadcast_impl_value(offsets, mask)
    mask = mask.handle if mask is not None else ir.value()

    value = _unwrap_if_constexpr(value)
    value = _semantic.to_tensor(value)
    _, value = _semantic.broadcast_impl_value(offsets, value)

    sem = _semantic._str_to_sem(sem)
    scope = _semantic._str_to_scope(scope)
    return _semantic.tensor(
        _semantic.builder.create_buffer_atomic_rmw(op, ptr.handle, offsets.handle, value.handle, sem, scope, mask),
        value.type)


_BUFFER_OFFSET_ARGS_DOC = """
        soffset (int32 scalar, optional): Scalar offset, in elements, added to every
            element's offset. It is held in the instruction's scalar offset register, so a
            uniform advance costs no per-lane arithmetic. Defaults to None (0)."""


@builtin
def buffer_load(ptr, offsets, mask=None, other=None, cache=None, *, cache_policy=None, scope=None, soffset=None,
                indices=None, _semantic=None):
    """
    AMD buffer load from global memory via a scalar base pointer or a buffer
    descriptor and a tensor of offsets instead of a tensor of pointers. This
    operation will load data directly into registers.

    Element ``i`` is read from ``base + soffset + offsets[i]``, plus
    ``indices[i] * stride`` with ``indices``. A scalar pointer gets a default
    descriptor with a 2**31 - 2 byte bound; use :func:`make_buffer_descriptor` for
    explicit bounds or a stride. Out-of-bounds elements read zero.

    To vectorize accesses whose alignment the compiler can't prove, use
    :func:`~triton.experimental.gluon.language.multiple_of` on the base pointer,
    :func:`~triton.experimental.gluon.language.max_contiguous` on the offsets and
    :func:`~triton.experimental.gluon.language.max_constancy` on the mask.

    Args:
        ptr (pointer to scalar or buffer_descriptor): Global memory base to load from.
        offsets (tensor): Offsets tensor for the load operation, in elements.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor or scalar, optional): Tensor or scalar providing default values for masked elements. Defaults to None.
        cache (str, optional): Legacy cache modifier specifier. Excludes ``cache_policy`` and ``scope``. Defaults to ``None``.
        cache_policy (CachePolicy, optional): Cache hints (CDNA3/CDNA4). Defaults to None.
        scope (str, optional): ``"cta"``, ``"gpu"`` or ``"sys"`` (CDNA3/CDNA4). The load reads data
            visible at that scope. It is not atomic and orders no other memory operation; pair it
            with an acquire to consume a payload. To poll a flag, use
            :func:`~triton.experimental.gluon.language.atomic_load` instead, which the compiler
            never hoists. Defaults to None (no coherence beyond the wave).""" + _BUFFER_OFFSET_ARGS_DOC + """
        indices (tensor, optional): Record index of each element, for a descriptor with a
            stride. Defaults to None.
    """
    _verify_buffer_ops(ptr, offsets, mask, other, allow_descriptor=True)
    elem_ty = _buffer_element_type(ptr)

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        offsets, mask = _semantic.broadcast_impl_value(offsets, mask)

    other = _unwrap_if_constexpr(other)
    if other is not None:
        other = _semantic.to_tensor(other)
        other = _semantic.cast(other, elem_ty)
        offsets, other = _semantic.broadcast_impl_value(offsets, other)

    offsets, indices = _buffer_indices(indices, offsets, ptr, _semantic)
    other = other.handle if other is not None else ir.value()
    mask = mask.handle if mask is not None else ir.value()
    policy, scope = _buffer_cache_policy(cache, cache_policy, scope, True, _semantic)

    ret_ty = offsets.type.with_element_ty(elem_ty)
    builder = _semantic.builder
    handle = builder.create_buffer_load(ret_ty.to_ir(builder), ptr.handle, offsets.handle, mask, other, policy,
                                        _buffer_scalar_operand(soffset, "soffset", _semantic), indices, scope)
    return ttgl.tensor(handle, ret_ty)


@builtin
def buffer_store(stored_value, ptr, offsets, mask=None, cache=None, *, cache_policy=None, scope=None, soffset=None,
                 indices=None, _semantic: GluonSemantic = None):
    """
    AMD buffer store a tensor directly to global memory via a scalar base pointer
    or a buffer descriptor and a tensor of offsets instead of a tensor of pointers.

    Element ``i`` is written to ``base + soffset + offsets[i]``, plus
    ``indices[i] * stride`` with ``indices``. Out-of-bounds elements are dropped;
    see :func:`buffer_load` and :func:`make_buffer_descriptor`.

    Args:
        stored_value (tensor to be stored): The tensor to be stored to global memory.
        ptr (pointer to scalar or buffer_descriptor): Global memory base to store to.
        offsets (tensor): Offsets tensor for the store operation, in elements.
        mask (tensor, optional): Mask tensor for predicated store. Defaults to None.
        cache (str, optional): Legacy cache modifier specifier. Excludes ``cache_policy`` and ``scope``. Defaults to ``None``.
        cache_policy (CachePolicy, optional): Cache hints (CDNA3/CDNA4). Defaults to None.
        scope (str, optional): ``"cta"``, ``"gpu"`` or ``"sys"`` (CDNA3/CDNA4). The store makes its
            data available at that scope. It is not atomic and orders no other memory operation;
            follow it with a release to publish. Defaults to None.""" + _BUFFER_OFFSET_ARGS_DOC + """
        indices (tensor, optional): Record index of each element, for a descriptor with a
            stride. Defaults to None.
    """
    _verify_buffer_ops(ptr, offsets, mask, allow_descriptor=True)

    offsets_shape = offsets.shape
    mask = _unwrap_if_constexpr(mask)
    if mask is None:
        offsets, stored_value = _semantic.broadcast_tensors(offsets, stored_value)
    else:
        offsets, stored_value, mask = _semantic.broadcast_tensors(offsets, stored_value, mask)
    if offsets_shape != offsets.shape:
        raise ValueError(f"Expected `offsets` argument to have shape {offsets.shape} but got {offsets_shape}")

    offsets, indices = _buffer_indices(indices, offsets, ptr, _semantic)
    mask = mask.handle if mask is not None else ir.value()
    policy, scope = _buffer_cache_policy(cache, cache_policy, scope, False, _semantic)

    _semantic.builder.create_buffer_store(stored_value.handle, ptr.handle, offsets.handle, mask, policy,
                                          _buffer_scalar_operand(soffset, "soffset", _semantic), indices, scope)


# Attribute that carries `cd_regclass` on the dot op; read by the MFMA lowering.
_CD_REGCLASS_ATTR = "amdg.cd_regclass"


def _check_cd_regclass(cd_regclass, acc):
    """Validate `cd_regclass` ("a" = AGPRs, "v" = VGPRs, None) for an MFMA on `acc`, and return it unwrapped."""
    cd_regclass = _unwrap_if_constexpr(cd_regclass)
    if cd_regclass is not None:
        _check(cd_regclass in ("a", "v"), lambda: f"cd_regclass must be None, 'a' or 'v', got {cd_regclass!r}")
        _check(isinstance(acc.type.layout, AMDMFMALayout),
               lambda: f"cd_regclass requires an accumulator with AMDMFMALayout, got {acc.type.layout}")
    return cd_regclass


def _set_cd_regclass(handle, cd_regclass, semantic):
    """Record a checked `cd_regclass` on the dot op that defines `handle`."""
    if cd_regclass is not None:
        handle.set_attr(_CD_REGCLASS_ATTR, semantic.builder.get_string_attr(cd_regclass))


@builtin
def mfma(a, b, acc, cd_regclass=None, _semantic: GluonSemantic = None):
    """
    Computes matrix multiplication ``a * b + acc`` using AMD native matrix core units.

    Args:
        a (tensor): The first operand of mfma.
        b (tensor): The second operand of mfma.
        acc (tensor): The accumulator tensor.
        cd_regclass (str, optional): Experimental. Register class for the accumulator input (C)
            and result (D) of the MFMA instructions: ``"a"`` for AGPRs or ``"v"`` for VGPRs.
            Requires an ``AMDMFMALayout`` accumulator. ``None`` (default) leaves the choice to
            the compiler.
    """
    assert acc is not None, "acc is required"
    ret_type = acc.type
    acc = ttgl._unwrap_if_constexpr(acc)

    cd_regclass = _check_cd_regclass(cd_regclass, acc)
    handle = _semantic.dot(a, b, acc, input_precision=knobs.language.fp32_default, max_num_imprecise_acc=None,
                           out_dtype=acc.dtype).handle
    _set_cd_regclass(handle, cd_regclass, _semantic)
    return ttgl.tensor(handle, ret_type)


@builtin
def scaled_upcast(src, scale, elem_type, axis=None, _semantic=None):
    """
    Upcast an fp4 or fp8 tensor and fold raw E8M0 scale payload into the
    CDNA3 scaled-upcast op.

    CDNA3 lowers this through the software-emulated scaled-upcast path; it
    does not use native hardware scaled-upcast instructions.

    The ``scale`` tensor must use raw E8M0 payload in ``int8`` or ``uint8``, and must
    already have the expanded output shape and scaled-upcast result layout.
    For fp4 inputs, that is the canonical unpacked layout implied by ``src``
    and ``axis``. ``elem_type`` must be ``fp16`` or ``bf16``.
    """
    axis = _unwrap_if_constexpr(axis)
    elem_type = _unwrap_if_constexpr(elem_type)
    assert scale.dtype in (int8, uint8), \
        f"Expected scale to use raw E8M0 payload in int8/uint8 but got {scale.dtype}"
    return _scaled_upcast(src, scale, elem_type, axis, _semantic)


"""
AMD Buffer Atomic RMW operations.
The supported operations are max, min, add, and, or, xor, xchg.
Similar to normal atomic ops: it loads data at ptr plus offsets, do `op` with `value`, and store result to `ptr` plus `offsets` with
the specified memory semantics and scope.

Buffer atomics access global memory via a scalar base pointer and a tensor of offsets instead of a tensor of pointers.
Similar to other buffer ops, the `mask` is a boolean vector that determines if a given element should be processed with
the atomic RMW op. Elements with `mask[i] == 0` are dropped (i.e., the atomic is not executed).

Buffer Atomic RMW ops return the pre-op value in the global memory.

Args:
    ptr (pointer to scalar): Global memory scalar base pointer to load from.
    offsets (tensor): Offsets tensor for the load operation.
    value (tensor): Another operand of `op`.
    mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
    sem (str, optional): Memory Semantic Descriptor. Default is None which means acq_rel memory semantic.
    scope (str, optional): Memory Sync Scope for atomic accesses. Default is None and it will be mapped to `gpu`, which is called `agent` for AMDGPU. Please ref https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942 for details.
"""


@builtin
def buffer_atomic_max(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):
    return _buffer_atomic_rmw_impl('max', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_min(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('min', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_add(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('add', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_and(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('and', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_or(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('or', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xor(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xor', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xchg(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xchg', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)
