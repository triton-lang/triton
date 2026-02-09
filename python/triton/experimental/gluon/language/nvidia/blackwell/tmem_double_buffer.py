"""
TMEM Double Buffering for Blackwell (SM100+).
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple
from triton.experimental.gluon.language._core import builtin, constexpr, base_type, base_value

if TYPE_CHECKING:
    from . import tensor_memory_descriptor, TensorMemoryLayout

__all__ = ["TMEMDoubleBuffer", "TMEMBufferPair", "allocate_double_buffer", "allocate_double_buffer_pair"]


class tmem_double_buffer_type(base_type):

    def __init__(self, inner_type, block_m, block_n):
        self.inner_type, self.block_m, self.block_n = inner_type, block_m, block_n

    def to_ir(self, builder):
        return self.inner_type.to_ir(builder)

    def _unflatten_ir(self, handles, cursor):
        inner, cursor = self.inner_type._unflatten_ir(handles, cursor)
        return TMEMDoubleBuffer(inner, self.block_m, self.block_n), cursor

    def _flatten_ir_types(self, builder, out):
        self.inner_type._flatten_ir_types(builder, out)

    def __str__(self):
        return f"tmem_double_buffer<{self.block_m}x{self.block_n}>"

    def __eq__(self, other):
        return type(self) is type(other) and self.block_m == other.block_m and self.block_n == other.block_n

    def mangle(self):
        return f"TMDB{self.block_m}x{self.block_n}"


class TMEMDoubleBuffer(base_value):

    def __init__(self, desc, block_m, block_n):
        self._desc, self._block_m, self._block_n = desc, block_m, block_n
        self.type = tmem_double_buffer_type(desc.type, block_m, block_n)
        self.handle = desc.handle

    def _flatten_ir(self, handles):
        self._desc._flatten_ir(handles)

    @property
    def block_m(self):
        return self._block_m

    @property
    def block_n(self):
        return self._block_n

    @builtin
    def get_buffer(self, phase, _semantic=None):
        if isinstance(phase, constexpr):
            phase_val = phase.value
        elif isinstance(phase, int):
            phase_val = phase
        else:
            raise NotImplementedError("Use allocate_double_buffer_pair() for runtime phase.")
        return self._desc.slice(phase_val * self._block_n, self._block_n, _semantic=_semantic)


@builtin
def allocate_double_buffer(element_ty, block_m, block_n, layout, _semantic=None):
    from triton.experimental.gluon.language._core import _unwrap_if_constexpr
    from . import allocate_tensor_memory, TensorMemoryLayout
    element_ty, block_m, block_n, layout = [_unwrap_if_constexpr(x) for x in [element_ty, block_m, block_n, layout]]
    double_layout = TensorMemoryLayout(block=(layout.block[0], layout.block[1] * 2), col_stride=layout.col_stride,
                                       cta_split_num=layout.cta_split_num, two_ctas=layout.two_ctas)
    desc = allocate_tensor_memory(element_ty, [block_m, block_n * 2], double_layout, value=None, _semantic=_semantic)
    return TMEMDoubleBuffer(desc, block_m, block_n)


class tmem_buffer_pair_type(base_type):

    def __init__(self, inner_type, block_m, block_n):
        self.inner_type, self.block_m, self.block_n = inner_type, block_m, block_n

    def to_ir(self, builder):
        return self.inner_type.to_ir(builder)

    def _unflatten_ir(self, handles, cursor):
        buf0, cursor = self.inner_type._unflatten_ir(handles, cursor)
        buf1, cursor = self.inner_type._unflatten_ir(handles, cursor)
        return TMEMBufferPair(buf0, buf1, self.block_m, self.block_n), cursor

    def _flatten_ir_types(self, builder, out):
        self.inner_type._flatten_ir_types(builder, out)
        self.inner_type._flatten_ir_types(builder, out)

    def __str__(self):
        return f"tmem_buffer_pair<{self.block_m}x{self.block_n}>"

    def __eq__(self, other):
        return type(self) is type(other) and self.block_m == other.block_m and self.block_n == other.block_n

    def mangle(self):
        return f"TMBP{self.block_m}x{self.block_n}"


class TMEMBufferPair(base_value):

    def __init__(self, buf0, buf1, block_m, block_n):
        self._buf0, self._buf1, self._block_m, self._block_n = buf0, buf1, block_m, block_n
        self.type = tmem_buffer_pair_type(buf0.type, block_m, block_n)
        self.handle = buf0.handle

    def _flatten_ir(self, handles):
        self._buf0._flatten_ir(handles)
        self._buf1._flatten_ir(handles)

    @property
    def block_m(self):
        return self._block_m

    @property
    def block_n(self):
        return self._block_n

    @builtin
    def index(self, phase, _semantic=None):
        if isinstance(phase, constexpr):
            return self._buf0 if phase.value == 0 else self._buf1
        if isinstance(phase, int):
            return self._buf0 if phase == 0 else self._buf1
        raise NotImplementedError("Runtime tensor phase not yet supported.")


@builtin
def allocate_double_buffer_pair(element_ty, block_m, block_n, layout, _semantic=None):
    from triton.experimental.gluon.language._core import _unwrap_if_constexpr
    from . import allocate_tensor_memory
    element_ty, block_m, block_n, layout = [_unwrap_if_constexpr(x) for x in [element_ty, block_m, block_n, layout]]
    buf0 = allocate_tensor_memory(element_ty, [block_m, block_n], layout, value=None, _semantic=_semantic)
    buf1 = allocate_tensor_memory(element_ty, [block_m, block_n], layout, value=None, _semantic=_semantic)
    return TMEMBufferPair(buf0, buf1, block_m, block_n)
