import torch

import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.language.core import builtin
from typing import TypeVar, Type
import builtins
import os
import pathlib
from triton.compiler.code_generator import flatten_values_to_ir

T = TypeVar('T')
TensorTy = TypeVar('TensorTy')

triton.language.__all__.append("py_custom_op")
tensor: Type[TensorTy] = tl.tensor
builder: ir.builder

TRITON_BUILTIN = "__triton_builtin__"


def _unwrap_if_constexpr(o):
    if isinstance(o, list):
        return [_unwrap_if_constexpr(x) for x in o]
    if isinstance(o, builtins.tuple):
        return builtins.tuple(_unwrap_if_constexpr(x) for x in o)
    if isinstance(o, tuple):
        return tuple(_unwrap_if_constexpr(x) for x in o)
    return o.value if isinstance(o, tl.constexpr) else o


DEVICE = triton.runtime.driver.active.get_active_torch_device()


# A plugin op registered with AddOpWithPyArgCallback: it accepts an arbitrary
# number of positional `mlir.Value` operands plus a `mode` keyword argument
# (a Python str), and is exposed to Python as `builder.create_py_custom_op`.
@builtin
def py_custom_op(a, b, mode: tl.constexpr = "add", _semantic=None):
    a = _unwrap_if_constexpr(a)
    b = _unwrap_if_constexpr(b)
    builder = _semantic.builder
    arg_handles = flatten_values_to_ir([a, b])
    result = builder.create_py_custom_op(*arg_handles, mode=mode)
    return tl.tensor(result, a.type)


@triton.jit
def py_arg_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    MODE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = py_custom_op(x, x, mode=MODE)
    tl.store(output_ptr + offsets, output, mask=mask)


def test_py_arg_ops(tmp_path: pathlib.Path):
    if os.environ.get('TRITON_EXT_ENABLED', '0') == '0':
        return
    size = 8
    x = torch.ones(size, device=DEVICE, dtype=torch.float32)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    out_add = torch.empty_like(x)
    h = py_arg_kernel[grid](x, out_add, n_elements, BLOCK_SIZE=32, MODE="add")
    assert "arith.addf" in h.asm["source"]

    out_mul = torch.empty_like(x)
    h = py_arg_kernel[grid](x, out_mul, n_elements, BLOCK_SIZE=32, MODE="mul")
    assert "arith.mulf" in h.asm["source"]
