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

triton.language.__all__.append("custom_op")
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


@builtin
def custom_op(x, sanitize_overflow: tl.constexpr = True, _semantic=None):
    x = _unwrap_if_constexpr(x)
    builder = _semantic.builder
    arg_handles = []
    arg_handles.extend(flatten_values_to_ir([x]))
    return tl.tensor(builder.create_custom_op(arg_handles), x.type)


@triton.jit
def add_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = custom_op(x)
    tl.store(output_ptr + offsets, output, mask=mask)


def test_custom_ops(tmp_path: pathlib.Path):
    if os.environ.get('TRITON_EXT_ENABLED', '0') == '0':
        return
    size = 8
    x = torch.zeros(size, device=DEVICE, dtype=torch.float32)
    output_triton = torch.empty_like(x)
    n_elements = output_triton.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    h = add_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=32)

    src = h.asm["source"]
    assert "arith.addf" in src
