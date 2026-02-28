import torch

import triton
import triton.language as tl
from triton._C.libtriton import ir, passes
from triton.language.core import builtin
from typing import TypeVar, Type
from functools import wraps
import builtins
import os
from triton import knobs
import pathlib
import hashlib
import importlib
import inspect
import sys
import textwrap

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

def builtin(fn: T) -> T:
    """Mark a function as a builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_semantic" not in kwargs or kwargs["_semantic"] is None:
            raise ValueError("Did you forget to add @triton.jit ? "
                             "(`_semantic` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    setattr(wrapper, TRITON_BUILTIN, True)

    return wrapper



DEVICE = triton.runtime.driver.active.get_active_torch_device()

def get_key():
    return pathlib.Path(__file__).read_text()


def get_hash():
    return hashlib.sha256(get_key().encode('utf-8')).hexdigest()

def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
    if all(arg is None for arg in (stages, options, language, capability)):
        return get_key(), get_hash()
    module_name = 'dynamic_module'
    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    stage_src = textwrap.dedent(inspect.getsource(self.make_ttir))
    stage_src = 'from triton._C.libtriton import ir, passes, llvm, amd, nvidia\n' + stage_src
    # Inject plugin pass right after loop unroll in the dynamically loaded stage source
    stage_src = stage_src.replace(
        "pm = ir.pass_manager(mod.context)",
        "pm = ir.pass_manager(mod.context)\n    passes.plugin.plugingpu_farith_conversion(pm, opt.num_warps, 32, opt.num_ctas)\n"
    )
    # print(stage_src)
    exec(stage_src, module.__dict__)
    make_lambda = lambda f: lambda src, metadata: f(src, metadata, options, capability)
    stages["ttir"] = make_lambda(module.make_ttir)
    return get_key(), get_hash()

@builtin
def custom_op(x, sanitize_overflow: tl.constexpr = True, _semantic=None):
    x = _unwrap_if_constexpr(x)
    builder = _semantic.getBuilder()
    return tl.tensor(builder.create_custom_op(x.handle), x.type)
@triton.jit
def add_kernel(x_ptr,
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

if __name__ == "__main__":
    size = 8
    x = torch.zeros(size, device=DEVICE, dtype=torch.float32)
    # y = torch.ones(size, device=DEVICE, dtype=torch.float32)
    # output_torch = x + y
    output_triton = torch.empty_like(x)
    n_elements = output_triton.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    knobs.runtime.add_stages_inspection_hook = inspect_stages_hook
    h = add_kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=32)
    print(output_triton)
    print(h.asm["ttir"])

    # print(f'The maximum difference between torch and custom triton op is '
    #       f'{torch.max(torch.abs(output_torch - output_triton))}')
