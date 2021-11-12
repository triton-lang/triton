import torch
import triton
from triton.code_gen import JITFunction
import triton.language as tl
import os
import shutil

tmpdir = ".tmp"

@triton.jit
def function_1(i):
    i = i + 1
    i = function_2(i)
    return i


@triton.jit
def function_2(i):
    i = i + 1
    return i

@triton.jit
def kernel(X, i, BLOCK: tl.constexpr):
    i = i + 1
    i = function_1(i)
    tl.store(X, i)


def apply_src_change(target, old, new):
    delattr(kernel.fn, 'hash')
    delattr(function_1.fn, 'hash')
    delattr(function_2.fn, 'hash')
    function_1.src = function_1.src.replace(old, new)
    target.src = target.src.replace(old, new)
    ret = target.cache_key
    target.src = target.src.replace(new, old)
    return ret

def test_nochange():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 1')
    assert baseline == updated

def test_toplevel_change():
    baseline = kernel.cache_key
    updated = apply_src_change(kernel, 'i + 1', 'i + 2')
    assert baseline != updated

def test_nested1_change():
    baseline = kernel.cache_key
    updated = apply_src_change(function_1, 'i + 1', 'i + 2')
    assert baseline != updated

def test_reuse():
    counter = 0
    def inc_counter(key, binary):
        nonlocal counter
        counter += 1
    os.environ["TRITON_CACHE_DIR"] = tmpdir
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    JITFunction.cache_hook = inc_counter
    x = torch.empty(1, dtype=torch.int32, device='cuda')
    for i in range(10):
        kernel[(1,)](x, 43, BLOCK=1024)
    assert counter == 1
