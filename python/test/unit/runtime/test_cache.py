import os
import shutil

import pytest
import torch

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction

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


@triton.jit(do_not_specialize=["i"])
def kernel_nospec(X, i, BLOCK: tl.constexpr):
    i = i + 1
    i = function_1(i)
    tl.store(X, i)


def apply_src_change(target, old, new):
    kernel.hash = None
    function_1.hash = None
    function_2.hash = None
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


def reset_tmp_dir():
    os.environ["TRITON_CACHE_DIR"] = tmpdir
    if os.path.exists(tmpdir):
        # https://stackoverflow.com/questions/303200/how-do-i-remove-delete-a-folder-that-is-not-empty
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_reuse():
    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    JITFunction.cache_hook = inc_counter
    reset_tmp_dir()
    x = torch.empty(1, dtype=torch.int32, device='cuda')
    for i in range(10):
        kernel[(1, )](x, 1, BLOCK=1024)
    assert counter == 1


@pytest.mark.parametrize('mode', ['enable', 'disable'])
def test_specialize(mode):
    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1

    JITFunction.cache_hook = inc_counter
    reset_tmp_dir()
    x = torch.empty(1, dtype=torch.int32, device='cuda')
    function = {'enable': kernel, 'disable': kernel_nospec}[mode]
    target = {'enable': 4, 'disable': 1}[mode]
    for i in [1, 2, 4, 8, 16, 32]:
        function[(1, )](x, i, BLOCK=512)
    assert counter == target


def test_annotation():

    @triton.jit
    def kernel(X, i: tl.int32):
        tl.store(X, i)

    x = torch.empty(1, dtype=torch.int32, device='cuda')

    device = torch.cuda.current_device()
    kernel[(1, )](x, 1)
    kernel[(1, )](x, 8)
    kernel[(1, )](x, 16)
    kernel[(1, )](x, 17)
    assert len(kernel.cache[device]) == 4


def test_constexpr_not_callable() -> None:

    @triton.jit
    def kernel(X, c: tl.constexpr):
        tl.store(X, 2)

    x = torch.empty(1, dtype=torch.int32, device='cuda')
    error = False
    try:
        kernel[(1, )](x, c="str")
    except BaseException:
        error = True
    assert error is False
    # try and catch
    try:
        kernel[(1, )](x, c=tl.abs)
    except BaseException:
        error = True
    assert error is True


def test_jit_warmup_cache() -> None:

    @triton.jit
    def kernel_add(a, b, o, N: tl.constexpr):
        idx = tl.arange(0, N)
        tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))

    args = [
        torch.randn(32, dtype=torch.float32, device="cuda"),
        torch.randn(32, dtype=torch.float32, device="cuda"),
        torch.randn(32, dtype=torch.float32, device="cuda"),
        32,
    ]
    device = torch.cuda.current_device()
    assert len(kernel_add.cache[device]) == 0
    kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add.cache[device]) == 1
    kernel_add.warmup(*args, grid=(1, ))
    assert len(kernel_add.cache[device]) == 1
    kernel_add.warmup(*args, grid=(1, ))
    assert len(kernel_add.cache[device]) == 1


def test_jit_debug() -> None:

    @triton.jit
    def kernel_add(a, b, o, N: tl.constexpr):
        idx = tl.arange(0, N)
        tl.device_assert(idx < 32, "idx < 32")
        tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))

    device = torch.cuda.current_device()
    assert len(kernel_add.cache[device]) == 0
    kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add.cache[device]) == 1
    kernel_add.debug = False
    kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add.cache[device]) == 2
    kernel_add.debug = True
    kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add.cache[device]) == 3
    bins = list(kernel_add.cache[device].values())
    assert bins[2].asm['ttir'] != bins[1].asm['ttir']


@triton.jit
def add_fn(a, b, o, N: tl.constexpr):
    idx = tl.arange(0, N)
    tl.store(o + idx, tl.load(a + idx) + tl.load(b + idx))


def test_jit_noinline() -> None:

    @triton.jit
    def kernel_add_device(a, b, o, N: tl.constexpr):
        add_fn(a, b, o, N)

    device = torch.cuda.current_device()
    assert len(kernel_add_device.cache[device]) == 0
    kernel_add_device.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add_device.cache[device]) == 1
    bins = list(kernel_add_device.cache[device].values())
    inline_ttir = bins[0].asm['ttir']
    add_fn.noinline = True
    add_fn.hash = None
    kernel_add_device.hash = None
    kernel_add_device.cache[device].clear()
    kernel_add_device.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1, ))
    assert len(kernel_add_device.cache[device]) == 1
    bins = list(kernel_add_device.cache[device].values())
    noinline_ttir = bins[0].asm['ttir']
    assert inline_ttir != noinline_ttir


def test_memory_leak() -> None:

    @triton.jit
    def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
        xnumel = 10
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)
