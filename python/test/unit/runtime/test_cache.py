import multiprocessing
import os
import re
import shutil
from collections import namedtuple

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
        shutil.rmtree(tmpdir)


def test_reuse():
    counter = 0

    def inc_counter(*args, **kwargs):
        nonlocal counter
        counter += 1
    JITFunction.cache_hook = inc_counter
    reset_tmp_dir()
    x = torch.empty(1, dtype=torch.int32, device='cuda')
    for i in range(10):
        kernel[(1,)](x, 1, BLOCK=1024)
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
    target = {'enable': 3, 'disable': 1}[mode]
    for i in [1, 2, 4, 8, 16, 32]:
        function[(1,)](x, i, BLOCK=512)
    assert counter == target


@pytest.mark.parametrize("value, value_type", [
    (-1, 'i32'), (0, 'i32'), (1, 'i32'), (-2**31, 'i32'), (2**31 - 1, 'i32'),
    (2**32, 'i64'), (2**63 - 1, 'i64'), (-2**63, 'i64'),
    (2**31, 'u32'), (2**32 - 1, 'u32'), (2**63, 'u64'), (2**64 - 1, 'u64')
])
def test_value_specialization(value: int, value_type: str, device='cuda') -> None:

    @triton.jit
    def kernel(VALUE, X):
        pass

    cache_str = None

    def get_cache_str(*args, **kwargs):
        nonlocal cache_str
        cache_str = kwargs["repr"]
    triton.JITFunction.cache_hook = get_cache_str
    reset_tmp_dir()
    x = torch.tensor([3.14159], device='cuda')
    kernel[(1, )](value, x)
    triton.JITFunction.cache_hook = None

    cache_str_match = re.match(r".*VALUE: (\w+).*", cache_str)
    spec_type = None if cache_str_match is None else cache_str_match.group(1)
    assert spec_type == value_type


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
        tl.store(o + idx,
                 tl.load(a + idx) + tl.load(b + idx))

    args = [
        torch.randn(32, dtype=torch.float32, device="cuda"),
        torch.randn(32, dtype=torch.float32, device="cuda"),
        torch.randn(32, dtype=torch.float32, device="cuda"),
        32,
    ]
    assert len(kernel_add.cache) == 0
    kernel_add.warmup(torch.float32, torch.float32, torch.float32, 32, grid=(1,))
    assert len(kernel_add.cache) == 1
    kernel_add.warmup(*args, grid=(1,))
    assert len(kernel_add.cache) == 1
    kernel_add.warmup(*args, grid=(1,))
    assert len(kernel_add.cache) == 1


def test_compile_in_subproc() -> None:
    @triton.jit
    def kernel_sub(a, b, o, N: tl.constexpr):
        idx = tl.arange(0, N)
        tl.store(o + idx,
                 tl.load(a + idx) - tl.load(b + idx) * 777)

    major, minor = torch.cuda.get_device_capability(0)
    cc = major * 10 + minor
    config = namedtuple("instance_descriptor", [
        "divisible_by_16", "equal_to_1"])(
        tuple(range(4)),
        ())

    proc = multiprocessing.Process(
        target=triton.compile,
        kwargs=dict(
            fn=kernel_sub,
            signature={0: "*fp32", 1: "*fp32", 2: "*fp32"},
            device=0,
            constants={3: 32},
            configs=[config],
            warm_cache_only=True,
            cc=cc,
        ))
    proc.start()
    proc.join()
    assert proc.exitcode == 0


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
