import gc
import tracemalloc
import pytest
import pathlib
import os

import torch
import triton
import triton.language as tl
from triton._internal_testing import is_cuda, is_hip


def test_metadata() -> None:

    used_hook = False

    def _launch_metadata(grid, kernel, args):
        ret = dict()
        ret["grid"] = grid
        ret["value"] = args["x"]
        return ret

    def hook(launch_metadata):
        nonlocal used_hook
        metadata = launch_metadata.get()
        assert metadata["grid"] == (1, 3, 2)
        assert metadata["value"] == 6
        used_hook = True

    @triton.jit(launch_metadata=_launch_metadata)
    def kernel(x):
        pass

    # launch kernel
    triton.knobs.runtime.launch_enter_hook.add(hook)
    kernel[(1, 3, 2)](6)
    triton.knobs.runtime.launch_enter_hook.remove(hook)
    assert used_hook


def test_memory_leak(device) -> None:

    @triton.jit
    def kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
        xnumel = 10
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tl.store(out_ptr0 + (x0 + tl.zeros([XBLOCK], tl.int32)), tmp0, xmask)

    tracemalloc.start()
    try:
        inp = torch.randn(10, device=device)
        out = torch.randn(10, device=device)
        kernel[(10, )](inp, out, 10, XBLOCK=16)
        gc.collect()
        begin, _ = tracemalloc.get_traced_memory()
        for _ in range(100):
            kernel[(10, )](inp, out, 10, XBLOCK=16)
        gc.collect()
        end, _ = tracemalloc.get_traced_memory()
        assert end - begin < 30000
    finally:
        tracemalloc.stop()


def test_load_hook() -> None:

    used_start_hook = False
    start_hash = None

    def hook_start(module, function, name, metadata_group, hash):
        nonlocal used_start_hook
        nonlocal start_hash
        start_hash = hash
        used_start_hook = True

    used_end_hook = False
    end_hash = None

    def hook_end(module, function, name, metadata_group, hash):
        nonlocal used_end_hook
        nonlocal end_hash
        end_hash = hash
        used_end_hook = True

    @triton.jit
    def kernel(x):
        pass

    # launch kernel
    triton.knobs.runtime.kernel_load_start_hook.add(hook_start)
    triton.knobs.runtime.kernel_load_end_hook.add(hook_end)
    kernel[(1, 3, 2)](6)
    assert used_start_hook
    assert used_end_hook
    assert start_hash == end_hash
    triton.knobs.runtime.kernel_load_start_hook.remove(hook_start)
    triton.knobs.runtime.kernel_load_end_hook.remove(hook_end)


def test_multiple_hooks() -> None:

    start0 = False
    end0 = False
    start1 = False
    end1 = False

    def hook_start0(module, function, name, metadata_group, hash):
        nonlocal start0
        start0 = True

    def hook_end0(module, function, name, metadata_group, hash):
        nonlocal end0
        end0 = True

    def hook_start1(module, function, name, metadata_group, hash):
        nonlocal start1
        start1 = True

    def hook_end1(module, function, name, metadata_group, hash):
        nonlocal end1
        end1 = True

    triton.knobs.runtime.kernel_load_start_hook.add(hook_start0)
    triton.knobs.runtime.kernel_load_end_hook.add(hook_end0)
    triton.knobs.runtime.kernel_load_start_hook.add(hook_start1)
    triton.knobs.runtime.kernel_load_end_hook.add(hook_end1)

    @triton.jit
    def kernel(x):
        pass

    kernel[(1, )](6)

    assert start0
    assert end0
    assert start1
    assert end1

    triton.knobs.runtime.kernel_load_start_hook.remove(hook_start0)
    triton.knobs.runtime.kernel_load_end_hook.remove(hook_end0)
    triton.knobs.runtime.kernel_load_start_hook.remove(hook_start1)
    triton.knobs.runtime.kernel_load_end_hook.remove(hook_end1)


@pytest.mark.parametrize("options", [
    {"num_warps": 1},
    {"enable_fp_fusion": False},
    {"extern_libs": {}},
])
def test_launch_with_options(options) -> None:
    if "extern_libs" in options:
        # copied from tutorials/07-extern-functions.py
        current_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
        if is_cuda():
            libdir = current_dir.parent.parent.parent.parent / 'third_party/nvidia/backend/lib'
            options["extern_libs"] = {"libdevice": str(libdir / 'libdevice.10.bc')}
        elif is_hip():
            libdir = current_dir.parent.parent.parent.parent / 'third_party/amd/backend/lib'
            options["extern_libs"] = {"ocml": str(libdir / 'ocml.bc'), "ockl": str(libdir / 'ockl.bc')}

    compile_info = {}
    counter = 0

    def compile_info_hook(key, repr, fn, compile, is_manual_warmup, already_compiled):
        nonlocal compile_info
        compile_info = compile

    def cache_hook(*args, **kwargs):
        nonlocal counter
        counter += 1

    @triton.jit
    def kernel(x):
        pass

    triton.knobs.runtime.jit_post_compile_hook = compile_info_hook
    triton.knobs.runtime.jit_cache_hook = cache_hook

    # run first without options
    kernel[(1, 1, 1)](6)
    assert counter == 1

    # run with options, should lead to new compilation
    kernel[(1, 1, 1)](6, **options)
    assert counter == 2

    # run a second time for testing kernel-cache look-up
    kernel[(1, 1, 1)](6, **options)
    assert counter == 2

    # check the options are passed on to compile_info correctly
    option_key, option_val = next(iter(options.items()))
    if option_key == "extern_libs":
        # HIPOptions overwrite the extern_libs option, so we skip the test
        # passing and specializing options still is tested
        if not is_hip():
            assert compile_info[option_key] == tuple(option_val.items())
    else:
        assert compile_info[option_key] == option_val

    triton.knobs.runtime.jit_post_compile_hook = None
    triton.knobs.runtime.jit_cache_hook = None
