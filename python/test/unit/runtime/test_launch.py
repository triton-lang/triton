import gc
import tracemalloc
import pytest
import pathlib
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, get_ident
from types import SimpleNamespace
import numpy as np

import torch
import triton
import triton.language as tl
from triton._internal_testing import is_cuda, is_hip
from triton.compiler import compiler


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


def test_concurrent_kernel_load(monkeypatch) -> None:
    launcher_created = Event()
    release_load = Event()
    waiter_entered = Event()
    end_hook_entered = Event()
    release_end_hook = Event()
    start_hook_reentered = Event()
    module = object()
    function = object()
    launched_functions = []
    load_calls = 0

    def launcher(*args):
        launched_functions.append(args[4])

    class ObservableLock:

        def __init__(self):
            self.lock = Lock()
            self.first_owner = None

        def __enter__(self):
            self.lock.acquire()
            owner = get_ident()
            if self.first_owner is None:
                self.first_owner = owner
            elif owner != self.first_owner:
                waiter_entered.set()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.lock.release()

    def make_launcher(src, metadata):
        launcher_created.set()
        return launcher

    def load_binary(name, kernel, shared, device):
        nonlocal load_calls
        load_calls += 1
        assert release_load.wait(timeout=5)
        return module, function, 32, 0, 1024

    utils = SimpleNamespace(
        get_device_properties=lambda _: {"max_shared_mem": 1},
        load_binary=load_binary,
        unload_module=lambda _: None,
    )
    fake_driver = SimpleNamespace(
        get_current_device=lambda: 0,
        get_current_stream=lambda _: 0,
        get_current_target=lambda: SimpleNamespace(warp_size=32),
        launcher_cls=make_launcher,
        utils=utils,
    )
    monkeypatch.setattr(compiler.driver, "_active", fake_driver)
    monkeypatch.setattr(compiler, "max_shared_mem", lambda _: 1)

    compiled = object.__new__(compiler.CompiledKernel)
    compiled.src = object()
    compiled.metadata = SimpleNamespace(shared=0, num_warps=4, target=SimpleNamespace(arch=89))
    compiled.metadata_group = {}
    compiled.packed_metadata = ()
    compiled.hash = "hash"
    compiled.name = "kernel"
    compiled.kernel = b""
    compiled.module = None
    compiled._module_pid = None
    compiled.function = None
    compiled._run = None
    # Simulate a fork copying a lock held by a thread that does not survive in
    # the child. The child must replace that lock before lazy initialization.
    stale_lock = Lock()
    stale_lock.acquire()
    stale_state = SimpleNamespace(lock=stale_lock, event=Event(), owner=1, run=None, handles_ready=False)
    current_state = SimpleNamespace(lock=ObservableLock(), event=Event(), owner=None, run=None, handles_ready=False)
    compiled._init_states = {-1: stale_state}
    compiled._handle_generation = getattr(compiler, "_handle_generation", 0)
    if hasattr(compiler, "_HandleInitState"):
        monkeypatch.setattr(compiler, "_HandleInitState", lambda: current_state)

    def hook_start(module, function, name, metadata_group, hash):
        assert compiled.run is launcher
        start_hook_reentered.set()

    def hook_end(module, function, name, metadata_group, hash):
        end_hook_entered.set()
        assert compiled.module is module
        assert compiled.function is function
        compiled[(1, 1, 1)]()
        assert launched_functions[-1] is function
        assert release_end_hook.wait(timeout=5)

    triton.knobs.runtime.kernel_load_start_hook.add(hook_start)
    triton.knobs.runtime.kernel_load_end_hook.add(hook_end)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(lambda: compiled.run)
            assert launcher_created.wait(timeout=5)
            assert start_hook_reentered.wait(timeout=5)
            second = pool.submit(lambda: compiled.run)
            try:
                assert waiter_entered.wait(timeout=5)
                assert not second.done()
                release_load.set()
                assert end_hook_entered.wait(timeout=5)
                assert not second.done()
            finally:
                release_load.set()
                release_end_hook.set()
            assert first.result(timeout=5) is launcher
            assert second.result(timeout=5) is launcher
        assert load_calls == 1
        assert compiled.module is module
        assert compiled.function is function

        # A child must discard even fully published handles from its parent.
        compiled.module = object()
        compiled._module_pid = -1
        compiled.function = object()
        compiled._run = launcher
        compiled._handle_generation = -1
        assert compiled.run is launcher
        assert load_calls == 2
        assert compiled.module is module
        assert compiled.function is function
    finally:
        release_load.set()
        release_end_hook.set()
        triton.knobs.runtime.kernel_load_start_hook.remove(hook_start)
        triton.knobs.runtime.kernel_load_end_hook.remove(hook_end)
        compiled.module = None
        compiled._module_pid = None
        stale_lock.release()


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
def test_launch_with_options(options, monkeypatch) -> None:
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

    monkeypatch.setattr(triton.knobs.runtime, "jit_post_compile_hook", compile_info_hook)
    monkeypatch.setattr(triton.knobs.runtime, "jit_cache_hook", cache_hook)

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


@pytest.mark.interpreter
def test_pre_run_hooks(device):

    @triton.jit
    def add_kernel(a_ptr, n_elements: tl.constexpr):
        offsets = tl.arange(0, n_elements)
        a = tl.load(a_ptr + offsets)
        a += 2
        tl.store(a_ptr + offsets, a)

    def my_hook(*args, **kwargs):
        args[0].zero_()

    add_kernel.add_pre_run_hook(my_hook)

    n_elements = 4
    a = torch.ones(n_elements, device=device, dtype=torch.int32)
    add_kernel[(1, )](a, n_elements)
    assert torch.all(a == 2)

    a = torch.ones(n_elements, device=device, dtype=torch.int32)
    add_kernel.run(a, n_elements, grid=(1, ), warmup=False)
    assert torch.all(a == 2)


def test_interpreter_implicit_cvt_bool() -> None:
    from triton.runtime.interpreter import _implicit_cvt

    value = _implicit_cvt(True)

    assert value.dtype == tl.int1
    assert value.handle.data.dtype == np.bool_
    assert bool(value.handle.data[0]) is True
