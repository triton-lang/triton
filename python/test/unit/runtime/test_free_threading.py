import os
import subprocess
import sys
import sysconfig
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest


def _run_python(source, *, force_no_gil=True):
    env = os.environ.copy()
    if force_no_gil:
        env["PYTHON_GIL"] = "0"
    else:
        env.pop("PYTHON_GIL", None)
    return subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.skipif(sysconfig.get_config_var("Py_GIL_DISABLED") != 1, reason="requires a free-threaded Python build")
def test_free_threaded_import_keeps_gil_disabled():
    result = _run_python(
        "import sys\n"
        "assert not sys._is_gil_enabled()\n"
        "import triton\n"
        "from triton.runtime.driver import driver\n"
        "assert driver.active is not None\n"
        "assert not sys._is_gil_enabled()\n",
        force_no_gil=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sysconfig.get_config_var("Py_GIL_DISABLED") != 1, reason="requires a free-threaded Python build")
def test_native_specialize_concurrent_first_use():
    result = _run_python("import gc, threading\n"
                         "from concurrent.futures import ThreadPoolExecutor\n"
                         "from triton._C.libtriton import native_specialize_impl\n"
                         "from triton.backends.compiler import BaseBackend\n"
                         "start = threading.Barrier(17)\n"
                         "def specialize(_):\n"
                         "    start.wait(10)\n"
                         "    return native_specialize_impl(BaseBackend, 2, False, False, True)\n"
                         "def collect():\n"
                         "    start.wait(10)\n"
                         "    for _ in range(8):\n"
                         "        gc.collect()\n"
                         "with ThreadPoolExecutor(17) as pool:\n"
                         "    results = [pool.submit(specialize, worker) for worker in range(16)]\n"
                         "    collector = pool.submit(collect)\n"
                         "    assert [future.result(10) for future in results] == [('i32', None)] * 16\n"
                         "    collector.result(10)\n")
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sysconfig.get_config_var("Py_GIL_DISABLED") != 1, reason="requires a free-threaded Python build")
def test_free_threaded_concurrent_dispatch_distinct_streams_and_devices():
    import torch
    import triton
    import triton.language as tl

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    assert not sys._is_gil_enabled()

    @triton.jit
    def plain(out_ptr, in_ptr, n_elements: tl.constexpr, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        value = tl.load(in_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, value + 1, mask=mask)

    def do_bench(call, quantiles):
        call()
        return (0.0, 0.0, 0.0)

    @triton.autotune(configs=[triton.Config({"BLOCK": 128}),
                              triton.Config({"BLOCK": 256})], key=["n_elements"], do_bench=do_bench)
    @triton.jit
    def tuned(out_ptr, in_ptr, n_elements, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        value = tl.load(in_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, value + 2, mask=mask)

    workers = 4
    num_devices = torch.cuda.device_count()
    start = threading.Barrier(workers)
    hook_lock = threading.Lock()
    hook_calls = {"enter": 0, "exit": 0}

    def enter_hook(metadata):
        with hook_lock:
            hook_calls["enter"] += 1

    def exit_hook(metadata):
        with hook_lock:
            hook_calls["exit"] += 1

    def launch(worker):
        device = worker % num_devices
        torch.cuda.set_device(device)
        stream = torch.cuda.Stream(device=device)
        src = torch.full((4096, ), worker, device=f"cuda:{device}", dtype=torch.float32)
        plain_dst = torch.empty_like(src)
        tuned_dst = torch.empty_like(src)
        start.wait(20)
        with torch.cuda.stream(stream):
            for _ in range(8):
                plain[(32, )](plain_dst, src, 4096, BLOCK=128)
                tuned[(32, )](tuned_dst, src, 4096)
        stream.synchronize()
        return worker, float(plain_dst.sum()), float(tuned_dst.sum())

    triton.knobs.runtime.launch_enter_hook.add(enter_hook)
    triton.knobs.runtime.launch_exit_hook.add(exit_hook)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(launch, worker) for worker in range(workers)]
            outputs = [future.result(timeout=60) for future in futures]
    finally:
        triton.knobs.runtime.launch_enter_hook.remove(enter_hook)
        triton.knobs.runtime.launch_exit_hook.remove(exit_hook)
    assert hook_calls["enter"] > 0
    assert hook_calls["enter"] == hook_calls["exit"]
    assert sorted(outputs) == [(worker, float((worker + 1) * 4096), float((worker + 2) * 4096))
                               for worker in range(workers)]
    assert not sys._is_gil_enabled()
