"""
Reproducibility tests for Proton.
Each test should invoke one or more GPU kernels and check the validity of their profiling results.
"""

import torch
import triton
import triton.profiler as proton
import json
import pytest
from typing import NamedTuple
import pathlib
import threading

import triton.language as tl
from triton.profiler.hooks.launch import COMPUTE_METADATA_SCOPE_NAME
import triton.profiler.hooks.launch as proton_launch


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@pytest.mark.parametrize("context", ["shadow", "python"])
def test_torch(context, tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_torch.hatchet"
    proton.start(str(temp_file.with_suffix("")), context=context)
    proton.enter_scope("test")
    torch.ones((2, 2), device="cuda")
    proton.exit_scope()
    proton.finalize()
    with temp_file.open() as f:
        data = json.load(f)
    if context == "shadow":
        assert len(data[0]["children"]) == 1
        assert data[0]["children"][0]["frame"]["name"] == "test"
        assert data[0]["children"][0]["children"][0]["metrics"]["time (ns)"] > 0
    elif context == "python":
        assert len(data[0]["children"]) == 1
        # bfs search until find the "elementwise_kernel" and then check its children
        queue = [data[0]]
        while len(queue) > 0:
            parent_frame = queue.pop(0)
            for child in parent_frame["children"]:
                if "elementwise_kernel" in child["frame"]["name"]:
                    assert len(child["children"]) == 0
                    return
                queue.append(child)


def test_triton(tmp_path: pathlib.Path):

    @triton.jit
    def foo(x, y):
        tl.store(y, tl.load(x))

    x = torch.tensor([2], device="cuda")
    y = torch.zeros_like(x)
    temp_file = tmp_path / "test_triton.hatchet"
    proton.start(str(temp_file.with_suffix("")))
    with proton.scope("test0"):
        with proton.scope("test1"):
            foo[(1, )](x, y)
    with proton.scope("test2"):
        foo[(1, )](x, y)
    proton.finalize()
    with temp_file.open() as f:
        data = json.load(f)
    assert len(data[0]["children"]) == 2
    assert data[0]["children"][0]["frame"]["name"] == "test0"
    assert len(data[0]["children"][0]["children"]) == 1
    assert data[0]["children"][0]["children"][0]["frame"]["name"] == "test1"
    assert data[0]["children"][1]["frame"]["name"] == "test2"


def test_cudagraph(tmp_path: pathlib.Path):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    @triton.jit
    def foo(x, y, z):
        tl.store(z, tl.load(y) + tl.load(x))

    def fn():
        a = torch.ones((2, 2), device="cuda")
        b = torch.ones((2, 2), device="cuda")
        c = a + b
        foo[(1, )](a, b, c)

    temp_file = tmp_path / "test_cudagraph.hatchet"
    proton.start(str(temp_file.with_suffix("")), context="shadow")

    # warmup
    # four kernels
    fn()

    # no kernels
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()

    proton.enter_scope("test")
    g.replay()
    g.reset()
    torch.cuda.synchronize()
    proton.exit_scope()
    proton.finalize()

    with temp_file.open() as f:
        data = json.load(f)
    # CUDA/HIP graph may also invoke additional kernels to reset outputs
    # {torch.ones, add, foo, test}
    assert len(data[0]["children"]) >= 4
    # find the test frame
    test_frame = None
    for child in data[0]["children"]:
        if child["frame"]["name"] == "test":
            test_frame = child
            break
    assert test_frame is not None
    # {torch.ones, add, foo}
    if is_hip():
        assert len(test_frame["children"]) >= 2
    else:
        assert len(test_frame["children"]) >= 3
    assert test_frame["children"][0]["metrics"]["time (ns)"] > 0


def test_metrics(tmp_path: pathlib.Path):

    @triton.jit
    def foo(x, y):
        tl.store(y, tl.load(x))

    x = torch.tensor([2], device="cuda")
    y = torch.zeros_like(x)
    temp_file = tmp_path / "test_metrics.hatchet"
    proton.start(str(temp_file.with_suffix("")))
    with proton.scope("test0", {"foo": 1.0}):
        foo[(1, )](x, y)
    proton.finalize()
    with temp_file.open() as f:
        data = json.load(f)
    assert len(data[0]["children"]) == 1
    assert data[0]["children"][0]["frame"]["name"] == "test0"
    assert data[0]["children"][0]["metrics"]["foo"] == 1.0


def test_scope_backward(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_scope_backward.hatchet"
    proton.start(str(temp_file.with_suffix("")))
    with proton.scope("ones1"):
        a = torch.ones((100, 100), device="cuda", requires_grad=True)
    with proton.scope("plus"):
        a2 = a * a * a
    with proton.scope("ones2"):
        loss = torch.ones_like(a2)

    # Backward triggers two kernels in a single scope
    with proton.scope("backward"):
        a2.backward(loss)
    proton.finalize()
    with temp_file.open() as f:
        data = json.load(f)
    assert len(data[0]["children"]) == 4


def test_cpu_timed_scope(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_cpu_timed_scope.hatchet"
    proton.start(str(temp_file.with_suffix("")))
    with proton.cpu_timed_scope("test0"):
        with proton.cpu_timed_scope("test1"):
            torch.ones((100, 100), device="cuda")
    proton.finalize()
    with temp_file.open() as f:
        data = json.load(f)
    assert len(data[0]["children"]) == 1
    test0_frame = data[0]["children"][0]
    assert test0_frame["metrics"]["cpu_time (ns)"] > 0
    test1_frame = test0_frame["children"][0]
    assert test1_frame["metrics"]["cpu_time (ns)"] > 0
    kernel_frame = test1_frame["children"][0]
    assert kernel_frame["metrics"]["time (ns)"] > 0


def test_hook_launch(tmp_path: pathlib.Path):

    def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        # get arg's element size
        element_size = args["x"].element_size()  # non-const
        size = args["size"]  # const
        key = "flops" + str(element_size * 8)
        num_ctas = metadata.num_ctas
        return {"name": f"foo_test_{num_ctas}ctas_{size}elems", key: 1.0}

    @triton.jit(launch_metadata=metadata_fn)
    def foo(x, size: tl.constexpr, y):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    x = torch.tensor([2], device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    temp_file = tmp_path / "test_hook_triton.hatchet"
    proton.start(str(temp_file.with_suffix("")), hook="triton")
    with proton.scope("test0"):
        foo[(1, )](x, 1, y, num_warps=4)
    proton.finalize()
    with temp_file.open() as f:
        data = json.load(f)
    assert len(data[0]["children"]) == 1
    assert data[0]["children"][0]["frame"]["name"] == "test0"
    assert data[0]["children"][0]["children"][0]["frame"]["name"] == "foo_test_1ctas_1elems"
    assert data[0]["children"][0]["children"][0]["metrics"]["flops32"] == 1.0
    assert data[0]["children"][0]["children"][0]["metrics"]["time (ns)"] > 0


@pytest.mark.parametrize("context", ["shadow", "python"])
def test_hook_launch_context(tmp_path: pathlib.Path, context: str):

    def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        x = args["x"]
        # A gpu kernel, but it should be under the metadata state
        return {"name": "foo_test", "bytes": x.sum().item()}

    @triton.jit(launch_metadata=metadata_fn)
    def foo(x, size: tl.constexpr, y):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    x = torch.tensor([2], device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    temp_file = tmp_path / "test_hook.hatchet"
    proton.start(str(temp_file.with_suffix("")), hook="triton", context=context)
    with proton.scope("test0"):
        foo[(1, )](x, 1, y, num_warps=4)
    proton.finalize()
    with temp_file.open() as f:
        data = json.load(f)
    # bfs search until find the reduce kernel and then check its parent
    queue = [data[0]]
    while len(queue) > 0:
        parent_frame = queue.pop(0)
        for child in parent_frame["children"]:
            if "reduce" in child["frame"]["name"]:
                assert parent_frame["frame"]["name"] == COMPUTE_METADATA_SCOPE_NAME
                return
            queue.append(child)


def test_hook_with_third_party(tmp_path: pathlib.Path):
    third_party_hook_invoked = False

    def third_party_hook(metadata) -> None:
        nonlocal third_party_hook_invoked
        third_party_hook_invoked = True

    triton.knobs.runtime.launch_enter_hook.add(third_party_hook)

    proton_hook_invoked = False

    def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        nonlocal proton_hook_invoked
        proton_hook_invoked = True
        return {"name": "foo_test"}

    @triton.jit(launch_metadata=metadata_fn)
    def foo(x, size: tl.constexpr, y):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    x = torch.tensor([2], device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    temp_file = tmp_path / "test_hook_with_third_party.hatchet"
    proton.start(str(temp_file.with_suffix("")), hook="triton")
    foo[(1, )](x, 1, y, num_warps=4)
    proton.finalize()
    triton.knobs.runtime.launch_enter_hook.remove(third_party_hook)
    with temp_file.open() as f:
        data = json.load(f)
    assert len(data[0]["children"]) == 1
    assert data[0]["children"][0]["frame"]["name"] == "foo_test"
    assert data[0]["children"][0]["metrics"]["time (ns)"] > 0


def test_hook_multiple_threads(tmp_path: pathlib.Path):

    def metadata_fn_foo(grid: tuple, metadata: NamedTuple, args: dict):
        return {"name": "foo_test"}

    @triton.jit(launch_metadata=metadata_fn_foo)
    def foo(x, size: tl.constexpr, y):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    def metadata_fn_bar(grid: tuple, metadata: NamedTuple, args: dict):
        return {"name": "bar_test"}

    @triton.jit(launch_metadata=metadata_fn_bar)
    def bar(x, size: tl.constexpr, y):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    x_foo = torch.tensor([2], device="cuda", dtype=torch.float32)
    y_foo = torch.zeros_like(x_foo)
    x_bar = torch.tensor([2], device="cuda", dtype=torch.float32)
    y_bar = torch.zeros_like(x_bar)

    temp_file = tmp_path / "test_hook.hatchet"
    proton.start(str(temp_file.with_suffix("")), hook="triton")

    all_ids = set()

    # start multiple threads
    def invoke_foo():
        for _ in range(100):
            foo[(1, )](x_foo, 1, y_foo, num_warps=4)
            all_ids.add(proton_launch.id.get())

    def invoke_bar():
        for _ in range(100):
            bar[(1, )](x_bar, 1, y_bar, num_warps=4)
            all_ids.add(proton_launch.id.get())

    thread_foo = threading.Thread(target=invoke_foo)
    thread_bar = threading.Thread(target=invoke_bar)
    thread_foo.start()
    thread_bar.start()
    thread_foo.join()
    thread_bar.join()

    proton.finalize()
    assert len(all_ids) == 200

    with temp_file.open() as f:
        data = json.load(f)
    root = data[0]["children"]
    assert "foo_test" in root[0]["frame"]["name"] or root[1]["frame"]["name"]
    assert "bar_test" in root[0]["frame"]["name"] or root[1]["frame"]["name"]
    assert root[0]["metrics"]["count"] == 100
    assert root[1]["metrics"]["count"] == 100


def test_pcsampling(tmp_path: pathlib.Path):
    if is_hip():
        pytest.skip("HIP backend does not support pc sampling")

    import os

    if os.environ.get("PROTON_SKIP_PC_SAMPLING_TEST", "0") == "1":
        pytest.skip("PC sampling test is disabled")

    @triton.jit
    def foo(x, y, size: tl.constexpr):
        offs = tl.arange(0, size)
        for _ in range(1000):
            tl.store(y + offs, tl.load(x + offs))

    temp_file = tmp_path / "test_pcsampling.hatchet"
    proton.start(str(temp_file.with_suffix("")), hook="triton", backend="cupti", mode="pcsampling")
    with proton.scope("init"):
        x = torch.ones((1024, ), device="cuda", dtype=torch.float32)
        y = torch.zeros_like(x)
    with proton.scope("test"):
        foo[(1, )](x, y, x.size()[0], num_warps=4)
    proton.finalize()
    with temp_file.open() as f:
        data = json.load(f)
    init_frame = data[0]["children"][0]
    test_frame = data[0]["children"][1]
    # With line mapping
    assert "foo" in test_frame["children"][0]["frame"]["name"]
    assert test_frame["children"][0]["children"][0]["metrics"]["num_samples"] > 0
    assert "@" in test_frame["children"][0]["children"][0]["frame"]["name"]
    # Without line mapping
    assert "elementwise" in init_frame["children"][0]["frame"]["name"]
    assert init_frame["children"][0]["metrics"]["num_samples"] > 0


def test_deactivate(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_deactivate.hatchet"
    session_id = proton.start(str(temp_file.with_suffix("")), hook="triton")
    proton.deactivate(session_id)
    torch.randn((10, 10), device="cuda")
    proton.activate(session_id)
    torch.zeros((10, 10), device="cuda")
    proton.deactivate(session_id)
    proton.finalize()
    with temp_file.open() as f:
        data = json.load(f)
    # Root shouldn't have device id
    assert "device_id" not in data[0]["metrics"]
    assert len(data[0]["children"]) == 1
    assert "device_id" in data[0]["children"][0]["metrics"]


def test_multiple_sessions(tmp_path: pathlib.Path):
    temp_file0 = tmp_path / "test_multiple_sessions0.hatchet"
    temp_file1 = tmp_path / "test_multiple_sessions1.hatchet"
    session_id0 = proton.start(str(temp_file0.with_suffix("")))
    session_id1 = proton.start(str(temp_file1.with_suffix("")))
    with proton.scope("scope0"):
        torch.randn((10, 10), device="cuda")
        torch.randn((10, 10), device="cuda")
    proton.deactivate(session_id0)
    proton.finalize(session_id0)
    with proton.scope("scope1"):
        torch.randn((10, 10), device="cuda")
    proton.finalize(session_id1)
    # kernel has been invoked twice in session 0 and three times in session 1
    with temp_file0.open() as f:
        data = json.load(f)
    assert data[0]["children"][0]["frame"]["name"] == "scope0"
    assert int(data[0]["children"][0]["children"][0]["metrics"]["count"]) == 2
    with temp_file1.open() as f:
        data = json.load(f)
    scope0_count = int(data[0]["children"][0]["children"][0]["metrics"]["count"])
    scope1_count = int(data[0]["children"][1]["children"][0]["metrics"]["count"])
    assert scope0_count + scope1_count == 3


def test_trace(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_trace.chrome_trace"
    proton.start(str(temp_file.with_suffix("")), data="trace")

    @triton.jit
    def foo(x, y, size: tl.constexpr):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    with proton.scope("init"):
        x = torch.ones((1024, ), device="cuda", dtype=torch.float32)
        y = torch.zeros_like(x)

    with proton.scope("test"):
        foo[(1, )](x, y, x.size()[0], num_warps=4)

    proton.finalize()

    with temp_file.open() as f:
        data = json.load(f)
        trace_events = data["traceEvents"]
        assert len(trace_events) == 3
        assert trace_events[-1]["name"] == "foo"
        assert trace_events[-1]["args"]["call_stack"] == ["ROOT", "test", "foo"]


def test_scope_multiple_threads(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_scope_threads.hatchet"
    proton.start(str(temp_file.with_suffix("")))

    N = 50
    thread_names = ["threadA", "threadB"]

    def worker(prefix: str):
        for i in range(N):
            name = f"{prefix}_{i}"
            proton.enter_scope(name)
            torch.ones((1, ), device="cuda")
            proton.exit_scope()

    threads = [threading.Thread(target=worker, args=(tname, )) for tname in thread_names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    proton.finalize()

    with temp_file.open() as f:
        data = json.load(f)

    children = data[0]["children"]
    assert len(children) == N * len(thread_names)
    names = {c["frame"]["name"] for c in children}
    expected = {f"{t}_{i}" for t in thread_names for i in range(N)}
    assert names == expected
