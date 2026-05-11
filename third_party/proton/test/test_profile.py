"""
Reproducibility tests for Proton.
Each test should invoke one or more GPU kernels and check the validity of their profiling results.
"""

import torch
import triton
import triton.profiler as proton
import json
import gc
import pytest
from typing import NamedTuple
import pathlib
import threading
from contextlib import contextmanager

import triton.language as tl
import triton.profiler.hooks.launch as proton_launch
from triton.profiler.state import COMPUTE_METADATA_SCOPE_NAME
import triton.profiler.viewer as viewer
from triton._internal_testing import is_hip, is_cuda, is_blackwell


def _find_frame_by_name(frame, name):
    queue = [frame]
    while queue:
        current = queue.pop(0)
        if current["frame"]["name"] == name:
            return current
        queue.extend(current["children"])
    return None


@contextmanager
def cuda_graph_without_gc(*args, **kwargs):
    # A loaded Triton CompiledKernel may be finalized by Python's cyclic GC.
    # Its destructor unloads the CUDA module, which is illegal during CUDA
    # stream capture and invalidates the graph. Keep GC disabled only for the
    # capture window and restore the caller's previous GC state afterwards.
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        with torch.cuda.graph(*args, **kwargs) as graph:
            yield graph
    finally:
        if gc_was_enabled:
            gc.enable()


@pytest.mark.parametrize("context", ["shadow", "python"])
def test_torch(context, tmp_path: pathlib.Path, device: str):
    temp_file = tmp_path / "test_torch.hatchet"
    proton.start(str(temp_file.with_suffix("")), context=context)
    proton.enter_scope("test")
    torch.ones((2, 2), device=device)
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
        import re
        while len(queue) > 0:
            parent_frame = queue.pop(0)
            for child in parent_frame["children"]:
                if "elementwise_kernel" in child["frame"]["name"]:
                    assert len(child["children"]) == 0
                    # check the regex of the parent name matches
                    # file_name:line_number@function_name
                    regex = r".+:\d+@.+"
                    assert re.match(regex, parent_frame["frame"]["name"])
                    return
                queue.append(child)


def test_triton(tmp_path: pathlib.Path, device: str):

    @triton.jit
    def foo(x, y):
        tl.store(y, tl.load(x))

    x = torch.tensor([2], device=device)
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


@pytest.mark.skipif(not is_cuda(), reason="HIP backend does not reliably attribute cudagraph replay launches to scopes")
def test_cudagraph(tmp_path: pathlib.Path, device: str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        return {"name": "foo_test"}

    @triton.jit(launch_metadata=metadata_fn)
    def foo(x, y, z):
        tl.store(z, tl.load(y) + tl.load(x))

    def fn():
        a = torch.ones((2, 2), device=device)
        b = torch.ones((2, 2), device=device)
        c = a + b
        foo[(1, )](a, b, c)

    temp_file = tmp_path / "test_cudagraph.hatchet"
    proton.start(str(temp_file.with_suffix("")), context="shadow")

    # warmup
    # four kernels
    fn()

    # no kernels
    g = torch.cuda.CUDAGraph()
    with cuda_graph_without_gc(g):
        for i in range(10):
            with proton.scope(f"iter_{i}"):
                fn()

    with proton.scope("test0"):
        g.replay()

    with proton.scope("test1"):
        g.replay()

    g.reset()

    with cuda_graph_without_gc(g):  # this will create new graphexecs
        for i in range(10):
            with proton.scope(f"new_iter_{i}"):
                fn()

    with proton.scope("test2"):
        g.replay()

    proton.finalize()

    with temp_file.open() as f:
        data = json.load(f)
    # find the test frame
    test0_frame = None
    test1_frame = None
    test2_frame = None
    for child in data[0]["children"]:
        if child["frame"]["name"] == "test0":
            test0_frame = child
        if child["frame"]["name"] == "test1":
            test1_frame = child
        if child["frame"]["name"] == "test2":
            test2_frame = child
    assert test0_frame is not None
    assert test1_frame is not None
    assert test2_frame is not None
    # {torch.ones, add, foo}
    if is_hip():
        assert len(test0_frame["children"]) >= 2
        assert test0_frame["children"][0]["metrics"]["time (ns)"] > 0
    else:
        # cuda backend supports "<captured_at>" annotation
        for test_frame in [test0_frame, test1_frame, test2_frame]:
            child = _find_frame_by_name(test_frame, "<captured_at>")
            assert child is not None
            # check all iterations
            total_iters = 0
            for child in child["children"]:
                iter_frame = "iter" if test_frame != test2_frame else "new_iter"
                if iter_frame in child["frame"]["name"]:  # TODO(Keren): remove empty frames
                    if "time (ns)" in child["children"][0]["metrics"]:
                        total_iters += 1
            # 0...9 iterations
            assert total_iters == 10


@pytest.mark.skipif(not is_cuda(), reason="Only CUDA backend supports cudagraph replay")
def test_cudagraph_not_captured_by_profiler(tmp_path: pathlib.Path, capfd, device: str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    @triton.jit
    def foo(x, y, z):
        tl.store(z, tl.load(y) + tl.load(x))

    def fn():
        a = torch.ones((2, 2), device=device)
        b = torch.ones((2, 2), device=device)
        c = a + b
        foo[(1, )](a, b, c)

    # Build/capture graph before profiler starts.
    fn()
    g = torch.cuda.CUDAGraph()
    with cuda_graph_without_gc(g):
        fn()

    temp_file = tmp_path / "test_cudagraph_not_captured_by_profiler.hatchet"
    proton.start(str(temp_file.with_suffix("")), context="shadow")
    with proton.scope("replay0"):
        g.replay()
    with proton.scope("replay1"):
        g.replay()
    proton.finalize()

    captured = capfd.readouterr()
    assert captured.err.count("Cannot find graph for graphExecId:") == 1
    assert "start profiling before the graph is created" in captured.err

    with temp_file.open() as f:
        data = json.load(f)
    replay0_frame = None
    replay1_frame = None
    for child in data[0]["children"]:
        if child["frame"]["name"] == "replay0":
            replay0_frame = child
        elif child["frame"]["name"] == "replay1":
            replay1_frame = child
    assert replay0_frame is not None
    assert replay1_frame is not None
    assert len(replay0_frame["children"]) >= 3
    assert len(replay1_frame["children"]) >= 3

    def has_positive_time_metric(node):
        if node["metrics"].get("time (ns)", 0) > 0:
            return True
        return any(has_positive_time_metric(child) for child in node["children"])

    assert has_positive_time_metric(replay0_frame)
    assert has_positive_time_metric(replay1_frame)


@pytest.mark.skipif(not is_cuda(), reason="Only CUDA backend supports cudagraph deactivation")
def test_cudagraph_deactivate(tmp_path, device: str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    @triton.jit
    def foo(x, y, z):
        tl.store(z, tl.load(y) + tl.load(x))

    def fn(session):
        with proton.scope("scope_a"):
            a = torch.ones((2, 2), device=device)
        proton.deactivate(session)
        with proton.scope("scope_b"):
            b = torch.ones((2, 2), device=device)
        proton.activate(session)
        with proton.scope("scope_c"):
            c = a + b
        foo[(1, )](a, b, c)

    temp_file = tmp_path / "test_cudagraph_deactivate.hatchet"
    session = proton.start(str(temp_file.with_suffix("")), context="shadow", hook="triton")

    # warmup
    fn(session)

    # no kernels
    g = torch.cuda.CUDAGraph()
    with cuda_graph_without_gc(g):
        for i in range(10):
            with proton.scope(f"iter_{i}"):
                fn(session)

    with proton.scope("test0"):
        g.replay()

    proton.finalize()

    with temp_file.open() as f:
        data = json.load(f)

    # scope a and c should be recorded, b should be skipped
    children = data[0]["children"]
    test0_frame = None
    for child in children:
        if child["frame"]["name"] == "test0":
            test0_frame = child
            break
    assert test0_frame is not None
    capture_frame = _find_frame_by_name(test0_frame, "<captured_at>")
    assert capture_frame is not None
    iter_frame = _find_frame_by_name(capture_frame, "iter_0")
    assert iter_frame is not None
    scope_a_frame = None
    scope_b_frame = None
    scope_c_frame = None
    for child in iter_frame["children"]:
        if child["frame"]["name"] == "scope_a":
            scope_a_frame = child
        if child["frame"]["name"] == "scope_b":
            scope_b_frame = child
        if child["frame"]["name"] == "scope_c":
            scope_c_frame = child
    assert scope_a_frame is not None
    assert scope_b_frame is None
    assert scope_c_frame is not None


@pytest.mark.skipif(not is_cuda(), reason="Only CUDA backend supports cudagraph replay")
@pytest.mark.parametrize("data_format", ["hatchet", "hatchet_msgpack"])
def test_cudagraph_filters_unlinked_virtual_scopes(tmp_path: pathlib.Path, data_format: str, device: str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    @triton.jit
    def foo(x, y, z):
        tl.store(z, tl.load(y) + tl.load(x))

    a = torch.ones((2, 2), device=device)
    b = torch.ones((2, 2), device=device)
    c = torch.empty_like(a)

    temp_file = tmp_path / f"test_cudagraph_filters_unlinked_virtual_scopes.{data_format}"
    proton.start(str(temp_file.with_suffix("")), context="shadow")

    # Warmup to avoid one-time setup effects in replay output.
    foo[(1, )](a, b, c)

    g = torch.cuda.CUDAGraph()
    with cuda_graph_without_gc(g):
        with proton.scope("iter_with_kernel"):
            foo[(1, )](a, b, c)
        with proton.scope("iter_without_kernel"):
            pass

    with proton.scope("replay"):
        g.replay()

    proton.finalize(output_format=data_format)

    if data_format == "hatchet_msgpack":
        import msgpack

        with temp_file.open("rb") as f:
            data = msgpack.load(f, raw=False, strict_map_key=False)
    else:
        with temp_file.open() as f:
            data = json.load(f)

    replay_frame = next(
        (child for child in data[0]["children"] if child["frame"]["name"] == "replay"),
        None,
    )
    assert replay_frame is not None
    capture_frame = _find_frame_by_name(replay_frame, "<captured_at>")
    assert capture_frame is not None

    capture_children = capture_frame["children"]
    capture_child_names = {child["frame"]["name"] for child in capture_children}
    assert "iter_with_kernel" in capture_child_names
    assert "iter_without_kernel" not in capture_child_names

    iter_with_kernel_frame = next(
        (child for child in capture_children if child["frame"]["name"] == "iter_with_kernel"),
        None,
    )
    assert iter_with_kernel_frame is not None
    assert len(iter_with_kernel_frame["children"]) > 0
    assert iter_with_kernel_frame["children"][0]["metrics"]["time (ns)"] > 0


@pytest.mark.skipif(not is_cuda(), reason="Only CUDA backend supports metrics profiling in cudagraphs")
def test_cudagraph_multi_stream(tmp_path: pathlib.Path, device: str):
    """
    kernels in a cudagraph can be launched using multiple internal streams, without
    a deterministic order.
    """

    capture_stream = torch.cuda.Stream()
    side_stream = torch.cuda.Stream()
    torch.cuda.set_stream(capture_stream)

    kernel_x_name = "kernel_x"
    kernel_y_name = "kernel_y"
    kernel_z_name = "kernel_z"

    kernel_x_metrics = {"flops": 0.0, "bytes": 1_572_864}
    kernel_y_metrics = {"flops": 134_217_728.0, "bytes": 17_829_888}
    kernel_z_metrics = {"flops": 1_073_741_824.0, "bytes": 29_818_880}

    @triton.jit
    def wait_for_flag_kernel(flag):
        while tl.load(flag, volatile=True) == 0:
            pass

    @triton.jit
    def set_flag_kernel(flag):
        tl.store(flag, 1)

    @triton.jit
    def metadata_delay_kernel(scratch, BLOCK: tl.constexpr, ITERS: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        values = offsets.to(tl.float32)
        for _ in tl.static_range(0, ITERS):
            values = values * 1.0001 + 1.0
        tl.store(scratch + offsets, values)

    def kernel_x_metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        wait_for_flag_kernel[(1, )](args["gate"], num_warps=1)
        metadata_delay_kernel[(2048, )](args["delay_scratch"], BLOCK=256, ITERS=64, num_warps=8)
        return {"name": kernel_x_name, "flops": args["kernel_x_flops"], "bytes": args["kernel_x_bytes"]}

    def kernel_y_metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        return {"name": kernel_y_name, "flops": args["kernel_y_flops"], "bytes": args["kernel_y_bytes"]}

    def kernel_z_metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        return {"name": kernel_z_name, "flops": args["kernel_z_flops"], "bytes": args["kernel_z_bytes"]}

    @triton.jit(launch_metadata=kernel_x_metadata_fn)
    def kernel_x(x, y, kernel_x_flops, kernel_x_bytes, delay_scratch, gate):
        tl.store(y, tl.load(x) + 1.0)

    @triton.jit(launch_metadata=kernel_y_metadata_fn)
    def kernel_y(x, y, kernel_y_flops, kernel_y_bytes):
        tl.store(y, tl.load(x) + 2.0)

    @triton.jit(launch_metadata=kernel_z_metadata_fn)
    def kernel_z(x, y, kernel_z_flops, kernel_z_bytes):
        tl.store(y, tl.load(x) + 3.0)

    def find_frame(node, name: str):
        queue = [node]
        while queue:
            cur = queue.pop(0)
            if cur["frame"]["name"] == name:
                return cur
            queue.extend(cur["children"])
        return None

    x = torch.tensor([1.0], device=device)
    y = torch.empty_like(x)
    delay_scratch = torch.empty((2048 * 256, ), device=device)
    kernel_x_flops = torch.tensor([kernel_x_metrics["flops"]], device=device)
    kernel_x_bytes = torch.tensor([kernel_x_metrics["bytes"]], device=device, dtype=torch.int64)
    kernel_y_flops = torch.tensor([kernel_y_metrics["flops"]], device=device)
    kernel_y_bytes = torch.tensor([kernel_y_metrics["bytes"]], device=device, dtype=torch.int64)
    kernel_z_flops = torch.tensor([kernel_z_metrics["flops"]], device=device)
    kernel_z_bytes = torch.tensor([kernel_z_metrics["bytes"]], device=device, dtype=torch.int64)
    gate = torch.ones((1, ), device=device, dtype=torch.int32)

    temp_file = tmp_path / "test_cudagraph_multi_stream.hatchet"
    session = proton.start(str(temp_file.with_suffix("")), context="shadow", hook="triton")
    try:
        kernel_x[(1, )](x, y, kernel_x_flops, kernel_x_bytes, delay_scratch, gate, num_warps=1)
        kernel_y[(1, )](x, y, kernel_y_flops, kernel_y_bytes, num_warps=1)
        kernel_z[(1, )](x, y, kernel_z_flops, kernel_z_bytes, num_warps=1)
        torch.cuda.synchronize()
        gate.zero_()
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        start_event = torch.cuda.Event()
        with cuda_graph_without_gc(graph, stream=capture_stream):
            start_event.record()
            # x and y are executed concurrently
            kernel_x[(1, )](x, y, kernel_x_flops, kernel_x_bytes, delay_scratch, gate, num_warps=1)
            with torch.cuda.stream(side_stream):
                side_stream.wait_event(start_event)
                kernel_y[(1, )](x, y, kernel_y_flops, kernel_y_bytes, num_warps=1)
                kernel_z[(1, )](x, y, kernel_z_flops, kernel_z_bytes, num_warps=1)
                set_flag_kernel[(1, )](gate, num_warps=1)
            capture_stream.wait_stream(side_stream)

        with proton.scope("replay"):
            graph.replay()
        torch.cuda.synchronize()
    finally:
        proton.finalize(session)

    with temp_file.open() as f:
        data = json.load(f)

    replay_frame = find_frame(data[0], "replay")
    assert replay_frame is not None
    capture_frame = find_frame(replay_frame, "<captured_at>")
    assert capture_frame is not None

    for name, expected in [
        (kernel_x_name, kernel_x_metrics),
        (kernel_y_name, kernel_y_metrics),
        (kernel_z_name, kernel_z_metrics),
    ]:
        frame = find_frame(capture_frame, name)
        assert frame is not None
        assert frame["metrics"]["flops"] == expected["flops"]
        assert frame["metrics"]["bytes"] == expected["bytes"]


def test_metrics(tmp_path: pathlib.Path, device: str):

    @triton.jit
    def foo(x, y):
        tl.store(y, tl.load(x))

    x = torch.tensor([2], device=device)
    y = torch.zeros_like(x)
    temp_file = tmp_path / "test_metrics.hatchet"
    proton.start(str(temp_file.with_suffix("")))
    with proton.scope("test0", {"foo": 1.0, "bar": [1, 2, 3], "baz": [1.0, 2.0, 3.0]}):
        foo[(1, )](x, y)
    proton.finalize()
    with temp_file.open() as f:
        data = json.load(f)
    assert len(data[0]["children"]) == 1
    assert data[0]["children"][0]["frame"]["name"] == "test0"
    assert data[0]["children"][0]["metrics"]["foo"] == 1.0
    assert data[0]["children"][0]["metrics"]["bar"] == [1, 2, 3]


def test_scope_backward(tmp_path: pathlib.Path, device: str):
    temp_file = tmp_path / "test_scope_backward.hatchet"
    proton.start(str(temp_file.with_suffix("")))
    with proton.scope("ones1"):
        a = torch.ones((100, 100), device=device, requires_grad=True)
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


def test_cpu_timed_scope(tmp_path: pathlib.Path, device: str):
    temp_file = tmp_path / "test_cpu_timed_scope.hatchet"
    proton.start(str(temp_file.with_suffix("")))
    with proton.cpu_timed_scope("test0"):
        with proton.cpu_timed_scope("test1"):
            torch.ones((100, 100), device=device)
    proton.finalize()
    with temp_file.open() as f:
        data = json.load(f)
    assert len(data[0]["children"]) == 1
    test0_frame = data[0]["children"][0]
    assert test0_frame["metrics"]["cpu_time (ns)"] > 0
    test1_frame = test0_frame["children"][0]
    assert test1_frame["metrics"]["cpu_time (ns)"] > 0


def test_get_data(tmp_path: pathlib.Path, device: str):
    temp_file = tmp_path / "test_get_data.hatchet"
    session = proton.start(str(temp_file.with_suffix("")), context="shadow")

    @triton.jit
    def foo(x, y, size: tl.constexpr):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    with proton.scope("test"):
        x = torch.ones((2, 2), device=device)
        foo[(1, )](x, x, 4)
        foo[(1, )](x, x, 4)

    proton.deactivate(session, flushing=True)

    database = proton.data.get(session)
    gf, _, _, _ = viewer.get_raw_metrics(database)
    foo_frame = gf.filter("MATCH ('*', c) WHERE c.'name' =~ '.*foo.*' AND c IS LEAF").dataframe
    ones_frame = gf.filter("MATCH ('*', c) WHERE c.'name' =~ '.*elementwise.*' AND c IS LEAF").dataframe

    assert len(foo_frame) == 1
    assert int(foo_frame["count"].values[0]) == 2
    assert len(ones_frame) == 1
    assert int(ones_frame["count"].values[0]) == 1

    import msgpack
    msgpack_data = proton.data.get_msgpack(session)
    database_unpacked = msgpack.loads(msgpack_data, raw=False, strict_map_key=False)
    assert database == database_unpacked

    proton.finalize()


def test_clear_data(tmp_path: pathlib.Path, device: str):
    temp_file = tmp_path / "test_clear_data.hatchet"
    session = proton.start(str(temp_file.with_suffix("")), context="shadow")

    with proton.scope("test0"):
        x = torch.ones((2, 2), device=device)
        x + x  # type: ignore

    proton.deactivate(session, flushing=True)
    proton.data.clear(session)
    try:
        database = proton.data.get(session)
    except RuntimeError as e:
        assert "has no data" in str(e)

    proton.activate(session)
    with proton.scope("test1"):
        x * x  # type: ignore
    proton.deactivate(session, flushing=True)
    database = proton.data.get(session)

    proton.finalize()
    assert len(database[0]["children"]) == 1
    assert database[0]["children"][0]["frame"]["name"] == "test1"
    kernel_frame = database[0]["children"][0]["children"][0]
    assert "elementwise" in kernel_frame["frame"]["name"]


def test_clear_data_up_to_phase(tmp_path: pathlib.Path, device: str):
    temp_file = tmp_path / "test_clear_data_up_to_phase.hatchet"
    session = proton.start(str(temp_file.with_suffix("")), context="shadow")

    with proton.scope("phase0"):
        x = torch.ones((2, 2), device=device)
        x + x  # type: ignore

    phase1 = proton.data.advance_phase(session)
    with proton.scope("phase1"):
        x = torch.ones((2, 2), device=device)
        x + x  # type: ignore

    proton.deactivate(session, flushing=True)

    # Clear a range of phases.
    proton.data.clear(session, phase=phase1, clear_up_to_phase=True)
    database = proton.data.get(session, phase=phase1)
    assert len(database[0]["children"]) == 0

    proton.finalize()


def test_data_is_phase_complete(tmp_path: pathlib.Path, device: str):
    temp_path = tmp_path / "test_data_is_phase_complete.hatchet"
    session = proton.start(str(temp_path.with_suffix("")), context="shadow")

    def fn():
        with proton.scope("test0"):
            x = torch.ones((2, 2), device=device)
            x + x  # type: ignore

    fn()
    assert not proton.data.is_phase_complete(session, 0)

    proton.deactivate(session)
    # likely the GPU has not completed the data yet
    assert not proton.data.is_phase_complete(session, 0)

    proton.activate(session)
    phase = proton.data.advance_phase(session)
    fn()
    proton.deactivate(session, flushing=True)
    # session 0 is a previous phase but we have called deactivate with flushing
    assert proton.data.is_phase_complete(session, 0)
    # phase 1 is the current phase so cannot be a completed phase
    assert not proton.data.is_phase_complete(session, phase)
    proton.data.advance_phase(session)
    # phase 0 should remain completed after advancing phases
    assert proton.data.is_phase_complete(session, phase - 1)

    proton.finalize()


def test_hook_launch(tmp_path: pathlib.Path, device: str):

    def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        # get arg's element size
        element_size = args["x"].element_size()  # non-const
        size = args["size"]  # const
        key = "flops" + str(element_size * 8)
        num_ctas = metadata.num_ctas
        # Return an extra metric key beyond the historical flops/bytes allowlist.
        return {"name": f"foo_test_{num_ctas}ctas_{size}elems", key: 1.0, "extra_metric": 7.0}

    @triton.jit(launch_metadata=metadata_fn)
    def foo(x, size: tl.constexpr, y):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    x = torch.tensor([2], device=device, dtype=torch.float32)
    y = torch.zeros_like(x)
    temp_file = tmp_path / "test_hook_launch.hatchet"
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
    assert data[0]["children"][0]["children"][0]["metrics"]["extra_metric"] == 7.0
    assert data[0]["children"][0]["children"][0]["metrics"]["time (ns)"] > 0


def test_hook_launch_filter(tmp_path: pathlib.Path, device: str):

    foo_metadata_invoked = False
    bar_metadata_invoked = False

    def foo_metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        nonlocal foo_metadata_invoked
        foo_metadata_invoked = True
        return {"name": "foo_meta", "flops": 1.0}

    def bar_metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        nonlocal bar_metadata_invoked
        bar_metadata_invoked = True
        return {"name": "bar_meta", "flops": 2.0}

    @triton.jit(launch_metadata=foo_metadata_fn)
    def foo(x, size: tl.constexpr, y):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    @triton.jit(launch_metadata=bar_metadata_fn)
    def bar(x, size: tl.constexpr, y):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    x = torch.tensor([2], device=device, dtype=torch.float32)
    y = torch.zeros_like(x)
    temp_file = tmp_path / "test_hook_launch_filter.hatchet"

    # Only allow kernels whose compiled name matches "foo" (via prefix regex).
    launch_hook = proton_launch.LaunchHook()
    launch_hook.configure(include=".*foo")
    proton.start(str(temp_file.with_suffix("")), hook=launch_hook)
    with proton.scope("test0"):
        foo[(1, )](x, 1, y, num_warps=4)
        bar[(1, )](x, 1, y, num_warps=4)
    proton.finalize()
    # Reset singleton hook state to avoid leaking filter settings across tests.
    launch_hook.configure(include=None, exclude=None)

    assert foo_metadata_invoked is True
    assert bar_metadata_invoked is False

    with temp_file.open() as f:
        data = json.load(f)

    # Ensure the "foo_meta" override exists and "bar_meta" does not.
    all_names = set()
    queue = [data[0]]
    while queue:
        node = queue.pop()
        if "frame" in node and "name" in node["frame"]:
            all_names.add(node["frame"]["name"])
        queue.extend(node.get("children", []))

    assert "foo_meta" in all_names
    assert "bar_meta" not in all_names


@pytest.mark.parametrize("context", ["shadow", "python"])
def test_hook_launch_context(tmp_path: pathlib.Path, context: str, device: str):

    def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        x = args["x"]
        # A gpu kernel, but it should be under the metadata state
        return {"name": "foo_test", "bytes": x.sum().item()}

    @triton.jit(launch_metadata=metadata_fn)
    def foo(x, size: tl.constexpr, y):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    x = torch.tensor([2], device=device, dtype=torch.float32)
    y = torch.zeros_like(x)
    temp_file = tmp_path / "test_hook_launch_context.hatchet"
    proton.start(str(temp_file.with_suffix("")), hook="triton", context=context)
    with proton.scope("test0"):
        foo[(1, )](x, 1, y, num_warps=4)
    proton.finalize()
    with temp_file.open() as f:
        data = json.load(f)
    # bfs search until find the reduce kernel and then check its parent
    queue = [(data[0], [data[0]["frame"]["name"]])]
    while len(queue) > 0:
        parent_frame, parent_path = queue.pop(0)
        for child in parent_frame["children"]:
            if "reduce" in child["frame"]["name"]:
                assert parent_frame["frame"]["name"] != COMPUTE_METADATA_SCOPE_NAME
                assert parent_path[-2] == COMPUTE_METADATA_SCOPE_NAME
                return
            queue.append((child, parent_path + [child["frame"]["name"]]))


def test_hook_with_third_party(tmp_path: pathlib.Path, device: str):
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

    x = torch.tensor([2], device=device, dtype=torch.float32)
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


def test_hook_multiple_threads(tmp_path: pathlib.Path, device: str):

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

    x_foo = torch.tensor([2], device=device, dtype=torch.float32)
    y_foo = torch.zeros_like(x_foo)
    x_bar = torch.tensor([2], device=device, dtype=torch.float32)
    y_bar = torch.zeros_like(x_bar)

    temp_file = tmp_path / "test_hook_multiple_threads.hatchet"
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


def test_pcsampling(tmp_path: pathlib.Path, device: str):
    if not is_cuda():
        pytest.skip("Only CUDA backend supports pc sampling")

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
        x = torch.ones((1024, ), device=device, dtype=torch.float32)
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


def test_deactivate(tmp_path: pathlib.Path, device: str):
    temp_file = tmp_path / "test_deactivate.hatchet"
    session_id = proton.start(str(temp_file.with_suffix("")), hook="triton")
    proton.deactivate(session_id)
    torch.randn((10, 10), device=device)
    proton.activate(session_id)
    torch.zeros((10, 10), device=device)
    proton.deactivate(session_id)
    proton.finalize()
    with temp_file.open() as f:
        data = json.load(f)
    # Root shouldn't have device id
    assert "device_id" not in data[0]["metrics"]
    assert len(data[0]["children"]) == 1
    assert "device_id" in data[0]["children"][0]["metrics"]


def test_multiple_sessions(tmp_path: pathlib.Path, device: str):
    temp_file0 = tmp_path / "test_multiple_sessions_0.hatchet"
    temp_file1 = tmp_path / "test_multiple_sessions_1.hatchet"
    session_id0 = proton.start(str(temp_file0.with_suffix("")))
    session_id1 = proton.start(str(temp_file1.with_suffix("")))
    with proton.scope("scope0"):
        torch.randn((10, 10), device=device)
        torch.randn((10, 10), device=device)
    proton.deactivate(session_id0)
    proton.finalize(session_id0)
    with proton.scope("scope1"):
        torch.randn((10, 10), device=device)
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


@pytest.mark.skipif(not is_cuda(), reason="Only CUDA backend supports metrics profiling in cudagraphs")
def test_multiple_sessions_cudagraph_metric_kernels(tmp_path: pathlib.Path, device: str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    foo_iters = 3
    bar_iters = 2

    def foo_metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        x = args["x"]
        # Tensor custom metric in graph capture mode launches metric kernels.
        return {"name": "foo_with_metric", "sum_metric": x.sum()}

    def bar_metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        # Name-only metadata (no custom metric).
        return {"name": "bar_without_metric"}

    @triton.jit(launch_metadata=foo_metadata_fn)
    def foo(x, y, z):
        tl.store(z, tl.load(y) + tl.load(x))

    @triton.jit(launch_metadata=bar_metadata_fn)
    def bar(x, y, z):
        tl.store(z, tl.load(y) - tl.load(x))

    x = torch.ones((2, 2), device=device)
    y = torch.ones((2, 2), device=device)
    z = torch.empty_like(x)

    # Compile kernels before profiling starts to reduce unrelated profile noise.
    foo[(1, )](x, y, z)
    bar[(1, )](x, y, z)

    temp_file0 = tmp_path / "test_multiple_sessions_cudagraph_metric_kernels_0.hatchet"
    temp_file1 = tmp_path / "test_multiple_sessions_cudagraph_metric_kernels_1.hatchet"
    session_id0 = proton.start(str(temp_file0.with_suffix("")), context="shadow", hook="triton")
    session_id1 = proton.start(str(temp_file1.with_suffix("")), context="shadow", hook="triton")

    proton.deactivate(session_id1)

    graph_foo = torch.cuda.CUDAGraph()
    with cuda_graph_without_gc(graph_foo):
        for _ in range(foo_iters):
            foo[(1, )](x, y, z)
    with proton.scope("session0_replay"):
        graph_foo.replay()

    proton.deactivate(session_id0)
    proton.activate(session_id1)

    graph_bar = torch.cuda.CUDAGraph()
    with cuda_graph_without_gc(graph_bar):
        for _ in range(bar_iters):
            bar[(1, )](x, y, z)
    with proton.scope("session1_replay"):
        graph_bar.replay()

    proton.finalize(session_id0)
    proton.finalize(session_id1)

    def get_frame_by_name(node, name: str):
        queue = [node]
        while queue:
            cur = queue.pop(0)
            if cur["frame"]["name"] == name:
                return cur
            queue.extend(cur["children"])
        return None

    def get_all_names(node):
        names = set()
        queue = [node]
        while queue:
            cur = queue.pop(0)
            names.add(cur["frame"]["name"])
            queue.extend(cur["children"])
        return names

    with temp_file0.open() as f:
        data0 = json.load(f)
    with temp_file1.open() as f:
        data1 = json.load(f)

    session0_replay_frame = get_frame_by_name(data0[0], "session0_replay")
    session1_replay_frame = get_frame_by_name(data1[0], "session1_replay")
    assert session0_replay_frame is not None
    assert session1_replay_frame is not None

    capture0 = _find_frame_by_name(session0_replay_frame, "<captured_at>")
    capture1 = _find_frame_by_name(session1_replay_frame, "<captured_at>")
    assert capture0 is not None
    assert capture1 is not None

    foo_frame0 = get_frame_by_name(capture0, "foo_with_metric")
    bar_frame0 = get_frame_by_name(capture0, "bar_without_metric")
    foo_frame1 = get_frame_by_name(capture1, "foo_with_metric")
    bar_frame1 = get_frame_by_name(capture1, "bar_without_metric")

    assert foo_frame0 is not None
    assert bar_frame0 is None
    assert foo_frame1 is None
    assert bar_frame1 is not None

    assert foo_frame0["metrics"]["sum_metric"] == float(foo_iters * x.numel())
    assert int(foo_frame0["metrics"]["count"]) == foo_iters
    assert "sum_metric" not in bar_frame1["metrics"]
    assert int(bar_frame1["metrics"]["count"]) == bar_iters


def test_trace(tmp_path: pathlib.Path, device: str):
    temp_file = tmp_path / "test_trace.chrome_trace"
    proton.start(str(temp_file.with_suffix("")), data="trace")

    @triton.jit
    def foo(x, y, size: tl.constexpr):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    with proton.scope("init"):
        x = torch.ones((1024, ), device=device, dtype=torch.float32)
        y = torch.zeros_like(x)

    with proton.scope("test"):
        foo[(1, )](x, y, x.size()[0], num_warps=4)

    proton.finalize()

    with temp_file.open() as f:
        data = json.load(f)
        trace_events = data["traceEvents"]
        assert trace_events[-1]["name"] == "foo"
        assert trace_events[-1]["args"]["call_stack"] == ["ROOT", "test", "foo"]


@pytest.mark.skipif(not is_cuda(), reason="Only CUDA backend supports metrics profiling in cudagraphs")
def test_trace_flexible_metrics_scope_ranges(tmp_path: pathlib.Path, device: str):

    @triton.jit
    def foo(x, y, size: tl.constexpr):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    x = torch.ones((1024, ), device=device, dtype=torch.float32)
    y = torch.zeros_like(x)
    temp_file = tmp_path / "test_trace_flexible_metrics_scope_ranges.chrome_trace"
    proton.start(str(temp_file.with_suffix("")), data="trace")

    with proton.scope("scope_3", metrics={"m3": 3.0}):
        with proton.scope("scope_2", metrics={"m2": 2.0}):
            with proton.scope("scope_1", metrics={"m1": 1.0}):
                foo[(1, )](x, y, x.size()[0], num_warps=4)
            with proton.scope("scope_4"):
                foo[(1, )](x, y, x.size()[0], num_warps=4)
            with proton.scope("scope_5"):
                foo[(1, )](x, y, x.size()[0], num_warps=4)
        with proton.scope("scope_6"):
            with proton.scope("scope_7"):
                foo[(1, )](x, y, x.size()[0], num_warps=4)

    proton.finalize()
    with temp_file.open() as f:
        trace_events = json.load(f)["traceEvents"]
    kernel_events = [event for event in trace_events if event.get("cat") == "kernel" and event["name"] == "foo"]
    metric_events = [event for event in trace_events if event.get("cat") == "metric"]
    scope_events = [event for event in trace_events if event.get("cat") == "scope"]
    flow_events = [event for event in trace_events if event.get("cat") == "flow"]

    assert (len(kernel_events), len(metric_events), len(scope_events), len(flow_events)) == (4, 3, 4, 8)

    assert {tuple(event["args"]["call_stack"])
            for event in kernel_events} == {
                ("ROOT", "scope_3", "scope_2", "scope_1", "foo"),
                ("ROOT", "scope_3", "scope_2", "scope_4", "foo"),
                ("ROOT", "scope_3", "scope_2", "scope_5", "foo"),
                ("ROOT", "scope_3", "scope_6", "scope_7", "foo"),
            }

    metric_by_name = {next(iter(event["args"]["metrics"])): event for event in metric_events}
    assert {
        name: (event["name"], tuple(event["args"]["call_stack"]), event["args"]["metrics"])
        for name, event in metric_by_name.items()
    } == {
        "m1": ("scope_1: <m1, 1.000000>", ("ROOT", "scope_3", "scope_2", "scope_1"), {"m1": "1.000000"}),
        "m2": ("scope_2: <m2, 2.000000>", ("ROOT", "scope_3", "scope_2"), {"m2": "2.000000"}),
        "m3": ("scope_3: <m3, 3.000000>", ("ROOT", "scope_3"), {"m3": "3.000000"}),
    }

    assert {tuple(event["args"]["call_stack"])
            for event in scope_events} == {
                ("ROOT", "scope_3", "scope_2", "scope_4"),
                ("ROOT", "scope_3", "scope_2", "scope_5"),
                ("ROOT", "scope_3", "scope_6"),
                ("ROOT", "scope_3", "scope_6", "scope_7"),
            }

    gpu_tid = kernel_events[0]["tid"]
    cpu_tid = metric_by_name["m1"]["tid"]
    flow_starts = {event["id"]: event for event in flow_events if event["ph"] == "s"}
    flow_finishes = {event["id"]: event for event in flow_events if event["ph"] == "f"}
    assert set(flow_starts) == set(flow_finishes)
    assert len(flow_starts) == 4
    assert all(event["name"] == "launch->kernel" and event["bp"] == "e" and event["tid"] == cpu_tid
               for event in flow_starts.values())
    assert all(event["name"] == "launch->kernel" and event["bp"] == "e" and event["tid"] == gpu_tid
               for event in flow_finishes.values())


def test_trace_flexible_metrics_no_kernel_anchor(tmp_path: pathlib.Path):
    temp_file = tmp_path / "test_trace_flexible_metrics_no_kernel_anchor.chrome_trace"
    proton.start(str(temp_file.with_suffix("")), data="trace")

    with proton.scope("metric_only", metrics={"foo": 1.0}):
        pass

    proton.finalize()
    with temp_file.open() as f:
        trace_events = json.load(f)["traceEvents"]
    assert len(trace_events) == 1
    assert (
        trace_events[0]["cat"],
        trace_events[0]["name"],
        trace_events[0]["args"]["call_stack"],
        trace_events[0]["args"]["metrics"],
    ) == ("metric", "metric_only: <foo, 1.000000>", ["ROOT", "metric_only"], {"foo": "1.000000"})


@pytest.mark.skipif(not is_cuda(), reason="Only CUDA backend supports cudagraph trace reconstruction")
def test_trace_cudagraph_graph_scope_ranges(tmp_path: pathlib.Path, device: str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    @triton.jit
    def foo(x, y, size: tl.constexpr):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    x = torch.ones((128, ), device=device, dtype=torch.float32)
    y = torch.zeros_like(x)
    metric_tensor = torch.tensor(1.0, device=device)

    foo[(1, )](x, y, x.numel(), num_warps=4)
    torch.cuda.synchronize(torch.device(device))

    def fn():
        with proton.scope("a"):
            with proton.scope("b"):
                with proton.scope("c", metrics={"m1": metric_tensor}):
                    foo[(1, )](x, y, x.numel(), num_warps=4)
                foo[(1, )](x, y, x.numel(), num_warps=4)
            foo[(1, )](x, y, x.numel(), num_warps=4)

    temp_file = tmp_path / "test_trace_cudagraph_graph_scope_ranges.chrome_trace"
    proton.start(str(temp_file.with_suffix("")), data="trace", context="shadow")

    # warmup
    fn()

    g = torch.cuda.CUDAGraph()
    with cuda_graph_without_gc(g):
        fn()

    with proton.scope("test0"):
        g.replay()

    proton.finalize()
    with temp_file.open() as f:
        trace_events = json.load(f)["traceEvents"]

    thread_name_events = [
        event for event in trace_events if event.get("ph") == "M" and event.get("name") == "thread_name"
    ]
    graph_tids = [event["tid"] for event in thread_name_events if event["args"]["name"].startswith("Graph: Stream ")]
    assert len(graph_tids) == 1
    graph_tid = graph_tids[0]

    graph_scope_events = [event for event in trace_events if event.get("cat") == "scope" and event["tid"] == graph_tid]
    assert {"<captured_at>", "a", "b", "c"}.issubset({event["name"] for event in graph_scope_events})
    assert not any(event.get("cat") == "metric" and event["tid"] == graph_tid for event in trace_events)

    replay_kernel_events = [
        event for event in trace_events
        if event.get("cat") == "kernel" and event.get("args", {}).get("call_stack", [])[:2] == ["ROOT", "test0"]
    ]
    foo_events = [event for event in replay_kernel_events if event["name"] == "foo"]
    metric_kernel_events = [event for event in replay_kernel_events if event["name"] == "<metric>"]
    metadata_kernel_events = [
        event for event in replay_kernel_events
        if any(frame == COMPUTE_METADATA_SCOPE_NAME for frame in event.get("args", {}).get("call_stack", []))
    ]

    assert len(foo_events) == 3
    assert {tuple(event["args"]["call_stack"])
            for event in foo_events} == {
                ("ROOT", "test0", "<captured_at>", "a", "b", "c", "foo"),
                ("ROOT", "test0", "<captured_at>", "a", "b", "foo"),
                ("ROOT", "test0", "<captured_at>", "a", "foo"),
            }
    assert len(metric_kernel_events) == 1
    assert metric_kernel_events[0]["args"]["call_stack"] == [
        "ROOT",
        "test0",
        "<captured_at>",
        "a",
        "b",
        "c",
        COMPUTE_METADATA_SCOPE_NAME,
        "<metric>",
    ]
    assert all(event["name"] != "foo" for event in metadata_kernel_events)

    test0_scope = next(
        event for event in trace_events
        if event.get("cat") == "scope" and event.get("args", {}).get("call_stack", []) == ["ROOT", "test0"])
    replay_gpu_tid = foo_events[0]["tid"]
    first_replay_kernel = min(replay_kernel_events, key=lambda event: event["ts"])
    flow_finish = next(event for event in trace_events
                       if event.get("cat") == "flow" and event["ph"] == "f" and event["name"] == "launch->kernel"
                       and event["tid"] == replay_gpu_tid and event["ts"] == first_replay_kernel["ts"])
    flow_start = next(event for event in trace_events
                      if event.get("cat") == "flow" and event["ph"] == "s" and event["id"] == flow_finish["id"])
    assert flow_start["tid"] == test0_scope["tid"]
    assert test0_scope["ts"] == flow_start["ts"] <= flow_finish["ts"]


@pytest.mark.parametrize("profile_kind,suffix", [("tree", ".hatchet"), ("trace", ".chrome_trace")],
                         ids=["tree", "trace"])
def test_multi_stream(profile_kind: str, suffix: str, tmp_path: pathlib.Path, device: str):

    @triton.jit
    def foo(x, y, size: tl.constexpr):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    temp_file = tmp_path / f"test_multi_stream{suffix}"
    device_obj = torch.device(device)
    x = torch.ones((1024, ), device=device_obj, dtype=torch.float32)
    outputs = [torch.zeros_like(x) for _ in range(2)]
    streams = [torch.cuda.Stream(device=device_obj) for _ in range(2)]
    scope_names = [f"stream_scope_{idx}" for idx in range(len(streams))]

    foo[(1, )](x, outputs[0], x.numel(), num_warps=4)
    torch.cuda.synchronize(device_obj)

    start_kwargs = {"data": "trace"} if profile_kind == "trace" else {}
    proton.start(str(temp_file.with_suffix("")), **start_kwargs)

    for scope_name, stream, output in zip(scope_names, streams, outputs):
        with torch.cuda.stream(stream):
            with proton.scope(scope_name):
                foo[(1, )](x, output, x.numel(), num_warps=4)

    for stream in streams:
        stream.synchronize()
    proton.finalize()

    with temp_file.open() as f:
        data = json.load(f)

    if profile_kind == "trace":
        assert "traceEvents" in data
        kernel_events = [event for event in data["traceEvents"] if event["name"] == "foo"]
        assert len(kernel_events) == len(scope_names)
        assert len({event["tid"] for event in kernel_events}) == len(scope_names)
        for scope_name in scope_names:
            matching_events = [event for event in kernel_events if scope_name in event["args"]["call_stack"]]
            assert len(matching_events) == 1
    else:
        root = data[0]
        scope_0 = next(child for child in root["children"] if child["frame"]["name"] == "stream_scope_0")
        scope_1 = next(child for child in root["children"] if child["frame"]["name"] == "stream_scope_1")
        assert len(scope_0["children"]) > 0
        assert len(scope_1["children"]) > 0
        assert scope_0["children"][0]["metrics"]["time (ns)"] > 0
        assert scope_1["children"][0]["metrics"]["time (ns)"] > 0


def test_scope_multiple_threads(tmp_path: pathlib.Path, device: str):
    temp_file = tmp_path / "test_scope_multiple_threads.hatchet"
    proton.start(str(temp_file.with_suffix("")))

    N = 50
    thread_names = ["threadA", "threadB"]

    def worker(prefix: str):
        for i in range(N):
            name = f"{prefix}_{i}"
            proton.enter_scope(name)
            torch.ones((1, ), device=device)
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


@pytest.mark.skipif(not is_cuda() and not is_hip(), reason="Only CUDA/HIP backend supports NVTX profiling")
@pytest.mark.parametrize("enable_nvtx", [None, True, False])
def test_nvtx_range_push_pop(enable_nvtx, fresh_knobs, tmp_path: pathlib.Path, device: str):
    if enable_nvtx is not None:
        fresh_knobs.proton.enable_nvtx = enable_nvtx
    temp_file = tmp_path / "test_nvtx_range_push_pop.hatchet"
    proton.start(str(temp_file.with_suffix("")))

    with proton.scope("proton_scope"):
        torch.cuda.nvtx.range_push("nvtx_range0")
        torch.cuda.nvtx.range_push("nvtx_range1")
        torch.ones((1, ), device=device)
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_pop()

    proton.finalize()

    with temp_file.open() as f:
        data = json.load(f)

    children = data[0]["children"]
    assert len(children) == 1
    proton_scope = children[0]
    assert proton_scope["frame"]["name"] == "proton_scope"
    assert len(proton_scope["children"]) == 1
    if enable_nvtx or enable_nvtx is None:
        nvtx_range0 = proton_scope["children"][0]
        assert nvtx_range0["frame"]["name"] == "nvtx_range0"
        assert len(nvtx_range0["children"]) == 1
        nvtx_range1 = nvtx_range0["children"][0]
        assert nvtx_range1["frame"]["name"] == "nvtx_range1"
        assert len(nvtx_range1["children"]) == 1
        kernel = nvtx_range1["children"][0]
    else:
        kernel = proton_scope["children"][0]
    assert "elementwise" in kernel["frame"]["name"]
    assert kernel["metrics"]["count"] == 1


def test_tensor_metrics_scope(tmp_path: pathlib.Path, device: str):
    temp_file = tmp_path / "test_tensor_metrics_scope.hatchet"
    proton.start(str(temp_file.with_suffix("")))

    x = torch.ones((10, 10), device=device, dtype=torch.float32)
    x_mean = x.mean()
    x_std = x.std()
    with proton.scope("test", metrics={"x_mean": x_mean, "x_std": x_std}):
        torch.randn((10, 10), device=device)
        torch.zeros_like(x)

    proton.finalize()

    with temp_file.open() as f:
        data = json.load(f)

    children = data[0]["children"]
    assert len(children) == 4
    # get the test frame
    test_frame = None
    for child in children:
        if child["frame"]["name"] == "test":
            test_frame = child
            break
    assert test_frame is not None
    assert test_frame["metrics"]["x_mean"] == 1.0
    assert test_frame["metrics"]["x_std"] == 0.0


def test_tensor_metrics_hook(tmp_path: pathlib.Path, device: str):
    temp_file = tmp_path / "test_tensor_metrics_hook.hatchet"

    def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        metric_value = torch.tensor(8.0, device=device)
        return {"name": "foo_test", "flops": metric_value}

    @triton.jit(launch_metadata=metadata_fn)
    def foo(x, size: tl.constexpr, y):
        offs = tl.arange(0, size)
        tl.store(y + offs, tl.load(x + offs))

    x = torch.ones((8, ), device=device, dtype=torch.float32)
    y = torch.zeros_like(x)

    proton.start(str(temp_file.with_suffix("")), hook="triton")
    foo[(1, )](x, x.numel(), y, num_warps=4)
    proton.finalize()

    with temp_file.open() as f:
        data = json.load(f)

    children = data[0]["children"]
    # metadata scope + foo_test
    assert len(children) == 2
    foo_test_frame = None
    for child in children:
        if child["frame"]["name"] == "foo_test":
            foo_test_frame = child
            break
    assert foo_test_frame is not None
    assert foo_test_frame["metrics"]["flops"] == 8.0


@pytest.mark.skipif(not is_cuda(), reason="Only CUDA backend supports metrics profiling in cudagraphs")
def test_tensor_metrics_cudagraph_hook(tmp_path: pathlib.Path, device: str):
    """
    Test triton kernels launched from metadata hooks and hook="triton"
    """
    owner_name = "metadata_owner_kernel"

    @triton.jit
    def metadata_helper_kernel(metric_value):
        tl.store(metric_value, 8.0)

    def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        metadata_helper_kernel[(1, )](args["metric_value"], num_warps=1)
        return {"name": owner_name, "flops": args["metric_value"], "bytes": args["bytes_value"]}

    @triton.jit(launch_metadata=metadata_fn)
    def metadata_owner_kernel(x, y, metric_value, bytes_value):
        tl.store(y, tl.load(x) + 1.0)

    x = torch.tensor([1.0], device=device)
    y = torch.empty_like(x)
    metric_value = torch.tensor([0.0], device=device)
    bytes_value = torch.tensor([64], device=device, dtype=torch.int64)

    temp_file = tmp_path / "test_tensor_metrics_cudagraph_hook.hatchet"
    session = proton.start(str(temp_file.with_suffix("")), context="shadow", hook="triton")

    metadata_owner_kernel[(1, )](x, y, metric_value, bytes_value, num_warps=1)
    metric_value.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata_owner_kernel[(1, )](x, y, metric_value, bytes_value, num_warps=1)

    with proton.scope("replay"):
        graph.replay()
    proton.finalize(session)

    with temp_file.open() as f:
        data = json.load(f)

    replay_frame = _find_frame_by_name(data[0], "replay")
    assert replay_frame is not None
    capture_frame = _find_frame_by_name(replay_frame, "<captured_at>")
    assert capture_frame is not None

    owner_frame = _find_frame_by_name(capture_frame, owner_name)
    metadata_root_frame = _find_frame_by_name(capture_frame, COMPUTE_METADATA_SCOPE_NAME)
    metadata_frame = None
    if metadata_root_frame is not None:
        metadata_frame = _find_frame_by_name(metadata_root_frame, owner_name)
    assert owner_frame is not None
    assert metadata_frame is not None
    assert owner_frame["metrics"]["flops"] == 8.0
    assert owner_frame["metrics"]["bytes"] == 64
    assert _find_frame_by_name(metadata_frame, "<metric>") is not None
    assert _find_frame_by_name(metadata_frame, "metadata_helper_kernel") is not None


@pytest.mark.skipif(not is_cuda(), reason="Only CUDA backend supports metrics profiling in cudagraphs")
def test_tensor_metrics_cudagraph(tmp_path: pathlib.Path, device: str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        x = args["x"]
        x_sum = x.sum()
        return {"name": "foo_test", "bytes": x.numel() * x.element_size(), "flops": x_sum}

    @triton.jit(launch_metadata=metadata_fn)
    def foo(x, y, z):
        tl.store(z, tl.load(y) + tl.load(x))

    def fn():
        with proton.scope("scope_a", metrics={"bytes": 4 * 4}):
            a = torch.ones((2, 2), device=device)
        with proton.metadata_state():
            a_sum = a.sum()
        with proton.scope("scope_b", metrics={"sum": a_sum}):
            b = torch.ones((2, 2), device=device)
        c = a + b
        foo[(1, )](a, b, c)
        with proton.metadata_state():
            d = torch.arange(4, device="cuda")
        with proton.scope("scope_d", metrics={"vec": d}):
            e = d * 2  # noqa: F841

    temp_file = pathlib.Path("./") / "test_tensor_metrics_cudagraph.hatchet"
    proton.start(str(temp_file.with_suffix("")), context="shadow", hook="triton")

    # warmup
    fn()

    # no kernels
    g = torch.cuda.CUDAGraph()
    with cuda_graph_without_gc(g):
        for _ in range(10):
            fn()

    with proton.scope("test0"):
        g.replay()

    proton.finalize()

    with temp_file.open() as f:
        data = json.load(f)

    children = data[0]["children"]
    test0_frame = None
    for child in children:
        if child["frame"]["name"] == "test0":
            test0_frame = child
            break
    assert test0_frame is not None
    capture_at_frame = _find_frame_by_name(test0_frame, "<captured_at>")
    assert capture_at_frame is not None

    foo_test_frame = None
    scope_a_frame = None
    scope_b_frame = None
    scope_d_frame = None
    for child in capture_at_frame["children"]:
        if child["frame"]["name"] == "foo_test":
            foo_test_frame = child
        if child["frame"]["name"] == "scope_a":
            scope_a_frame = child
        if child["frame"]["name"] == "scope_b":
            scope_b_frame = child
        if child["frame"]["name"] == "scope_d":
            scope_d_frame = child
    metadata_root_frame = _find_frame_by_name(capture_at_frame, COMPUTE_METADATA_SCOPE_NAME)
    assert metadata_root_frame is not None
    metadata_foo_frame = _find_frame_by_name(metadata_root_frame, "foo")
    assert metadata_foo_frame is not None
    assert _find_frame_by_name(metadata_foo_frame, "<metric>") is not None
    assert foo_test_frame is not None
    assert foo_test_frame["metrics"]["bytes"] == 160
    assert foo_test_frame["metrics"]["flops"] == 40
    assert foo_test_frame["metrics"]["count"] == 10
    assert scope_a_frame is not None
    assert scope_a_frame["metrics"]["bytes"] == 160
    assert "count" not in scope_a_frame["metrics"]
    assert scope_b_frame is not None
    assert scope_b_frame["metrics"]["sum"] == 40.0
    assert "count" not in scope_b_frame["metrics"]
    assert scope_d_frame is not None
    assert scope_d_frame["metrics"]["vec"] == [0, 10, 20, 30]


@pytest.mark.skipif(not is_cuda(), reason="Only CUDA backend supports metrics profiling in cudagraphs")
def test_tensor_metrics_cudagraph_deactivate(tmp_path: pathlib.Path, device: str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    def fn(session):
        proton.deactivate(session)
        with proton.scope("scope_b", metrics={"sum": 4}):
            b = torch.ones((2, 2), device=device)
        proton.activate(session)
        c = b * 2  # noqa: F841

    temp_file = tmp_path / "test_tensor_metrics_cudagraph_deactivate.hatchet"
    session = proton.start(str(temp_file.with_suffix("")), context="shadow", hook="triton")

    # warmup
    fn(session)

    # no kernels
    g = torch.cuda.CUDAGraph()
    with cuda_graph_without_gc(g):
        for _ in range(10):
            fn(session)

    with proton.scope("test0"):
        g.replay()

    proton.finalize()

    # only a single kernel b * 2
    with temp_file.open() as f:
        data = json.load(f)
        children = data[0]["children"]
        test0_frame = None
        for child in children:
            if child["frame"]["name"] == "test0":
                test0_frame = child
                break
        assert test0_frame is not None
        capture_at_frame = _find_frame_by_name(test0_frame, "<captured_at>")
        assert capture_at_frame is not None
        scope_b_frame = None
        c_frame = None
        for child in capture_at_frame["children"]:
            if child["frame"]["name"] == "scope_b":
                scope_b_frame = child
            if "elementwise" in child["frame"]["name"]:
                c_frame = child
        assert scope_b_frame is None
        assert c_frame is not None
        assert c_frame["metrics"]["count"] == 10


@pytest.mark.skipif(not is_cuda(), reason="Only CUDA backend supports metrics profiling in cudagraphs")
def test_tensor_metrics_multi_device_cudagraph(tmp_path: pathlib.Path):
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least two CUDA devices")

    devices = [torch.device(f"cuda:{i}") for i in range(2)]
    streams = []
    for device in devices:
        with torch.cuda.device(device):
            streams.append(torch.cuda.Stream(device=device))

    def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        x = args["x"]
        x_sum = x.sum()
        device_idx = x.device.index
        return {"name": f"foo_test_{device_idx}", "bytes": x.numel() * x.element_size(), "flops": x_sum}

    @triton.jit(launch_metadata=metadata_fn)
    def foo(x, y, z):
        tl.store(z, tl.load(y) + tl.load(x))

    def run_on_device(device_id):
        with proton.scope(f"scope_a_{device_id}", metrics={"bytes": 4 * 4}):
            a = torch.ones((2, 2), device=f"cuda:{device_id}")
        with proton.metadata_state():
            a_sum = a.sum()
        with proton.scope(f"scope_b_{device_id}", metrics={"sum": a_sum}):
            b = torch.ones((2, 2), device=f"cuda:{device_id}")
        c = a + b
        foo[(1, )](a, b, c)

    temp_file = tmp_path / "test_tensor_metrics_multi_device_cudagraph.hatchet"
    proton.start(str(temp_file.with_suffix("")), context="shadow", hook="triton")

    graphs = []
    for device, stream in zip(devices, streams):
        with torch.cuda.device(device):
            torch.cuda.set_stream(stream)
            # warmup
            run_on_device(device.index)
            # graph capture
            g = torch.cuda.CUDAGraph()
            with cuda_graph_without_gc(g, stream=stream):
                for _ in range(10):
                    run_on_device(device.index)
        graphs.append((device, stream, g))

    for device, stream, graph in graphs:
        with torch.cuda.device(device):
            torch.cuda.set_stream(stream)
            with proton.scope(f"test_device_{device.index}"):
                graph.replay()

    proton.finalize()

    with temp_file.open() as f:
        data = json.load(f)

    children = data[0]["children"]
    for device in devices:
        device_name = f"test_device_{device.index}"
        launch_frame = next((child for child in children if child["frame"]["name"] == device_name), None)
        assert launch_frame is not None
        capture_at_frame = _find_frame_by_name(launch_frame, "<captured_at>")
        assert capture_at_frame is not None

        foo_frame = None
        scope_a_frame = None
        scope_b_frame = None
        for child in capture_at_frame["children"]:
            if child["frame"]["name"] == f"foo_test_{device.index}":
                foo_frame = child
            if child["frame"]["name"] == f"scope_a_{device.index}":
                scope_a_frame = child
            if child["frame"]["name"] == f"scope_b_{device.index}":
                scope_b_frame = child

        assert foo_frame is not None
        assert scope_a_frame is not None
        assert scope_b_frame is not None
        assert foo_frame["metrics"]["bytes"] == 160
        assert foo_frame["metrics"]["flops"] == 40
        assert foo_frame["metrics"]["device_id"] == str(device.index)
        assert scope_a_frame["metrics"]["bytes"] == 160
        assert scope_b_frame["metrics"]["sum"] == 40.0

    assert len(data) > 1
    cuda_devices = data[1].get("CUDA", {})
    assert len(cuda_devices) >= 2


@pytest.mark.parametrize("buffer_size", [256 * 1024, 64 * 1024 * 1024])
@pytest.mark.parametrize("data_format", ["hatchet_msgpack", "hatchet"])
def test_periodic_flushing(tmp_path, fresh_knobs, data_format, buffer_size, device: str):
    fresh_knobs.proton.profile_buffer_size = buffer_size
    temp_file = tmp_path / f"test_periodic_flushing.{data_format}"
    session = proton.start(str(temp_file.with_suffix("")), mode=f"periodic_flushing:format={data_format}")

    for i in range(5000):
        if i != 0 and i % 500 == 0:
            proton.data.advance_phase(session=session)
        with proton.scope(f"test_{i}", metrics={"count": 1}):
            torch.zeros((100), device=device)

    proton.finalize(output_format=data_format)

    # Find all *.hatchet files under the directory `tmp_path`
    import glob
    import msgpack
    hatchet_files = glob.glob(str(tmp_path / f"*.{data_format}"))
    assert len(hatchet_files) == 10
    num_scopes = 0
    for hatchet_file in hatchet_files:
        if data_format == "hatchet_msgpack":
            with open(hatchet_file, "rb") as f:
                data = msgpack.load(f, raw=False, strict_map_key=False)
        else:
            with open(hatchet_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        assert len(data[0]["children"]) == 500
        assert data[0]["children"][0]["metrics"]["count"] == 1
        assert data[0]["children"][0]["frame"]["name"].startswith("test_")
        assert data[0]["children"][0]["children"][0]["metrics"]["time (ns)"] > 0
        num_scopes += len(data[0]["children"])
    assert num_scopes == 5000


@pytest.mark.skipif(not is_cuda(), reason="Only CUDA backend supports metrics profiling in cudagraphs")
@pytest.mark.parametrize("buffer_size", [256 * 1024, 64 * 1024 * 1024])
@pytest.mark.parametrize("data_format", ["hatchet_msgpack", "hatchet"])
def test_periodic_flushing_cudagraph(tmp_path, fresh_knobs, data_format, buffer_size, device: str):
    fresh_knobs.proton.profile_buffer_size = buffer_size
    temp_file = tmp_path / f"test_periodic_flushing_cudagraph.{data_format}"
    session = proton.start(str(temp_file.with_suffix("")), mode=f"periodic_flushing:format={data_format}",
                           hook="triton")

    def metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        x = args["x"]
        x_sum = x.sum()
        return {"name": "foo_test", "bytes": x.numel() * x.element_size(), "flops": x_sum}

    @triton.jit(launch_metadata=metadata_fn)
    def foo(x, y, z):
        tl.store(z, tl.load(y) + tl.load(x))

    def fn():
        with proton.scope("scope_a", metrics={"bytes": 4 * 4}):
            a = torch.ones((2, 2), device=device)
        c = a + a
        foo[(1, )](a, a, c)

    # warmup
    fn()

    # Recycle GPU memory before graph capture to reduce memory pressure
    # when running with parallel test workers (-n 8).
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    # no kernels
    g = torch.cuda.CUDAGraph()
    with cuda_graph_without_gc(g):
        fn()

    test_iterations = 500
    with proton.scope("test0"):
        for i in range(test_iterations):
            if i != 0 and i % (test_iterations // 10) == 0:
                proton.data.advance_phase(session=session)
            g.replay()

    proton.finalize(output_format=data_format)

    # Find all *.hatchet files under the directory `tmp_path`
    import glob
    import msgpack
    hatchet_files = glob.glob(str(tmp_path / f"*.{data_format}"))
    assert len(hatchet_files) == 10
    for hatchet_file in hatchet_files:
        if data_format == "hatchet_msgpack":
            with open(hatchet_file, "rb") as f:
                data = msgpack.load(f, raw=False, strict_map_key=False)
        else:
            with open(hatchet_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        capture_frame = None
        for child in data[0]["children"]:
            if child["frame"]["name"] == "test0":
                capture_frame = _find_frame_by_name(child, "<captured_at>")
                break
        assert capture_frame is not None
        scope_a_frame = None
        foo_test_frame = None
        for child in capture_frame["children"]:
            if child["frame"]["name"] == "scope_a":
                scope_a_frame = child
            if child["frame"]["name"] == "foo_test":
                foo_test_frame = child
        assert scope_a_frame is not None
        assert foo_test_frame is not None
        assert scope_a_frame["metrics"]["bytes"] == test_iterations / 10 * 16
        assert foo_test_frame["metrics"]["bytes"] == test_iterations / 10 * 16
        assert foo_test_frame["metrics"]["flops"] == test_iterations / 10 * 4


@pytest.mark.skipif(not is_blackwell(), reason="HW trace is only supported on Blackwell GPUs")
def test_hw_trace(fresh_knobs, tmp_path: pathlib.Path, device: str):
    fresh_knobs.proton.enable_hw_trace = True
    temp_file = tmp_path / "test_hw_trace.hatchet"
    proton.start(str(temp_file.with_suffix("")), hook="triton")

    with proton.scope("init"):
        x = torch.ones((1024, ), device=device, dtype=torch.float32)  # noqa: F841

    proton.finalize()

    with temp_file.open() as f:
        data = json.load(f)
    kernel_frame = data[0]["children"][0]["children"][0]
    assert "elementwise" in kernel_frame["frame"]["name"]
    assert kernel_frame["metrics"]["time (ns)"] > 0
