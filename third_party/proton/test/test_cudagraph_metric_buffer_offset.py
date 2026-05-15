"""Regression coverage for Proton CUDA graph metric replay offsets."""

import gc
import json
import pathlib
from contextlib import contextmanager
from typing import NamedTuple

import pytest
import torch
import triton
import triton.language as tl
import triton.profiler as proton
from triton._internal_testing import is_cuda


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
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        with torch.cuda.graph(*args, **kwargs) as graph:
            yield graph
    finally:
        if gc_was_enabled:
            gc.enable()


@pytest.mark.skipif(not is_cuda(), reason="Only CUDA backend supports metrics profiling in cudagraphs")
def test_cudagraph_metric_queue_uses_live_buffer_offset(tmp_path: pathlib.Path, device: str):
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)

    def stale_metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        return {"name": "stale_metric_owner", "sum_metric": args["x"].sum()}

    def profiled_metadata_fn(grid: tuple, metadata: NamedTuple, args: dict):
        return {"name": "profiled_metric_owner", "sum_metric": args["x"].sum()}

    @triton.jit(launch_metadata=stale_metadata_fn)
    def stale_kernel(x, y):
        tl.store(y, tl.load(x) + 1.0)

    @triton.jit(launch_metadata=profiled_metadata_fn)
    def profiled_kernel(x, y):
        tl.store(y, tl.load(x) + 2.0)

    x = torch.ones((2, 2), device=device)
    y = torch.empty_like(x)

    # Compile before capture so the repro isolates graph metric replay state.
    stale_kernel[(1, )](x, y)
    profiled_kernel[(1, )](x, y)
    torch.cuda.synchronize()

    temp_file = tmp_path / "test_cudagraph_metric_queue_uses_live_buffer_offset.hatchet"
    session = proton.start(str(temp_file.with_suffix("")), context="shadow", hook="triton")
    try:
        stale_graph = torch.cuda.CUDAGraph()
        with cuda_graph_without_gc(stale_graph):
            stale_kernel[(1, )](x, y)

        profiled_graph = torch.cuda.CUDAGraph()
        with cuda_graph_without_gc(profiled_graph):
            profiled_kernel[(1, )](x, y)

        # These metadata-copy kernels still execute on the GPU, but inactive
        # Proton does not queue host-side pending graph metric state for them.
        proton.deactivate(session)
        stale_graph.replay()
        torch.cuda.synchronize()

        proton.activate(session)
        with proton.scope("profiled_replay"):
            profiled_graph.replay()
        torch.cuda.synchronize()
    finally:
        proton.finalize(session)

    with temp_file.open() as f:
        data = json.load(f)

    replay_frame = _find_frame_by_name(data[0], "profiled_replay")
    assert replay_frame is not None
    capture_frame = _find_frame_by_name(replay_frame, "<captured_at>")
    assert capture_frame is not None

    stale_frame = _find_frame_by_name(capture_frame, "stale_metric_owner")
    profiled_frame = _find_frame_by_name(capture_frame, "profiled_metric_owner")
    assert stale_frame is None
    assert profiled_frame is not None
    assert profiled_frame["metrics"]["sum_metric"] == float(x.numel())
