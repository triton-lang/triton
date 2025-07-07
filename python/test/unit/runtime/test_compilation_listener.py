import triton
import triton.language as tl

from triton.backends.compiler import GPUTarget
from triton.knobs import CompileTimes
from triton.compiler.compiler import ASTSource, IRSource
from triton._C.libtriton import ir

from typing import Any, Union

import torch


@triton.jit
def cumsum_kernel(ptr):
    block = ptr + tl.arange(0, 4)
    x = tl.load(block)
    tl.store(block, tl.cumsum(x, 0))


def test_compile_stats(device: str, fresh_knobs_except_libraries: Any, fresh_triton_cache: str) -> None:
    captured: Union[tuple[Union[ASTSource, IRSource], dict[str, Any], dict[str, Any], CompileTimes, bool], None] = None

    def compile_listener(src: Union[ASTSource, IRSource], metadata: dict[str, str], metadata_group: dict[str, Any],
                         times: CompileTimes, cache_hit: bool) -> None:
        nonlocal captured
        assert captured is None
        captured = (src, metadata, metadata_group, times, cache_hit)

    fresh_knobs_except_libraries.compilation.listener = compile_listener

    x = torch.randn(4, device=device)
    cumsum_kernel[(1, )](x)

    assert captured is not None

    # No cache hit at first
    assert not captured[4]

    # Expected metadata
    assert len(captured[1]["hash"]) > 0
    assert isinstance(captured[1]["target"], GPUTarget)

    # It in fact did take some time to do compilation
    assert captured[3].ir_initialization > 0
    assert captured[3].total_lowering > 0
    assert captured[3].store_results > 0
    assert captured[3].total > 0

    # Now lets create a new instance of the same kernel to pick up cache_hit=True
    cumsum_kernel.device_caches.clear()
    captured = None
    cumsum_kernel[(1, )](x)

    assert captured is not None
    # Cache hit!
    assert captured[4]

    # Expected metadata
    assert len(captured[1]["hash"]) > 0
    assert isinstance(captured[1]["target"], GPUTarget)

    # It in fact did take some time to do compilation
    assert captured[3].ir_initialization > 0
    assert captured[3].total_lowering == 0
    assert captured[3].store_results == 0
    assert captured[3].total > 0


def test_pass_listener(device: str, fresh_knobs_except_libraries: Any, fresh_triton_cache: str) -> None:
    pipeline_strs = []

    def pass_listener(manager) -> bool:
        assert isinstance(manager, ir.pass_manager)
        pipeline_strs.append(manager.get_pipeline_str())
        return True

    fresh_knobs_except_libraries.compilation.pass_listener = pass_listener

    x = torch.randn(4, device=device)
    cumsum_kernel[(1, )](x)

    # We expect at least a run for TTIR, TTGIR, and LLIR. However backends
    # can (and do) run more than three pipelines.
    assert len(pipeline_strs) >= 3, pipeline_strs

    # This callback is generally called before passes are populated, so there
    # isn't anything to check except "get_pipeline_str() returned something."
    assert all(pipeline_strs), pipeline_strs

    pipeline_strs.clear()
    cumsum_kernel[(1, )](x)

    # `pass_listener` is a compilation detail, so it should not trigger on a
    # cache hit.
    assert not pipeline_strs

    cumsum_kernel.device_caches.clear()
    cumsum_kernel[(1, )](x)
    assert not pipeline_strs
