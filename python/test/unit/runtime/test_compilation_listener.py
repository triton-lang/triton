import triton
import triton.language as tl

from triton.backends.compiler import GPUTarget
from triton.compiler.compiler import ASTSource, IRSource

from triton.runtime.config import TritonConfig, CompileTimes
from typing import Any, Union

import torch


def get_add_kernel() -> triton.JITFunction:
    # We need to define the kernel in a method so that we can test getting a
    # 'cache_hit=True' callback below.

    @triton.jit
    def add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    return add_kernel


def test_compile_stats(device: str, fresh_triton_cache: str) -> None:
    captured: tuple[Union[ASTSource, IRSource], dict[str, Any], CompileTimes, bool] | None = None

    def compile_listener(src: Union[ASTSource, IRSource], metadata: dict[str, Any], times: CompileTimes,
                         cache_hit: bool) -> None:
        nonlocal captured
        assert captured is None
        captured = (src, metadata, times, cache_hit)

    TritonConfig.compilation_listener = compile_listener

    kernel = get_add_kernel()
    x = torch.randn(4, device=device)
    y = torch.randn(4, device=device)
    out = torch.zeros_like(x)
    kernel[(4, )](x, y, out, 4, 4)

    assert captured is not None

    # No cache hit at first
    assert not captured[3]

    # Expected metadata
    assert len(captured[1]["hash"]) > 0
    assert isinstance(captured[1]["target"], GPUTarget)

    # It in fact did take some time to do compilation
    assert captured[2].prologue > 0
    assert captured[2].total_lowering > 0
    assert captured[2].epilogue > 0
    assert captured[2].total > 0

    # Now lets create a new instance of the same kernel to pick up cache_hit=True
    captured = None
    kernel = get_add_kernel()
    kernel[(4, )](x, y, out, 4, 4)

    assert captured is not None
    # Cache hit!
    assert captured[3]

    # Expected metadata
    assert len(captured[1]["hash"]) > 0
    assert isinstance(captured[1]["target"], GPUTarget)

    # It in fact did take some time to do compilation
    assert captured[2].prologue > 0
    assert captured[2].total_lowering == 0
    assert captured[2].epilogue == 0
    assert captured[2].total > 0
