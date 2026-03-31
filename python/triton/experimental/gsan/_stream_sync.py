from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass

import triton
import triton.language as tl

from ._allocator import get_runtime_state_layout
from ._utils import uint8_cuda_tensor_from_ptr


@dataclass(frozen=True)
class _RuntimeStateLayout:
    thread_state_region: object
    thread_state_stride_bytes: int
    thread_state_header_size_bytes: int
    num_sms: int
    num_threads: int


@functools.lru_cache()
def _runtime_state_layout(device: int) -> _RuntimeStateLayout:
    layout = get_runtime_state_layout(device)
    region_size = layout["thread_state_stride_bytes"] * layout["num_sms"]
    thread_state_region = uint8_cuda_tensor_from_ptr(layout["thread_state_base_ptr"], region_size, device)
    return _RuntimeStateLayout(
        thread_state_region=thread_state_region,
        thread_state_stride_bytes=layout["thread_state_stride_bytes"],
        thread_state_header_size_bytes=layout["thread_state_header_size_bytes"],
        num_sms=layout["num_sms"],
        num_threads=layout["num_threads"],
    )


@contextmanager
def _compile_without_gsan():
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.instrumentation_mode = ""
        yield


@functools.lru_cache()
def _compiled_sync_kernel(device: int, stride_bytes: int, num_sms: int, num_threads: int, header_bytes: int,
                          BLOCK_SIZE):
    with _compile_without_gsan():
        return _synchronize_vector_clocks_kernel.warmup(
            triton.MockTensor(tl.uint8),
            stride_bytes,
            num_sms,
            num_threads,
            header_bytes,
            BLOCK_SIZE=BLOCK_SIZE,
            grid=(1, ),
            num_warps=1,
        )


@triton.jit
def _synchronize_vector_clocks_kernel(thread_state_region, stride_bytes, num_sms, num_threads, header_bytes,
                                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_threads
    max_clocks = tl.full([BLOCK_SIZE], 0, tl.uint16)

    for sm in range(num_sms):
        vector_clock_ptr = (thread_state_region + sm * stride_bytes + header_bytes).to(tl.pointer_type(tl.uint16))
        thread_clocks = tl.load(vector_clock_ptr + offsets, mask=mask, other=0)
        max_clocks = tl.maximum(thread_clocks, max_clocks)

    for sm in range(num_sms):
        vector_clock_ptr = (thread_state_region + sm * stride_bytes + header_bytes).to(tl.pointer_type(tl.uint16))
        tl.store(vector_clock_ptr + offsets, max_clocks, mask=mask)


def synchronize_launch_stream(device: int) -> None:
    """This models the implicit synchronization between kernel launches.

    To do this, we compute the elementwise maximum of all SMs' vector clocks, and set every SM's clock to this starting point.

    This makes all reads and writes transitively visible to other threads.
    """
    layout = _runtime_state_layout(device)
    BLOCK_SIZE = 128
    grid = (triton.cdiv(layout.num_threads, BLOCK_SIZE), 1, 1)
    kernel = _compiled_sync_kernel(
        device,
        layout.thread_state_stride_bytes,
        layout.num_sms,
        layout.num_threads,
        layout.thread_state_header_size_bytes,
        BLOCK_SIZE,
    )
    kernel[grid](
        layout.thread_state_region,
        layout.thread_state_stride_bytes,
        layout.num_sms,
        layout.num_threads,
        layout.thread_state_header_size_bytes,
        BLOCK_SIZE,
    )
