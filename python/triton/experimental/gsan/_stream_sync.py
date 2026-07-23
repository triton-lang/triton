from __future__ import annotations

import functools
from dataclasses import dataclass

import triton
import triton.language as tl

from ._allocator import get_device_rank, get_runtime_state_layout
from ._utils import uint8_cuda_tensor_from_ptr


@dataclass(frozen=True)
class _RuntimeStateLayout:
    thread_state_region: object
    thread_state_stride_bytes: int
    thread_state_header_size_bytes: int
    num_sms: int
    num_threads: int


@dataclass
class _LaunchStreamState:
    clocks: object
    next_kernel_id: int = 0


@functools.lru_cache()
def _runtime_state_layout(runtime_state_device: int, access_device: int) -> _RuntimeStateLayout:
    layout = get_runtime_state_layout(runtime_state_device)
    region_size = layout["thread_state_stride_bytes"] * layout["num_sms"]
    thread_state_region = uint8_cuda_tensor_from_ptr(layout["thread_state_base_ptr"], region_size, access_device)
    return _RuntimeStateLayout(
        thread_state_region=thread_state_region,
        thread_state_stride_bytes=layout["thread_state_stride_bytes"],
        thread_state_header_size_bytes=layout["thread_state_header_size_bytes"],
        num_sms=layout["num_sms"],
        num_threads=layout["num_threads"],
    )


@functools.lru_cache()
def _launch_stream_state(device: int, stream: int) -> _LaunchStreamState:
    import torch

    layout = _runtime_state_layout(get_device_rank(device), device)
    cuda_stream = torch.cuda.ExternalStream(stream, device=device) if stream else torch.cuda.default_stream(device)
    with torch.cuda.device(device), torch.cuda.stream(cuda_stream):
        clocks = torch.zeros((3, layout.num_threads), dtype=torch.int32, device=device)
    return _LaunchStreamState(clocks=clocks)


def get_launch_stream_clock(device: int, stream: int):
    """Return a stream's triple-buffered clocks and monotonically increasing launch ID."""
    state = _launch_stream_state(device, stream)
    kernel_id = state.next_kernel_id
    state.next_kernel_id += 1
    return state.clocks, kernel_id


@triton.jit(do_not_specialize=["rank", "epoch"])
def _synchronize_process_group_barrier_kernel(counters, rank, epoch, WORLD_SIZE: tl.constexpr):
    tl.atomic_xchg(counters + rank, epoch, sem="release", scope="sys")
    for peer in tl.static_range(WORLD_SIZE):
        tl.atomic_poll(counters + peer, epoch, sem="acquire", scope="sys")


def synchronize_process_group_barrier(counters, rank: int, epoch: int, world_size: int) -> None:
    """Synchronize all ranks through an instrumented system-scope atomic."""
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.instrumentation_mode = "gsan"
        _synchronize_process_group_barrier_kernel[(1, )](counters, rank, epoch, WORLD_SIZE=world_size, num_warps=1)
