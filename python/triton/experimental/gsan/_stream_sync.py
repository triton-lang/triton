from __future__ import annotations

import functools
from dataclasses import dataclass
from contextlib import contextmanager

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
    table: object
    pointers: object
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


@functools.cache
def _clock_mem_pool(device: int):
    import torch

    with torch.cuda.device(device):
        return torch.cuda.MemPool()


@contextmanager
def _clock_storage(device: int, stream: int = 0):
    import torch

    cuda_stream = torch.cuda.ExternalStream(stream, device=device) if stream else torch.cuda.default_stream(device)
    with torch.cuda.device(device), torch.cuda.stream(cuda_stream), torch.cuda.use_mem_pool(_clock_mem_pool(device)):
        yield


def _make_launch_table(entries, waits, outputs, device):
    """Pack the fixed-width, five-u64 LaunchState ABI into device memory."""
    import torch

    pointer_values = [ptr for entry, wait in zip(entries, waits) for ptr in (*entry, *wait)]
    pointers = torch.tensor(pointer_values, dtype=torch.int64, device=device)
    rows = []
    offset = 0
    for entry, wait, output in zip(entries, waits, outputs):
        entry_ptr = pointers.data_ptr() + offset * 8 if entry else 0
        offset += len(entry)
        wait_ptr = pointers.data_ptr() + offset * 8 if wait else 0
        offset += len(wait)
        rows.append((entry_ptr, len(entry), wait_ptr, len(wait), output))
    table = torch.tensor(rows, dtype=torch.int64, device=device)
    return table, pointers


@functools.cache
def _launch_stream_state(device: int, stream: int) -> _LaunchStreamState:
    import torch

    layout = _runtime_state_layout(get_device_rank(device), device)
    with _clock_storage(device, stream):
        clocks = torch.zeros((3, layout.num_threads), dtype=torch.int32, device=device)
        ptrs = [clocks[i].data_ptr() for i in range(3)]
        entries, waits, outputs = [], [], []
        for slot in range(3):
            for pdl in (False, True):
                entries.append([ptrs[(slot + (1 if pdl else 2)) % 3]])
                waits.append([ptrs[(slot + 2) % 3]] if pdl else [])
                outputs.append(ptrs[slot])
        table, pointers = _make_launch_table(entries, waits, outputs, device)
    return _LaunchStreamState(clocks=clocks, table=table, pointers=pointers)


def get_launch_state(device: int, stream: int, launch_pdl: bool = False):
    """Reserve a stream launch and return its immutable descriptor table/index."""
    state = _launch_stream_state(device, stream)
    kernel_id = state.next_kernel_id
    state.next_kernel_id += 1
    return state.table, 2 * (kernel_id % 3) + launch_pdl


def _next_stream_clocks(device: int, stream: int):
    state = _launch_stream_state(device, stream)
    slot = state.next_kernel_id % 3
    return state.clocks[(slot + 2) % 3], state.clocks[slot]


def _reserve_stream_clocks(device: int, stream: int):
    clocks = _next_stream_clocks(device, stream)
    _launch_stream_state(device, stream).next_kernel_id += 1
    return clocks


def _current_stream_clock(device: int, stream: int):
    state = _launch_stream_state(device, stream)
    return state.clocks[(state.next_kernel_id - 1) % 3]


def _reset_caches() -> None:
    _launch_stream_state.cache_clear()
    _runtime_state_layout.cache_clear()


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
