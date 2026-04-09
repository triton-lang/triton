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


def _check_compatible_runtime_state_layout(lhs: _RuntimeStateLayout, rhs: _RuntimeStateLayout) -> None:
    if (lhs.thread_state_stride_bytes != rhs.thread_state_stride_bytes
            or lhs.thread_state_header_size_bytes != rhs.thread_state_header_size_bytes or lhs.num_sms != rhs.num_sms
            or lhs.num_threads != rhs.num_threads):
        raise RuntimeError("GSan runtime state layout mismatch across synchronized devices.")


@triton.jit
def _synchronize_process_group_barrier_kernel(local_thread_state_region, peer_thread_state_regions, stride_bytes,
                                              num_sms, num_threads, header_bytes, BLOCK_SIZE: tl.constexpr,
                                              N_PEERS: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_threads
    max_clocks = tl.full([BLOCK_SIZE], 0, tl.uint16)

    for peer in tl.static_range(N_PEERS):
        peer_thread_state_region = peer_thread_state_regions[peer]
        # Each rank first collapses its local per-SM clocks, so every SM on a peer
        # carries the same vector clock at this point. Reading SM0 is sufficient.
        vector_clock_ptr = (peer_thread_state_region + header_bytes).to(tl.pointer_type(tl.uint16))
        peer_clocks = tl.load(vector_clock_ptr + offsets, mask=mask, other=0)
        max_clocks = tl.maximum(peer_clocks, max_clocks)

    for sm in range(num_sms):
        vector_clock_ptr = (local_thread_state_region + sm * stride_bytes + header_bytes).to(tl.pointer_type(tl.uint16))
        tl.store(vector_clock_ptr + offsets, max_clocks, mask=mask)


def synchronize_launch_stream(device: int) -> None:
    """This models the implicit synchronization between kernel launches.

    To do this, we compute the elementwise maximum of all SMs' vector clocks, and set every SM's clock to this starting point.

    This makes all reads and writes transitively visible to other threads.
    """
    layout = _runtime_state_layout(device, device)
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


def synchronize_process_group_barrier(device: int, peer_devices: tuple[int, ...]) -> None:
    """Join vector clocks across the devices participating in a process-group barrier.

    The peer runtime-state mappings are already imported by symmetric-memory rendezvous.
    This helper computes the elementwise maximum of all participating devices' per-SM
    vector clocks and writes that join back into the local device's per-SM state.
    """
    if not peer_devices:
        return

    local_layout = _runtime_state_layout(device, device)
    peer_regions = []
    for peer_device in peer_devices:
        peer_layout = _runtime_state_layout(peer_device, device)
        _check_compatible_runtime_state_layout(local_layout, peer_layout)
        peer_regions.append(peer_layout.thread_state_region)

    BLOCK_SIZE = 128
    grid = (triton.cdiv(local_layout.num_threads, BLOCK_SIZE), 1, 1)
    with _compile_without_gsan():
        _synchronize_process_group_barrier_kernel[grid](
            local_layout.thread_state_region,
            tuple(peer_regions),
            local_layout.thread_state_stride_bytes,
            local_layout.num_sms,
            local_layout.num_threads,
            local_layout.thread_state_header_size_bytes,
            BLOCK_SIZE=BLOCK_SIZE,
            N_PEERS=len(peer_regions),
            num_warps=1,
        )
