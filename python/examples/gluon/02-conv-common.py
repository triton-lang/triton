import torch

import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier
from triton.experimental.gluon.language.nvidia.blackwell import clc

TORCH_GEMM_DTYPE = torch.bfloat16
GL_GEMM_DTYPE = gl.bfloat16


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def normalize_2d(value, name):
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"{name} must be an int or length-2 tuple/list, got {value!r}")


def maybe_pad_channel_dims_for_tma(*tensors, alignment_bytes=16):
    """Pad tensor channel dimensions so TMA-visible strides are aligned."""
    if not tensors:
        raise ValueError("Expected at least one tensor to pad")

    elem_bytes = tensors[0].element_size()
    if alignment_bytes % elem_bytes != 0:
        raise ValueError(f"alignment_bytes={alignment_bytes} must be divisible by element size {elem_bytes}")

    orig_channels = tensors[0].shape[-1]
    for tensor in tensors[1:]:
        if tensor.element_size() != elem_bytes:
            raise ValueError("All tensors must have the same element size")
        if tensor.shape[-1] != orig_channels:
            raise ValueError("All tensors must have the same channel dimension")

    channel_alignment = alignment_bytes // elem_bytes
    padded_channels = triton.cdiv(orig_channels, channel_alignment) * channel_alignment
    if padded_channels == orig_channels:
        return tensors[0] if len(tensors) == 1 else tensors

    padded_tensors = []
    for tensor in tensors:
        padded = tensor.new_zeros((*tensor.shape[:-1], padded_channels))
        padded[..., :orig_channels] = tensor
        padded_tensors.append(padded.contiguous())

    return padded_tensors[0] if len(padded_tensors) == 1 else tuple(padded_tensors)


def ensure_tma_compatible_strides(tensor, alignment_bytes=16):
    """Ensure all outer strides are 16-byte aligned for TMA descriptors."""
    elem_bytes = tensor.element_size()
    for stride in tensor.stride()[:-1]:
        if (stride * elem_bytes) % alignment_bytes != 0:
            out = torch.empty(
                tensor.shape,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            out.copy_(tensor)
            return out
    return tensor


@gluon.constexpr_function
def get_transposed_cga_layout(cga_layout):
    return tuple((basis[1], basis[0]) for basis in cga_layout)


@gluon.constexpr_function
def get_operand_cga_layout(cga_layout, op_idx):
    assert op_idx in (0, 1)
    if not cga_layout:
        return cga_layout
    assert cga_layout[0] == (1, 0)
    first = (1, 0) if op_idx == 0 else (0, 1)

    def broadcast(basis):
        return (basis[0], 0) if op_idx == 0 else (0, 2 * basis[1])

    return (first, *map(broadcast, cga_layout[1:]))


def validate_2cta_m_split(cga_layout):
    if cga_layout not in ((), ((1, 0), )):
        raise ValueError(f"Only single-CTA or 2-CTA M-split layouts are supported, got {cga_layout!r}")


def get_gluon_dtype_for_tensor(tensor):
    if tensor.dtype == torch.bfloat16:
        return GL_GEMM_DTYPE
    if tensor.dtype == torch.float32:
        return gl.float32
    raise ValueError(f"Unsupported tensor descriptor dtype: {tensor.dtype}")


@gluon.aggregate
class Counter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.jit
    def create(phase, num_barriers: gl.constexpr):
        return Counter(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def next(self, pred=True):
        incr = self.index + gl.where(pred, 1, 0)
        rollover = incr == self.num_barriers
        index = gl.where(rollover, 0, incr)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return Counter(index, phase, self.num_barriers)


@gluon.aggregate
class ClcTileScheduler:
    has_work: gl.tensor
    tile_id: gl.tensor
    clc_result_buffers: gl.shared_memory_descriptor
    clc_barriers: gl.shared_memory_descriptor
    clc_tile_id_buffers: gl.shared_memory_descriptor
    clc_tile_ready_bars: gl.shared_memory_descriptor
    clc_consumed_bars: gl.shared_memory_descriptor
    counter: Counter
    consumed_counter: Counter

    @gluon.jit
    def initialize(clc_result_buffers, clc_barriers, clc_tile_id_buffers, clc_tile_ready_bars, clc_consumed_bars):
        return ClcTileScheduler(
            gl.to_tensor(True),
            gl.program_id(axis=0),
            clc_result_buffers,
            clc_barriers,
            clc_tile_id_buffers,
            clc_tile_ready_bars,
            clc_consumed_bars,
            Counter.create(0, clc_barriers.shape[0]),
            Counter.create(0, clc_barriers.shape[0]),
        )

    @gluon.jit
    def step(self, iteration):
        consumed_counter = self.consumed_counter
        if iteration > 0:
            mbarrier.arrive(self.clc_consumed_bars.index(consumed_counter.index))
            consumed_counter = consumed_counter.next()

        counter = self.counter
        barrier = self.clc_barriers.index(counter.index)
        result = self.clc_result_buffers.index(counter.index)
        mbarrier.wait(barrier, counter.phase)
        clc_res = clc.load_result(result)
        mbarrier.wait(self.clc_tile_ready_bars.index(counter.index), counter.phase)

        tile_slot = self.clc_tile_id_buffers.index(counter.index)
        planar_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0],
                                                       [[0]] * (gl.num_ctas().bit_length() - 1))
        tile_id = tile_slot.load(planar_layout).reshape([]).to(gl.int32)
        has_work = clc_res.is_canceled()
        return ClcTileScheduler(
            has_work,
            tile_id,
            self.clc_result_buffers,
            self.clc_barriers,
            self.clc_tile_id_buffers,
            self.clc_tile_ready_bars,
            self.clc_consumed_bars,
            counter.next(),
            consumed_counter,
        )


@gluon.jit
def clc_tile_scheduler_partition(p):
    """Cancel future CTAs and publish their tile IDs to consumer partitions."""
    has_work = gl.to_tensor(True)
    state = Counter.create(0, p.clc_barriers.shape[0])
    consumed_state = Counter.create(1, p.clc_barriers.shape[0])
    num_slots: gl.constexpr = p.clc_barriers.shape[0]
    i = 0
    while has_work:
        mbarrier.wait(
            p.clc_consumed_bars.index(consumed_state.index),
            consumed_state.phase,
            pred=(i >= num_slots),
        )
        barrier = p.clc_barriers.index(state.index)
        result = p.clc_result_buffers.index(state.index)
        # clc.try_cancel returns a b128 payload.
        mbarrier.expect(barrier, 16)
        clc.try_cancel(result, barrier)
        mbarrier.wait(barrier, state.phase)

        clc_res = clc.load_result(result)
        has_work = clc_res.is_canceled()
        tile_id = gl.to_tensor(0)
        if has_work:
            tile_id = clc_res.program_id(0)

        tile_slot = p.clc_tile_id_buffers.index(state.index)
        planar_layout: gl.constexpr = gl.BlockedLayout([1], [32], [gl.num_warps()], [0],
                                                       [[0]] * (gl.num_ctas().bit_length() - 1))
        tile_slot.store(gl.full([1], tile_id.to(gl.int64), gl.int64, layout=planar_layout))
        mbarrier.arrive(p.clc_tile_ready_bars.index(state.index))
        state = state.next()
        consumed_state = consumed_state.next()
        i += 1


@gluon.jit
def invalidate_mbarrier_ring(bars):
    num_bars: gl.constexpr = bars.type.shape[0]
    for i in gl.static_range(num_bars):
        mbarrier.invalidate(bars.index(i))


__all__ = [
    "ClcTileScheduler",
    "Counter",
    "GL_GEMM_DTYPE",
    "TORCH_GEMM_DTYPE",
    "clc_tile_scheduler_partition",
    "ensure_tma_compatible_strides",
    "get_gluon_dtype_for_tensor",
    "get_operand_cga_layout",
    "get_transposed_cga_layout",
    "invalidate_mbarrier_ring",
    "is_blackwell",
    "is_cuda",
    "maybe_pad_channel_dims_for_tma",
    "normalize_2d",
    "validate_2cta_m_split",
]
