import torch

import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier

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
class PersistentTileScheduler:
    pid_start: gl.tensor
    pid_end: gl.tensor

    @gluon.jit
    def initialize(num_tiles):
        kernel_id = gl.program_id(axis=0)
        num_kernels = gl.num_programs(axis=0)
        pid_per_kernel = gl.cdiv(num_tiles, num_kernels)
        pid_start = kernel_id * pid_per_kernel
        pid_end = gl.minimum(pid_start + pid_per_kernel, num_tiles)
        return PersistentTileScheduler(pid_start, pid_end)

    @gluon.jit
    def get_num_tiles(self):
        return self.pid_end - self.pid_start

    @gluon.jit
    def get_tile_id(self, idx):
        return self.pid_start + idx


@gluon.jit
def init_mbarrier_ring(bars):
    num_bars: gl.constexpr = bars.type.shape[0]
    for i in gl.static_range(num_bars):
        mbarrier.init(bars.index(i), count=1)


@gluon.jit
def invalidate_mbarrier_ring(bars):
    num_bars: gl.constexpr = bars.type.shape[0]
    for i in gl.static_range(num_bars):
        mbarrier.invalidate(bars.index(i))


__all__ = [
    "Counter",
    "GL_GEMM_DTYPE",
    "PersistentTileScheduler",
    "TORCH_GEMM_DTYPE",
    "ensure_tma_compatible_strides",
    "init_mbarrier_ring",
    "invalidate_mbarrier_ring",
    "is_blackwell",
    "is_cuda",
    "maybe_pad_channel_dims_for_tma",
    "normalize_2d",
]
