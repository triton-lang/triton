from dataclasses import dataclass
from triton_bench.meta import cuda_capability_geq, rcp_max_finite
import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class BaseFlexData:
    dtype: torch.dtype | None = None

    def view(self, x: torch.Tensor):
        if self.dtype is None:
            return x
        return x.view(self.dtype)

    def reinterpret(self, x):
        if self.dtype is None or x.dtype.itemsize > 1:
            return x
        return x.view(self.dtype)


@dataclass(frozen=True)
class InFlexData(BaseFlexData):
    scale: torch.Tensor | None = None

    @property
    def is_per_batch(self):
        return False if self.scale is None else len(self.scale) > 1


@dataclass(frozen=True)
class OutFlexData(BaseFlexData):
    expected_scale: torch.Tensor | None = None
    actual_scale: torch.Tensor | None = None
    checksum_scale: torch.Tensor | None = None

    def __iter__(self):
        yield self.expected_scale
        yield self.actual_scale
        yield self.checksum_scale


def can_overflow_int32(tensor: torch.Tensor):
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32


def should_upcast_indices(*args):
    return any(tensor is not None and can_overflow_int32(tensor) for tensor in args)


# -------------------------------
# Kernels stuff
# -------------------------------


@triton.jit
def sm86_min_nan_xorsign_abs_f32(a, b):
    """Wrapper for min.NaN.xorsign.abs.f32 PTX instruction.

    Computes the minimum of the absolute values of the two inputs and sets its sign to the XOR of the signs of the inputs.
    NaN inputs are propagated to the output.

    Requires CUDA compute capability 8.6+ (A100 and A30 Ampere GPUs don't support it, but A40/A16/A10/A2, Ada, and Hopper GPUs do).
    """
    tl.static_assert(cuda_capability_geq(8, 6), "min.NaN.xorsign.abs.f32 requires CUDA compute capability 8.6+")
    tl.static_assert(a.dtype == tl.float32, "min.NaN.xorsign.abs.f32 requires float32 inputs")
    tl.static_assert(b.dtype == tl.float32, "min.NaN.xorsign.abs.f32 requires float32 inputs")

    return tl.inline_asm_elementwise(
        """{
    min.NaN.xorsign.abs.f32 $0, $1, $2;
    }""",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sm86_max_nan_xorsign_abs_f32(a, b):
    """Wrapper for max.NaN.xorsign.abs.f32 PTX instruction.

    Computes the maximum of the absolute values of the two inputs and sets its sign to the XOR of the signs of the inputs.
    NaN inputs are propagated to the output.

    Requires CUDA compute capability 8.6+ (A100 and A30 Ampere GPUs don't support it, but A40/A16/A10/A2, Ada, and Hopper GPUs do).
    """
    tl.static_assert(cuda_capability_geq(8, 6), "max.NaN.xorsign.abs.f32 requires CUDA compute capability 8.6+")
    tl.static_assert(a.dtype == tl.float32, "max.NaN.xorsign.abs.f32 requires float32 inputs")
    tl.static_assert(b.dtype == tl.float32, "max.NaN.xorsign.abs.f32 requires float32 inputs")

    return tl.inline_asm_elementwise(
        """{
    max.NaN.xorsign.abs.f32 $0, $1, $2;
    }""",
        "=r,r,r",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def load_scale(scale_ptr):
    return 1.0 if scale_ptr is None else tl.load(scale_ptr)


@triton.jit
def flex_to_float(x, scale_ptr):
    scale = load_scale(scale_ptr)
    return x.to(tl.float32) * scale


@triton.jit
def clip(x, limit):
    res = tl.minimum(x, limit)
    res = tl.maximum(-limit, res)
    return res


@triton.jit
def nan_propagating_absmax_reduce(x, axis=None):
    if cuda_capability_geq(8, 6):
        # abs-max-reduce as floating-point if `max.NaN.xorsign.abs.f32` is supported.
        x_absmax = tl.reduce(x, axis, sm86_max_nan_xorsign_abs_f32)
        # Note: sign of reduction result is the xor of signs of all inputs, explicitly clear the sign bit to fix it.
        x_absmax = x_absmax.to(tl.uint32, bitcast=True) & 0x7FFFFFFF
    else:
        # Clear the sign bit, max-reduce as integer (same as NaN-propagating max-reduce as float)
        masked_abs_x = x.to(tl.uint32, bitcast=True) & 0x7FFFFFFF
        x_absmax = tl.max(masked_abs_x, axis)

    return x_absmax


@triton.jit
def compute_scale(x, Out):
    x_absmax = nan_propagating_absmax_reduce(tl.ravel(x, can_reorder=True))

    # atomic_max does not propagate NaNs, so we replace them with +inf (0x7f800000).
    # We use integer minimum because NaNs are above +inf in integer representation.
    x_absmax = tl.minimum(x_absmax, 0x7F800000).to(tl.float32, bitcast=True)
    RCP_MAX_VALUE = rcp_max_finite(Out.dtype.element_ty)
    return tl.fma(x_absmax, RCP_MAX_VALUE.to(tl.float32, bitcast=True), 1.0e-30)


@triton.jit
def update_scale(x, scale_ptr, Out) -> None:
    if scale_ptr is not None:
        scale = compute_scale(x, Out)
        tl.atomic_max(scale_ptr, scale, sem="relaxed")


@triton.jit
def float_to_flex(
    x,
    expected_scale_ptr,
    actual_scale_ptr,
    checksum_scale_ptr,
    mask,
    Out,
    saturate_infs: tl.constexpr,
):
    invscale = 1.0 / load_scale(expected_scale_ptr)
    if checksum_scale_ptr is not None:
        x_int32 = x.to(tl.int32, bitcast=True)
        zero = tl.cast(0.0, tl.int32)
        if mask is not None:
            x_int32 = tl.where(mask, x_int32, zero)
        checksum_local = tl.xor_sum(tl.ravel(x_int32, can_reorder=True), 0)
        tl.atomic_add(checksum_scale_ptr, checksum_local)
    if mask is not None:
        if actual_scale_ptr is not None:
            x = tl.where(mask, x, 0.0)
    update_scale(x, actual_scale_ptr, Out)
    x = x * invscale
    # if expected_scale_ptr is not None, we applied flexpoint scale. We only want to clip in this case.
    if expected_scale_ptr is not None:
        if saturate_infs:
            CLIP_VALUE = max_finite(Out.dtype.element_ty)
            x = clip(x, CLIP_VALUE)
    return x
