# fmt: off


import numpy as np
import torch
import pytest
import triton
import triton.language as tl

from triton._internal_testing import is_cuda, is_hip, is_hip_cdna3, is_hip_cdna4


def matching_int(dtype):
    if dtype.primitive_bitwidth == 8:
        return torch.int8
    elif dtype.primitive_bitwidth == 16:
        return torch.int16
    elif dtype.primitive_bitwidth == 32:
        return torch.int32
    elif dtype.primitive_bitwidth == 64:
        return torch.int64
    else:
        raise ValueError('unsupported number of bits')

@triton.jit
def type_convert_triton(src, dst, rounding : tl.constexpr, BLOCK_SIZE : tl.constexpr):

    idxs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(src + idxs)
    y = x.to(dst.dtype.element_ty, fp_downcast_rounding=rounding)
    tl.store(dst + idxs, y)


def launch_type_convert_triton(src, src_dtype, dst_dtype, device, rounding=None, BLOCK_SIZE=4096):

    dst = torch.empty(src.shape, dtype=matching_int(dst_dtype), device=device)
    type_convert_triton[(src.shape[0] // BLOCK_SIZE,)](triton.reinterpret(src, src_dtype), triton.reinterpret(dst, dst_dtype), rounding, BLOCK_SIZE)
    return dst


@triton.jit
def exhaustive_populate(dst, offset, BLOCK_SIZE : tl.constexpr, force_odd : tl.constexpr, output_bits : tl.constexpr, max_repr : tl.constexpr):

    idxs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    vals = (idxs + offset).to(tl.uint32)

    # pseudorandom permutation:
    multiplier = vals << 1
    multiplier += 3511
    vals *= multiplier

    if force_odd:
        vals *= 2
        vals += 1

    if (output_bits == 8):
        vals &= 0xff
        avals = vals & 0x7f
    elif (output_bits == 16):
        vals &= 0xffff
        avals = vals & 0x7fff
    elif (output_bits == 32):
        avals = vals & 0x7fffffff

    vals = tl.where(avals <= max_repr, vals, 0)

    if (output_bits == 8):
        vals = vals.to(tl.uint8)
    elif (output_bits == 16):
        vals = vals.to(tl.uint16)

    vals = vals.to(dst.dtype.element_ty, bitcast=True)
    tl.store(dst + idxs, vals)


def launch_exhaustive_populate(dst_dtype, offset, numel, force_odd, output_bits, max_repr, device, BLOCK_SIZE=4096):

    assert(numel % BLOCK_SIZE == 0)
    dst = torch.empty((numel,), dtype=matching_int(dst_dtype), device=device)
    exhaustive_populate[(numel // BLOCK_SIZE,)](triton.reinterpret(dst, dst_dtype), offset, BLOCK_SIZE, force_odd, output_bits, max_repr)
    # 0x80 in float8e4b8 or float8e5b16 represents inf/nan. We don't need to have that
    # as input to the conversion kernels.
    if dst_dtype == tl.float8e4b8 or dst_dtype == tl.float8e5b16:
        dst = torch.where(dst == 0x80, 0, dst)
    return dst


@triton.jit
def arbitrary_fp32_downcast(x, rounding : tl.constexpr, exponent_bits : tl.constexpr, mantissa_bits : tl.constexpr, exponent_bias : tl.constexpr):

    tl.static_assert(x.dtype == tl.float32, "input must be float32")
    numbits_dst : tl.constexpr = 1 + exponent_bits + mantissa_bits
    tl.static_assert((numbits_dst == 8) or (numbits_dst == 16), "numbits_dst must be 8 or 16")

    x = x.to(tl.uint32, bitcast=True)

    mantissa = (x & 0x7fffff)
    exponent = ((x >> 23) & 0xff).to(tl.int32)
    mantissa = tl.where(exponent == 0, mantissa, mantissa + 0x800000).to(tl.int32)
    exponent = tl.where(exponent == 0, exponent, exponent - 1)

    sign = (x >> 31)

    exponent = exponent + exponent_bias - 127
    adjustment : tl.constexpr = 0.5 ** (23 - mantissa_bits)
    mantissa = mantissa.to(tl.float32) * adjustment

    # make exponent nonnegative:
    mantissa = tl.where(exponent > -16, mantissa, 0.0) # destination has fewer than 16 mantissa bits, so safe
    exponent = tl.where(exponent > -16, exponent, 0)
    mantissa = tl.where(exponent > -8, mantissa, mantissa * 0.00390625)
    exponent = tl.where(exponent > -8, exponent, exponent + 8)
    mantissa = tl.where(exponent > -4, mantissa, mantissa * 0.0625)
    exponent = tl.where(exponent > -4, exponent, exponent + 4)
    mantissa = tl.where(exponent > -2, mantissa, mantissa * 0.25)
    exponent = tl.where(exponent > -2, exponent, exponent + 2)
    mantissa = tl.where(exponent > -1, mantissa, mantissa * 0.5)
    exponent = tl.where(exponent > -1, exponent, exponent + 1)

    if rounding == 'rtne':
        # Bring the value to the range [2 ** 23, 2 ** 24]
        # where the representable floats map exactly to integers.
        # Addition has RTNE semantics.
        mantissa += 0x800000
        # Bring the value back to the original range.
        mantissa -= 0x800000
        mantissa = mantissa.to(tl.int32)
    elif rounding == 'rtz':
        mantissa = mantissa.to(tl.int32)
    else:
        raise ValueError('unrecognized rounding mode')

    # Reassemble output floating-point representation:
    exponent = exponent.to(tl.uint32)
    y = (sign << (exponent_bits + mantissa_bits)) + (exponent << mantissa_bits) + mantissa
    if numbits_dst == 8:
        y = y.to(tl.uint8)
    elif numbits_dst == 16:
        y = y.to(tl.uint16)
    return y


@triton.jit
def downcast_emulated(src, dst, rounding : tl.constexpr, BLOCK_SIZE : tl.constexpr, exponent_bits : tl.constexpr, mantissa_bits : tl.constexpr, exponent_bias : tl.constexpr):

    tl.static_assert(src.dtype.element_ty == tl.float32, "src dtype must be float32")

    idxs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(src + idxs)
    y = arbitrary_fp32_downcast(x, rounding, exponent_bits, mantissa_bits, exponent_bias)
    y = y.to(dst.dtype.element_ty, bitcast=True)
    tl.store(dst + idxs, y)


def launch_downcast_emulated(src, src_dtype, dst_dtype, rounding, exponent_bits, mantissa_bits, exponent_bias, device, BLOCK_SIZE=4096):

    dst = torch.empty(src.shape, dtype=matching_int(dst_dtype), device=device)
    downcast_emulated[(src.shape[0] // BLOCK_SIZE,)](
        triton.reinterpret(src, src_dtype), triton.reinterpret(dst, dst_dtype), rounding, BLOCK_SIZE, exponent_bits, mantissa_bits, exponent_bias)
    # 0x80 in float8e4b8 or float8e5b16 represents inf/nan. downcast_emulated kernel will
    # convert -0. in higher precision to 0x80 and thus need to fix the result to 0.
    if dst_dtype == tl.float8e4b8 or dst_dtype == tl.float8e5b16:
        dst = torch.where(dst == 0x80, 0, dst)
    return dst


@triton.jit
def upcast_emulated(src, dst, BLOCK_SIZE : tl.constexpr, exponent_bits : tl.constexpr, mantissa_bits : tl.constexpr, exponent_bias : tl.constexpr):

    exponent_compensator : tl.constexpr = 2.0 ** (127 - exponent_bias)

    numbits_src : tl.constexpr = 1 + exponent_bits + mantissa_bits
    tl.static_assert((numbits_src == 8) or (numbits_src == 16), "numbits_src must be 8 or 16")
    tl.static_assert(dst.dtype.element_ty == tl.float32, "dst dtype must be float32")

    idxs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(src + idxs)

    if numbits_src == 8:
        x = x.to(tl.uint8, bitcast=True)
    elif numbits_src == 16:
        x = x.to(tl.uint16, bitcast=True)

    x = x.to(tl.uint32)

    mantissa_mask : tl.constexpr = (1 << mantissa_bits) - 1
    exponent_mask : tl.constexpr = (1 << exponent_bits) - 1

    mantissa = x & mantissa_mask
    exponent = (x >> mantissa_bits) & exponent_mask
    sign = (x >> (numbits_src - 1))

    y = (sign << 31) | (exponent << 23) | (mantissa << (23 - mantissa_bits))
    y = y.to(tl.float32, bitcast=True)
    y = y * exponent_compensator

    tl.store(dst + idxs, y)


def launch_upcast_emulated(src, exponent_bits, mantissa_bits, exponent_bias, device, BLOCK_SIZE=4096):

    dst = torch.empty(src.shape, dtype=torch.int32, device=device)
    upcast_emulated[(src.shape[0] // BLOCK_SIZE,)](src, triton.reinterpret(dst, tl.float32), BLOCK_SIZE, exponent_bits, mantissa_bits, exponent_bias)
    return dst


def downcast_test(src_dtype, dst_dtype, rounding, exponent_bits, mantissa_bits, exponent_bias, max_repr, offset, device):

    src = launch_exhaustive_populate(src_dtype, offset << 24, 2**24, False, src_dtype.primitive_bitwidth, max_repr, device)
    dst = launch_type_convert_triton(src, src_dtype, dst_dtype, device=device, rounding=rounding)
    src = launch_type_convert_triton(src, src_dtype, tl.float32, device=device)

    dst2 = launch_downcast_emulated(src, tl.float32, dst_dtype, rounding, exponent_bits, mantissa_bits, exponent_bias, device=device)

    dst = launch_upcast_emulated(dst, exponent_bits, mantissa_bits, exponent_bias, device=device)
    dst2 = launch_upcast_emulated(dst2, exponent_bits, mantissa_bits, exponent_bias, device=device)

    if not (torch.equal(dst, dst2)):
        print('Error!!!')

        dst = dst.cpu().detach().numpy()
        dst2 = dst2.cpu().detach().numpy()
        src = src.cpu().detach().numpy()

        print(src[dst != dst2][0])
        print(dst[dst != dst2][0])
        print(dst2[dst != dst2][0])
        print(hex(src.view(np.uint32)[dst != dst2][0]))
        print(hex(dst.view(np.uint32)[dst != dst2][0]))
        print(hex(dst2.view(np.uint32)[dst != dst2][0]))
        print('')
        raise ValueError('%d elements mismatch' % (dst != dst2).sum())


def upcast_test(src_dtype, dst_dtype, exponent_bits, mantissa_bits, exponent_bias, max_repr, device):

    numbits_src = exponent_bits + mantissa_bits + 1

    src = launch_exhaustive_populate(src_dtype, 0, 65536, False, numbits_src, max_repr, device=device)

    dst = launch_type_convert_triton(src, src_dtype, dst_dtype, device=device)
    dst_to_float32 = launch_type_convert_triton(dst, dst_dtype, tl.float32, device=device)

    src_emulated_to_float32 = launch_upcast_emulated(src, exponent_bits, mantissa_bits, exponent_bias, device=device)

    assert(torch.equal(src_emulated_to_float32, dst_to_float32))


@pytest.mark.parametrize("src_dtype, dst_dtype", [
    ('float16', 'float32'),
    ('bfloat16', 'float32'),

    ('float8e5', 'float16'),
    ('float8e5', 'bfloat16'),
    ('float8e5', 'float32'),

    ('float8e4b15', 'float16'),
    # ('float8e4b15', 'bfloat16'), # Unsupported conversion from f8E4M3B11FNUZ to bf16
    ('float8e4b15', 'float32'),

    ('float8e4nv', 'float16'),
    ('float8e4nv', 'bfloat16'),
    ('float8e4nv', 'float32'),

    ('float8e4b8', 'float32'),
    ('float8e4b8', 'float16'),

    ('float8e5b16', 'float32'),
    ('float8e5b16', 'float16'),
])
def test_typeconvert_upcast(src_dtype, dst_dtype, device):

    # On HIP, fp8e4nv upcasting to fp32 is only supported on CDNA4, and
    # fp8e4nv upcasting to bf16 and fp16 is only supported on CDNA3 and CDNA4.
    if is_cuda():
        if ((src_dtype == 'float8e4nv' and torch.cuda.get_device_capability(0) < (8, 9))
            or src_dtype in ('float8e4b8', 'float8e5b16')):
            # If the dtype should error out in the given device, we assert that and return
            with pytest.raises(triton.CompilationError, match="not supported in this architecture"):
                launch_exhaustive_populate(getattr(tl, src_dtype), 0, 65536, False, 8, 0x7f, device=device)
            return
    elif is_hip():
        if  (src_dtype == 'float8e4nv' and not (is_hip_cdna3() or is_hip_cdna4())):
            pytest.skip(f"upcasting {src_dtype} to {dst_dtype} not supported in this architecture")
        if  (src_dtype in ('float8e4b15') or
            (src_dtype in ('float8e4b8', 'float8e5b16') and not is_hip_cdna3())):
            # If the dtype should error out in the given device, we assert that and return
            with pytest.raises(triton.CompilationError, match="not supported in this architecture"):
                launch_exhaustive_populate(getattr(tl, src_dtype), 0, 65536, False, 8, 0x7f, device=device)
            return

    # dtype : (exponent_bits, mantissa_bits, exponent_bias, max_repr)
    stuff = {
        'float8e4b15': (4, 3, 15, 0x7e),
        'float8e4nv': (4, 3, 7, 0x7e),
        'float8e5': (5, 2, 15, 0x7b),
        'float8e4b8': (4, 3, 8, 0x7f),
        'float8e5b16': (5, 2, 16, 0x7f),
        'float16': (5, 10, 15, 0x7bff),
        'bfloat16': (8, 7, 127, 0x7f7f),
    }[src_dtype]

    upcast_test(getattr(tl, src_dtype), getattr(tl, dst_dtype), *stuff, device=device)

@pytest.mark.parametrize("src_dtype, dst_dtype, rounding, max_repr", [
    ('float32', 'float16', 'rtne', 0x477fe000),
    ('float32', 'float16', 'rtz', 0x477fe000),
    ('float32', 'bfloat16', 'rtne', 0x7f7f0000),
    ('float32', 'bfloat16', 'rtz', 0x7f7f0000),
    ('float32', 'float8e5', 'rtne', 0x47600000),
    ('float32', 'float8e5', 'rtz', 0x47600000),
    ('float32', 'float8e4nv', 'rtne', 0x43e00000),
    ('float32', 'float8e4b8', 'rtne', 0x43700000),
    ('float32', 'float8e5b16', 'rtne', 0x47600000),
    # ('float32', 'float8e4b15', 'rtne', 0x3fe00000), # Skip, no HW rtne conversion from f32 to f8e4b15

    ('bfloat16', 'float8e5', 'rtne', 0x4760),
    ('bfloat16', 'float8e4nv', 'rtne', 0x43e0),

    ('float16', 'float8e5', 'rtne', 0x7b00),
    ('float16', 'float8e4nv', 'rtne', 0x5f00),

    ('bfloat16', 'float8e5b16', 'rtne', 0x4760),
    ('bfloat16', 'float8e4b8', 'rtne', 0x4370),

    ('float16', 'float8e5b16', 'rtne', 0x7b00),
    ('float16', 'float8e4b8', 'rtne', 0x5b80),
])
def test_typeconvert_downcast(src_dtype, dst_dtype, rounding, max_repr, device):

    if is_cuda():
        if src_dtype != 'float32' and torch.cuda.get_device_capability(0) < (9, 0):
            pytest.skip("non-float32 downcast tests only supported on NVGPU with compute capability 9.0+")

        if dst_dtype in ('float8e5', 'float8e4nv') and rounding == 'rtne' and torch.cuda.get_device_capability(0) < (9, 0):
            pytest.skip(f"{dst_dtype} downcast with RTNE rounding tests only supported on NVGPU with compute capability 9.0+")

        if dst_dtype in ('float8e5b16', 'float8e4b8') and rounding == 'rtne':
            pytest.skip(f"{dst_dtype} downcast with RTNE rounding tests only supported on AMDGPU CDNA3")

    if is_hip():
        if dst_dtype == 'float8e5' and rounding == 'rtne' and not is_hip_cdna4():
            pytest.skip(f"{dst_dtype} downcast with RTNE rounding tests only supported on CDNA4")

        if dst_dtype == 'float8e4nv':
            if not rounding == 'rtne':
                pytest.skip("float8e4nv downcast tests only supported with RTNE rounding on AMDGPU")
            if not (is_hip_cdna3() and src_dtype == 'float16' or is_hip_cdna4()):
                pytest.skip("float8e4nv downcast tests only supported on AMDGPU CDNA3 or on CDNA4 and from float16 with RTNE rounding")

        if dst_dtype in ('float8e5b16', 'float8e4b8') and rounding == 'rtne' and not is_hip_cdna3():
            pytest.skip(f"{dst_dtype} downcast with RTNE rounding tests only supported on AMDGPU CDNA3")

    # dtype : (exponent_bits, mantissa_bits, exponent_bias)
    stuff = {
        'float16': (5, 10, 15),
        'bfloat16': (8, 7, 127),
        'float8e5': (5, 2, 15),
        'float8e4b15': (4, 3, 15),
        'float8e4nv': (4, 3, 7),
        'float8e4b8': (4, 3, 8),
        'float8e5b16': (5, 2, 16),
    }[dst_dtype]

    for i in range(256):
        downcast_test(getattr(tl, src_dtype), getattr(tl, dst_dtype), rounding, *stuff, max_repr, i, device=device)
