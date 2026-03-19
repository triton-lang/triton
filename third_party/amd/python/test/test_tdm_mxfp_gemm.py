#!/usr/bin/env python3
"""
MXFP GEMM Kernel with Descriptor-based Scale Loading Support

This file contains a complete MXFP (Microscaling Floating Point) GEMM kernel
implementation with multiple scale loading strategies and a test driver.

Load Modes (for both A/B matrices and scales):
- "pointer": Standard pointer-based loads using tl.load (tt.load)
- "descriptor": Descriptor-based loads using tl.make_tensor_descriptor + desc.load

Both modes are supported on all architectures. On MI350, descriptor loads are
converted to buffer loads by the backend.

Scale Pre-shuffling:
When SCALE_PRESHUFFLE=True, scales are expected in a pre-shuffled layout that
enables more efficient memory access patterns.

Usage:
    # Run tests with pytest
    pytest mxfp_gemm_kernel.py -v

    # Run tests via CLI
    python mxfp_gemm_kernel.py --test

    # Run single benchmark
    python mxfp_gemm_kernel.py -M 1024 -N 1024 -K 512 --dtype_a float8_e5m2 --dtype_b float8_e5m2

    # With descriptor-based scale loading and preshuffling
    python mxfp_gemm_kernel.py -M 1024 -N 1024 -K 512 --scale_descriptor --scale_preshuffle
"""

import torch
import triton
import triton.language as tl
import argparse
import pytest

# ============================================================================
# Architecture Detection
# ============================================================================


def get_gpu_arch() -> str:
    """Get the GPU architecture name (e.g., 'gfx950', 'gfx1250')."""
    return triton.runtime.driver.active.get_current_target().arch


def supports_mxfp() -> bool:
    """Check if the current GPU supports MXFP (tl.dot_scaled)."""
    arch = get_gpu_arch()
    # MXFP/dot_scaled is supported on gfx950 (MI350) and gfx1250+
    return arch == 'gfx950' or arch.startswith('gfx125') or arch.startswith('gfx126')


# ============================================================================
# Helper Functions
# ============================================================================


def get_format_string(fpflag: int) -> str:
    """Convert fpflag to format string for tl.dot_scaled."""
    format_map = {
        4: "e2m1",  # fp4
        62: "e2m3",  # fp6
        63: "e3m2",  # fp6
        8: "e5m2",  # fp8
        83: "e4m3",  # fp8
    }
    return format_map.get(fpflag, "e5m2")


def get_fpflag(dtype: str) -> int:
    """Convert dtype string to fpflag."""
    dtype_map = {
        'float4': 4,
        'float8_e5m2': 8,
        'float8_e4m3': 83,
    }
    return dtype_map.get(dtype, 8)


def pack_scale(x: torch.Tensor, preshuffle_factor: int = 128) -> torch.Tensor:
    """
    Pre-shuffle scales for optimized memory access.

    Transforms scale tensor from [NON_K, K_SCALE] to [NON_K // 128, K_SCALE * 128].
    This layout enables coalesced memory access and efficient descriptor loads.

    The transformation groups 128 consecutive rows together, interleaving their
    scale values for better memory coalescing during GPU loads.

    Args:
        x: Scale tensor of shape [NON_K, K_SCALE] (uint8, E8M0 format)
        preshuffle_factor: Number of rows to group together (default 128)

    Returns:
        Pre-shuffled scale tensor of shape [NON_K // preshuffle_factor, K_SCALE * preshuffle_factor]

    Example:
        Input:  [256, 16] -> Output: [2, 2048]
    """
    if x is None:
        return x
    NON_K, K_SCALE = x.shape
    SCALE_KWIDTH = 4 if K_SCALE >= 4 else K_SCALE
    num_chunk_m = NON_K // preshuffle_factor
    num_chunk_k = K_SCALE // SCALE_KWIDTH

    # Reshape: [NON_K, K_SCALE] -> [num_chunk_m, 4, 32, num_chunk_k, SCALE_KWIDTH]
    x = x.view(num_chunk_m, 4, preshuffle_factor // 4, num_chunk_k, SCALE_KWIDTH)
    # Permute to interleave chunks
    x = x.permute(0, 3, 2, 1, 4).contiguous()
    # Final shape: [NON_K // preshuffle_factor, K_SCALE * preshuffle_factor]
    return x.view(NON_K // preshuffle_factor, K_SCALE * preshuffle_factor)


def fp8e8m0_to_float32(scale: torch.Tensor) -> torch.Tensor:
    """
    Convert E8M0 scale factors to float32.

    E8M0 format is an 8-bit exponent-only format (no mantissa):
        Value = 2^(exponent - 127)

    This function exploits IEEE float32's bit layout to perform the conversion
    efficiently by placing the 8-bit exponent directly into float32's exponent field.

    Args:
        scale: E8M0 scale tensor (uint8)

    Returns:
        Float32 tensor with scale values
    """
    scale = scale.view(torch.uint8)
    scale = scale.to(torch.int32)
    scale = scale << 23  # Shift exponent to float32's exponent field (bits 23-30)
    scale = scale.view(torch.float32)
    return scale


def torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block: int, M: int, N: int, K: int) -> torch.Tensor:
    """
    Reference implementation for MXFP GEMM: C = (A * scale_A) @ (B * scale_B)

    Args:
        a: Input matrix A (can be MXFP4Tensor or torch tensor)
        b: Input matrix B (can be MXFP4Tensor or torch tensor)
        a_scale: Scale factors for A [M, K // scale_block]
        b_scale: Scale factors for B [N, K // scale_block]
        scale_block: Number of elements per scale factor (typically 32)
        M, N, K: Matrix dimensions

    Returns:
        Output matrix C [M, N] in float32
    """
    # Convert scales from E8M0 to float32
    a_scale_f32 = fp8e8m0_to_float32(a_scale)
    b_scale_f32 = fp8e8m0_to_float32(b_scale)

    # Expand scales to match matrix dimensions
    a_scale_f32 = a_scale_f32.to(torch.float32).repeat_interleave(scale_block, dim=1)[:M, :K]
    b_scale_f32 = b_scale_f32.to(torch.float32).repeat_interleave(scale_block, dim=1).T.contiguous()[:K, :N]

    # Convert inputs to float32
    a_f32 = a.to(torch.float32)
    b_f32 = b.to(torch.float32)

    # Apply scales and compute matrix multiply
    a_scaled = a_f32 * a_scale_f32
    b_scaled = b_f32 * b_scale_f32

    return torch.matmul(a_scaled, b_scaled).to(torch.float32)


# ============================================================================
# Triton Kernel
# ============================================================================


@triton.jit
def mxgemm_kernel(
        # Data pointers
        a_ptr, b_ptr, output_ptr, a_scale, b_scale,
        # Dimensions
        M, N, K,
        # Strides
        stride_scale, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        # Data format flags
        fpflag_a: tl.constexpr, fpflag_b: tl.constexpr,
        # Block dimensions
        SCALE_BLOCK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        # Load modes
        USE_DESCRIPTOR_LOAD: tl.constexpr, USE_MASK: tl.constexpr,
        # Scale loading options
        SCALE_LOAD_MODE: tl.constexpr = "pointer",  # "pointer" or "descriptor"
        SCALE_PRESHUFFLE: tl.constexpr = False,  # Whether scales are pre-shuffled
):
    """
    MXFP GEMM kernel: C = A @ B with microscaling.

    Computes matrix multiplication where inputs A and B use microscaling formats
    (fp4, fp8) with per-group scale factors in E8M0 format.

    Supports two load modes for both A/B matrices and scales:
    - Pointer-based: uses tl.load (tt.load)
    - Descriptor-based: uses tl.make_tensor_descriptor + desc.load

    Both modes work on all supported architectures.

    Args:
        a_ptr: Pointer to A matrix [M, K] in MXFP format
        b_ptr: Pointer to B matrix [K, N] in MXFP format
        output_ptr: Pointer to output C matrix [M, N]
        a_scale: Pointer to A scales [M, K // SCALE_BLOCK] in E8M0 format
        b_scale: Pointer to B scales [N, K // SCALE_BLOCK] in E8M0 format
        fpflag_a: Format flag for A (4=e2m1/fp4, 8=e5m2/fp8, 83=e4m3/fp8)
        fpflag_b: Format flag for B
        SCALE_BLOCK: Elements per scale factor (typically 32 for MX formats)
        USE_DESCRIPTOR_LOAD: Use descriptor loads (desc.load) for A/B matrices
        USE_MASK: Use masked loads for boundary handling (when USE_DESCRIPTOR_LOAD=False)
        SCALE_LOAD_MODE: "pointer" for tl.load, "descriptor" for desc.load
        SCALE_PRESHUFFLE: Whether scales are in pre-shuffled layout for coalesced access
    """
    # Packing factor for FP4 (2 elements per byte)
    DIV_FACTOR_A: tl.constexpr = 2 if fpflag_a == 4 else 1
    DIV_FACTOR_B: tl.constexpr = 2 if fpflag_b == 4 else 1

    # Pre-shuffle constants
    PRESHUFFLE_FACTOR: tl.constexpr = 128 if SCALE_PRESHUFFLE else 1
    BLOCK_K_SCALE: tl.constexpr = BLOCK_K // SCALE_BLOCK
    BLOCK_M_PRESHUFFLED: tl.constexpr = BLOCK_M // PRESHUFFLE_FACTOR
    BLOCK_N_PRESHUFFLED: tl.constexpr = BLOCK_N // PRESHUFFLE_FACTOR
    BLOCK_K_SCALE_PRESHUFFLED: tl.constexpr = BLOCK_K_SCALE * PRESHUFFLE_FACTOR

    # Program ID with grouped ordering for L2 cache efficiency
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for A and B
    offs_k_a = tl.arange(0, BLOCK_K // DIV_FACTOR_A)
    offs_k_b = tl.arange(0, BLOCK_K // DIV_FACTOR_B)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N

    # =========================================================================
    # Scale pointer/descriptor setup based on SCALE_LOAD_MODE
    # =========================================================================
    # Descriptor loads require at least 16 bytes in last dimension. For uint8 scales,
    # this means BLOCK_K_SCALE >= 16 (standard) or BLOCK_K_SCALE_PRESHUFFLED >= 16 (preshuffled)
    USE_DESCRIPTOR_FOR_SCALES: tl.constexpr = (SCALE_LOAD_MODE == "descriptor") and (
        (SCALE_PRESHUFFLE and BLOCK_K_SCALE_PRESHUFFLED >= 16) or (not SCALE_PRESHUFFLE and BLOCK_K_SCALE >= 16))

    if USE_DESCRIPTOR_FOR_SCALES:
        # Descriptor-based scale loading using tl.make_tensor_descriptor + desc.load
        if SCALE_PRESHUFFLE:
            # Pre-shuffled layout: [M // 128, K_scale * 128]
            a_scale_desc = tl.make_tensor_descriptor(
                base=a_scale + pid_m * BLOCK_M_PRESHUFFLED * stride_scale,
                shape=(M // PRESHUFFLE_FACTOR, K // SCALE_BLOCK * PRESHUFFLE_FACTOR),
                strides=(stride_scale, 1),
                block_shape=(BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED),
            )
            b_scale_desc = tl.make_tensor_descriptor(
                base=b_scale + pid_n * BLOCK_N_PRESHUFFLED * stride_scale,
                shape=(N // PRESHUFFLE_FACTOR, K // SCALE_BLOCK * PRESHUFFLE_FACTOR),
                strides=(stride_scale, 1),
                block_shape=(BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE_PRESHUFFLED),
            )
        else:
            # Standard layout: [M, K // SCALE_BLOCK]
            a_scale_desc = tl.make_tensor_descriptor(
                base=a_scale + pid_m * BLOCK_M * stride_scale,
                shape=(M, K // SCALE_BLOCK),
                strides=(stride_scale, 1),
                block_shape=(BLOCK_M, BLOCK_K_SCALE),
            )
            b_scale_desc = tl.make_tensor_descriptor(
                base=b_scale + pid_n * BLOCK_N * stride_scale,
                shape=(N, K // SCALE_BLOCK),
                strides=(stride_scale, 1),
                block_shape=(BLOCK_N, BLOCK_K_SCALE),
            )
    else:
        # Pointer-based scale loading (default)
        offs_scale_k = tl.arange(0, BLOCK_K_SCALE)
        if SCALE_PRESHUFFLE:
            # Pre-shuffled: need to compute offsets differently
            offs_am_shuffled = pid_m * BLOCK_M_PRESHUFFLED + tl.arange(0, BLOCK_M_PRESHUFFLED)
            offs_bn_shuffled = pid_n * BLOCK_N_PRESHUFFLED + tl.arange(0, BLOCK_N_PRESHUFFLED)
            offs_scale_k_shuffled = tl.arange(0, BLOCK_K_SCALE_PRESHUFFLED)
            a_scale_ptrs = a_scale + offs_am_shuffled[:, None] * stride_scale + offs_scale_k_shuffled[None, :]
            b_scale_ptrs = b_scale + offs_bn_shuffled[:, None] * stride_scale + offs_scale_k_shuffled[None, :]
        else:
            # Standard layout
            a_scale_ptrs = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
            b_scale_ptrs = b_scale + offs_bn[:, None] * stride_scale + offs_scale_k[None, :]

    # =========================================================================
    # A and B matrix descriptor/pointer setup
    # =========================================================================
    if USE_DESCRIPTOR_LOAD:
        a_desc = tl.make_tensor_descriptor(
            base=a_ptr + (pid_m * BLOCK_M) * stride_am,
            shape=(M, K),
            strides=(stride_am, 1),
            block_shape=(BLOCK_M, BLOCK_K // DIV_FACTOR_A),
        )
        b_desc = tl.make_tensor_descriptor(
            base=b_ptr + (pid_n * BLOCK_N) * stride_bn,
            shape=(K, N),
            strides=(stride_bk, 1),
            block_shape=(BLOCK_K // DIV_FACTOR_B, BLOCK_N),
        )
    else:
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_a[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k_b[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # =========================================================================
    # Main computation loop
    # =========================================================================
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)

    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        # -----------------------------------------------------------------
        # Load scales based on SCALE_LOAD_MODE
        # -----------------------------------------------------------------
        if USE_DESCRIPTOR_FOR_SCALES:
            if SCALE_PRESHUFFLE:
                # Load pre-shuffled scales via descriptor load
                scale_a_raw = a_scale_desc.load([0, k * BLOCK_K_SCALE_PRESHUFFLED])
                scale_b_raw = b_scale_desc.load([0, k * BLOCK_K_SCALE_PRESHUFFLED])
                # Unshuffle in registers: [M_preshuffled, K_scale_preshuffled] -> [M, K_scale]
                SCALE_KWIDTH: tl.constexpr = 4 if BLOCK_K_SCALE >= 4 else BLOCK_K_SCALE
                scale_a = tl.reshape(
                    scale_a_raw,
                    (BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, PRESHUFFLE_FACTOR // 4, 4, SCALE_KWIDTH))
                scale_a = tl.permute(scale_a, (0, 3, 2, 1, 4))
                scale_a = tl.reshape(scale_a, (BLOCK_M, BLOCK_K_SCALE))

                scale_b = tl.reshape(
                    scale_b_raw,
                    (BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, PRESHUFFLE_FACTOR // 4, 4, SCALE_KWIDTH))
                scale_b = tl.permute(scale_b, (0, 3, 2, 1, 4))
                scale_b = tl.reshape(scale_b, (BLOCK_N, BLOCK_K_SCALE))
            else:
                # Load standard layout scales via descriptor load
                scale_a = a_scale_desc.load([0, k * BLOCK_K_SCALE])
                scale_b = b_scale_desc.load([0, k * BLOCK_K_SCALE])
        else:
            # Pointer-based scale loading
            if SCALE_PRESHUFFLE:
                # Load pre-shuffled scales
                scale_a_raw = tl.load(a_scale_ptrs)
                scale_b_raw = tl.load(b_scale_ptrs)
                # Unshuffle in registers
                SCALE_KWIDTH: tl.constexpr = 4 if BLOCK_K_SCALE >= 4 else BLOCK_K_SCALE
                scale_a = tl.reshape(
                    scale_a_raw,
                    (BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, PRESHUFFLE_FACTOR // 4, 4, SCALE_KWIDTH))
                scale_a = tl.permute(scale_a, (0, 3, 2, 1, 4))
                scale_a = tl.reshape(scale_a, (BLOCK_M, BLOCK_K_SCALE))

                scale_b = tl.reshape(
                    scale_b_raw,
                    (BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, PRESHUFFLE_FACTOR // 4, 4, SCALE_KWIDTH))
                scale_b = tl.permute(scale_b, (0, 3, 2, 1, 4))
                scale_b = tl.reshape(scale_b, (BLOCK_N, BLOCK_K_SCALE))

                # Advance pointers
                a_scale_ptrs += BLOCK_K_SCALE_PRESHUFFLED
                b_scale_ptrs += BLOCK_K_SCALE_PRESHUFFLED
            else:
                # Standard pointer load
                scale_a = tl.load(a_scale_ptrs)
                scale_b = tl.load(b_scale_ptrs)
                # Advance pointers
                a_scale_ptrs += BLOCK_K_SCALE
                b_scale_ptrs += BLOCK_K_SCALE

        # -----------------------------------------------------------------
        # Load A and B matrices
        # -----------------------------------------------------------------
        if USE_DESCRIPTOR_LOAD:
            a = a_desc.load([0, k * (BLOCK_K // DIV_FACTOR_A)])
            b = b_desc.load([k * (BLOCK_K // DIV_FACTOR_B), 0])
        else:
            k_remaining_a = K - k * (BLOCK_K // DIV_FACTOR_A)
            k_remaining_b = K - k * (BLOCK_K // DIV_FACTOR_B)
            valid_k_a = offs_k_a < k_remaining_a
            valid_k_b = offs_k_b < k_remaining_b
            if USE_MASK:
                a = tl.load(a_ptrs, mask=valid_k_a[None, :], other=0.)
                b = tl.load(b_ptrs, mask=valid_k_b[:, None], other=0.)
            else:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
            a_ptrs += (BLOCK_K // DIV_FACTOR_A) * stride_ak
            b_ptrs += (BLOCK_K // DIV_FACTOR_B) * stride_bk

        # -----------------------------------------------------------------
        # Scaled matrix multiply using tl.dot_scaled
        # -----------------------------------------------------------------
        if fpflag_a == 4 and fpflag_b == 4:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e2m1", accumulator)
        elif fpflag_a == 8 and fpflag_b == 8:
            accumulator = tl.dot_scaled(a, scale_a, "e5m2", b, scale_b, "e5m2", accumulator)
        elif fpflag_a == 83 and fpflag_b == 83:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b, scale_b, "e4m3", accumulator)
        elif fpflag_a == 4 and fpflag_b == 8:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e5m2", accumulator)
        elif fpflag_a == 8 and fpflag_b == 4:
            accumulator = tl.dot_scaled(a, scale_a, "e5m2", b, scale_b, "e2m1", accumulator)
        elif fpflag_a == 83 and fpflag_b == 8:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b, scale_b, "e5m2", accumulator)
        elif fpflag_a == 8 and fpflag_b == 83:
            accumulator = tl.dot_scaled(a, scale_a, "e5m2", b, scale_b, "e4m3", accumulator)
        elif fpflag_a == 4 and fpflag_b == 83:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b, scale_b, "e4m3", accumulator)
        elif fpflag_a == 83 and fpflag_b == 4:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b, scale_b, "e2m1", accumulator)

    # =========================================================================
    # Store output
    # =========================================================================
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)


# ============================================================================
# Test Driver
# ============================================================================


def init_data(dtype: str, d0: int, d1: int, device: str = 'cuda'):
    """Initialize input data based on dtype."""
    from triton.tools.mxfp import MXFP4Tensor

    if dtype == 'float4':
        data = torch.randint(1, 5, (d0, d1))
        return MXFP4Tensor(data=data)
    elif dtype == 'float8_e5m2':
        return torch.randint(1, 5, (d0, d1)).to(torch.float8_e5m2)
    elif dtype == 'float8_e4m3':
        return torch.randint(1, 5, (d0, d1)).to(torch.float8_e4m3fn)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def run_mxfp_gemm(
    M: int,
    N: int,
    K: int,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_K: int = 128,
    dtype_a: str = 'float8_e5m2',
    dtype_b: str = 'float8_e5m2',
    scale_descriptor: bool = False,
    scale_preshuffle: bool = False,
    use_descriptor_load: bool = True,
    use_mask: bool = False,
    num_warps: int = 4,
    group_size_m: int = 8,
    verify: bool = True,
):
    """
    Run MXFP GEMM kernel with specified configuration.

    Args:
        M, N, K: Matrix dimensions
        BLOCK_M, BLOCK_N, BLOCK_K: Tile sizes
        dtype_a, dtype_b: Data types for A and B ('float4', 'float8_e5m2', 'float8_e4m3')
        scale_descriptor: Use descriptor loads for scale loading
        scale_preshuffle: Use pre-shuffled scale layout
        use_descriptor_load: Use descriptor loads for A/B matrix loading
        num_warps: Number of warps per block
        verify: Verify results against reference

    Returns:
        Tuple of (output_tensor, reference_tensor, max_diff)
    """
    from triton.tools.mxfp import MXScaleTensor

    SCALE_BLOCK = 32
    PRESHUFFLE_FACTOR = 128

    # Validate configuration
    if scale_preshuffle and (BLOCK_M < PRESHUFFLE_FACTOR or BLOCK_N < PRESHUFFLE_FACTOR):
        raise ValueError(f"BLOCK_M and BLOCK_N must be >= {PRESHUFFLE_FACTOR} for scale preshuffling")

    torch.manual_seed(42)

    fpflag_a = get_fpflag(dtype_a)
    fpflag_b = get_fpflag(dtype_b)

    # Initialize data
    a = init_data(dtype_a, M, K)
    b = init_data(dtype_b, K, N)

    # Create scales
    a_scale = MXScaleTensor(size=(M, (K + SCALE_BLOCK - 1) // SCALE_BLOCK)).random(high=32.0).data
    b_scale = MXScaleTensor(size=(N, (K + SCALE_BLOCK - 1) // SCALE_BLOCK)).random(high=32.0).data

    # Compute reference
    c_ref = torch_gemm_mxfp(a, b, a_scale, b_scale, SCALE_BLOCK, M, N, K)

    # Pack fp4 data if needed
    if dtype_a == 'float4':
        a = a.to_packed_tensor(dim=1)
    if dtype_b == 'float4':
        b = b.to_packed_tensor(dim=0)

    # Pre-shuffle scales if enabled
    if scale_preshuffle:
        a_scale_input = pack_scale(a_scale, PRESHUFFLE_FACTOR)
        b_scale_input = pack_scale(b_scale, PRESHUFFLE_FACTOR)
    else:
        a_scale_input = a_scale
        b_scale_input = b_scale

    # Move to GPU
    a_d = a.data.contiguous().cuda()
    b_d = b.data.contiguous().cuda()
    a_scale_d = a_scale_input.cuda()
    b_scale_d = b_scale_input.cuda()
    c_d = torch.zeros(M, N, dtype=torch.float32, device='cuda')

    # Launch kernel
    num_blocks = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (num_blocks, )

    scale_load_mode = "descriptor" if scale_descriptor else "pointer"

    mxgemm_kernel[grid](
        a_d,
        b_d,
        c_d,
        a_scale_d,
        b_scale_d,
        M,
        N,
        K,
        a_scale_d.stride(0),
        a_d.stride(0),
        a_d.stride(1),
        b_d.stride(0),
        b_d.stride(1),
        c_d.stride(0),
        c_d.stride(1),
        fpflag_a,
        fpflag_b,
        SCALE_BLOCK,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        group_size_m,
        USE_DESCRIPTOR_LOAD=use_descriptor_load,
        USE_MASK=use_mask,
        SCALE_LOAD_MODE=scale_load_mode,
        SCALE_PRESHUFFLE=scale_preshuffle,
        num_warps=num_warps,
    )

    torch.cuda.synchronize()

    # Verify
    c_d_cpu = c_d.cpu()
    max_diff = (c_d_cpu - c_ref).abs().max().item()

    return c_d_cpu, c_ref, max_diff


# ============================================================================
# Pytest Test Functions
# ============================================================================


def requires_mxfp_support():
    """Skip test if current GPU does not support MXFP."""
    if not supports_mxfp():
        pytest.skip("MXFP requires gfx950 (MI350) or gfx1250+")


# Data values {1,2,3,4} are exactly representable in all formats
# (M, N, K, BM, BN, BK, dtype_a, dtype_b, use_descriptor_load, use_mask, scale_load_mode, scale_preshuffle)
CONFIGS = [
    # Descriptor load for A/B, pointer for scales
    (32, 32, 64, 32, 32, 64, 'float8_e5m2', 'float8_e5m2', True, False, 'pointer', False),
    (64, 64, 256, 32, 32, 256, 'float8_e5m2', 'float8_e5m2', True, False, 'pointer', False),
    (128, 128, 512, 64, 64, 128, 'float8_e5m2', 'float8_e5m2', True, False, 'pointer', False),
    (32, 32, 128, 32, 32, 128, 'float8_e5m2', 'float4', True, False, 'pointer', False),
    (64, 64, 256, 32, 32, 256, 'float4', 'float4', True, False, 'pointer', False),
    # Pointer load for A/B with mask
    (32, 32, 64, 32, 32, 64, 'float8_e5m2', 'float8_e5m2', False, True, 'pointer', False),
    (128, 128, 512, 64, 64, 128, 'float8_e5m2', 'float8_e5m2', False, True, 'pointer', False),
    # Scale loading modes and preshuffling (larger tiles for preshuffle)
    (256, 256, 512, 128, 128, 128, 'float8_e5m2', 'float8_e5m2', True, False, 'pointer', True),
    (256, 256, 512, 128, 128, 128, 'float8_e5m2', 'float4', True, False, 'pointer', True),
    (256, 256, 512, 128, 128, 128, 'float4', 'float8_e5m2', True, False, 'pointer', True),
    (256, 256, 512, 128, 128, 128, 'float4', 'float4', True, False, 'pointer', True),
    (256, 256, 512, 128, 128, 128, 'float8_e4m3', 'float8_e4m3', True, False, 'pointer', True),
    (256, 256, 512, 128, 128, 128, 'float8_e4m3', 'float8_e5m2', True, False, 'pointer', True),
    (256, 256, 512, 128, 128, 128, 'float8_e5m2', 'float8_e5m2', True, False, 'descriptor', True),
    (256, 256, 512, 128, 128, 128, 'float8_e5m2', 'float4', True, False, 'descriptor', True),
    (256, 256, 512, 128, 128, 128, 'float4', 'float8_e5m2', True, False, 'descriptor', True),
    (256, 256, 512, 128, 128, 128, 'float4', 'float4', True, False, 'descriptor', True),
    (256, 256, 512, 128, 128, 128, 'float8_e4m3', 'float8_e4m3', True, False, 'descriptor', True),
    (256, 256, 512, 128, 128, 128, 'float8_e4m3', 'float8_e5m2', True, False, 'descriptor', True),
]


@pytest.mark.parametrize(
    "M,N,K,BM,BN,BK,dtype_a,dtype_b,use_descriptor_load,use_mask,scale_load_mode,scale_preshuffle", CONFIGS, ids=[
        f"{c[6]}x{c[7]}_{c[0]}x{c[1]}x{c[2]}_b{c[3]}x{c[4]}x{c[5]}"
        f"_desc{c[8]}_mask{c[9]}_scale_{c[10]}_preshuffle{c[11]}" for c in CONFIGS
    ])
def test_mxgemm(M, N, K, BM, BN, BK, dtype_a, dtype_b, use_descriptor_load, use_mask, scale_load_mode,
                scale_preshuffle):
    """Test MXFP GEMM with various load modes, scale modes, and preshuffling."""
    requires_mxfp_support()

    output, ref, max_diff = run_mxfp_gemm(
        M,
        N,
        K,
        BM,
        BN,
        BK,
        dtype_a,
        dtype_b,
        scale_descriptor=(scale_load_mode == 'descriptor'),
        scale_preshuffle=scale_preshuffle,
        use_descriptor_load=use_descriptor_load,
        use_mask=use_mask,
        group_size_m=1,
    )

    torch.testing.assert_close(output, ref, rtol=1e-5, atol=0.02)


def run_tests():
    """Run tests using pytest. Returns True if all tests pass."""

    print("=" * 70)
    print("MXFP GEMM Kernel Test Suite")
    print("=" * 70)
    print(f"GPU Architecture: {get_gpu_arch()}")
    print(f"MXFP Support: {supports_mxfp()}")
    print("=" * 70)

    # Run pytest on this file
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    return exit_code == 0


def main():
    parser = argparse.ArgumentParser(description='MXFP GEMM Kernel')

    parser.add_argument('--test', action='store_true', help='Run test suite')
    parser.add_argument('-M', type=int, default=1024, help='M dimension')
    parser.add_argument('-N', type=int, default=1024, help='N dimension')
    parser.add_argument('-K', type=int, default=512, help='K dimension')
    parser.add_argument('-BM', type=int, default=128, help='BLOCK_M')
    parser.add_argument('-BN', type=int, default=128, help='BLOCK_N')
    parser.add_argument('-BK', type=int, default=128, help='BLOCK_K')
    parser.add_argument('--dtype_a', type=str, default='float8_e5m2', choices=['float4', 'float8_e5m2', 'float8_e4m3'])
    parser.add_argument('--dtype_b', type=str, default='float8_e5m2', choices=['float4', 'float8_e5m2', 'float8_e4m3'])
    parser.add_argument('--scale_descriptor', action='store_true', help='Use descriptor loads for scale loading')
    parser.add_argument('--scale_preshuffle', action='store_true', help='Use pre-shuffled scales')
    parser.add_argument('--num_warps', type=int, default=4, help='Number of warps')
    parser.add_argument('--no_verify', action='store_true', help='Skip verification')

    args = parser.parse_args()

    if args.test:
        success = run_tests()
        exit(0 if success else 1)

    print(f"Running MXFP GEMM: {args.dtype_a} x {args.dtype_b}")
    print(f"  Dimensions: M={args.M}, N={args.N}, K={args.K}")
    print(f"  Block sizes: BM={args.BM}, BN={args.BN}, BK={args.BK}")
    print(f"  Scale Descriptor Load: {args.scale_descriptor}, Scale Preshuffle: {args.scale_preshuffle}")

    output, ref, max_diff = run_mxfp_gemm(
        args.M,
        args.N,
        args.K,
        args.BM,
        args.BN,
        args.BK,
        args.dtype_a,
        args.dtype_b,
        args.scale_descriptor,
        args.scale_preshuffle,
        num_warps=args.num_warps,
        verify=not args.no_verify,
    )

    print("\nResults:")
    print(f"  Output range: [{output.min():.2f}, {output.max():.2f}]")
    print(f"  Reference range: [{ref.min():.2f}, {ref.max():.2f}]")
    print(f"  Max difference: {max_diff:.6f}")

    # Use relative error for correctness check (accounts for larger accumulated error in bigger matrices)
    max_val = ref.abs().max().item()
    rel_error = max_diff / (max_val + 1e-10)
    print(f"  Relative error: {rel_error:.2e}")

    # Pass if relative error < 1e-5 (very tight) or absolute error < 0.02 (for small outputs)
    if rel_error < 1e-5 or max_diff < 0.02:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")


if __name__ == '__main__':
    main()
