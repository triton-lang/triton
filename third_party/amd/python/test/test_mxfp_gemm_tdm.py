#!/usr/bin/env python3
"""
MXFP GEMM Kernel with Descriptor Loads and Pre-shuffled Scales

This file contains an optimized MXFP (Microscaling Floating Point) GEMM kernel
using descriptor-based loads for both matrices and scales, with pre-shuffled
scale layout for efficient memory access.

Features:
- Descriptor loads (tl.make_tensor_descriptor) for A, B matrices and scales
- Pre-shuffled scale layout (128-element groups) for coalesced memory access
- Supports fp4, fp8_e5m2, and fp8_e4m3 data types

"""

import torch
import triton
import triton.language as tl
import argparse
import pytest
from triton.tools.mxfp import MXScaleTensor, MXFP4Tensor
from triton._internal_testing import is_hip_gfx1250

# ============================================================================
# Constants
# ============================================================================

SCALE_BLOCK = 32  # Elements per scale factor (MX format standard)
PRESHUFFLE_FACTOR = 128  # Scale pre-shuffle grouping factor

dtype_to_triton_type = {
    'float4': 'e2m1',
    'float8_e5m2': 'e5m2',
    'float8_e4m3': 'e4m3',
}

# ============================================================================
# Helper Functions
# ============================================================================


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
    scale = scale.view(torch.uint8)
    scale = scale.to(torch.int32)
    scale = scale << 23  # Shift exponent to float32's exponent field (bits 23-30)
    scale = scale.view(torch.float32)
    return scale


def torch_gemm_mxfp(a, b, a_scale, b_scale, scale_block: int, M: int, N: int, K: int) -> torch.Tensor:
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
def mxgemm_kernel(a_ptr, b_ptr, output_ptr,  #
                  a_scale, b_scale,  #
                  M, N, K,  #
                  stride_scale,  #
                  stride_am, stride_ak,  #
                  stride_bk, stride_bn,  #
                  stride_cm, stride_cn,  #
                  DTYPE_A: tl.constexpr,  #
                  DTYPE_B: tl.constexpr,  #
                  SCALE_BLOCK: tl.constexpr,  #
                  BLOCK_M: tl.constexpr,  #
                  BLOCK_N: tl.constexpr,  #
                  BLOCK_K: tl.constexpr,  #
                  GROUP_SIZE_M: tl.constexpr,  #
                  ):
    """
    MXFP GEMM kernel: C = A @ B with microscaling.

    Computes matrix multiplication where inputs A and B use microscaling formats
    (fp4, fp8) with per-group scale factors in E8M0 format. Uses descriptor loads
    for both matrices and scales, with pre-shuffled scale layout for optimal performance.

    Args:
        a_ptr: Pointer to A matrix [M, K] in MXFP format
        b_ptr: Pointer to B matrix [K, N] in MXFP format
        output_ptr: Pointer to output C matrix [M, N]
        a_scale: Pointer to pre-shuffled A scales in E8M0 format
        b_scale: Pointer to pre-shuffled B scales in E8M0 format
        DTYPE_A: Format type for A ("e2m1", "e5m2", or "e4m3")
        DTYPE_B: Format type for B ("e2m1", "e5m2", or "e4m3")
        SCALE_BLOCK: Elements per scale factor (32 for MX formats)
    """
    # Packing factor for FP4 (2 elements per byte)
    DIV_FACTOR_A: tl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: tl.constexpr = 2 if DTYPE_B == "e2m1" else 1

    # Pre-shuffle constants (always enabled)
    PRESHUFFLE_FACTOR: tl.constexpr = 128
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

    # =========================================================================
    # Setup tensor descriptors for A, B matrices and scales
    # =========================================================================
    # Descriptor-based loading with pre-shuffled scales
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

    # A and B matrix descriptors
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

    # =========================================================================
    # Main computation loop
    # =========================================================================
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)

    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        # Load pre-shuffled scales via descriptor load
        scale_a_raw = a_scale_desc.load([0, k * BLOCK_K_SCALE_PRESHUFFLED])
        scale_b_raw = b_scale_desc.load([0, k * BLOCK_K_SCALE_PRESHUFFLED])

        # Unshuffle in registers: [preshuffled_dim, K_scale_preshuffled] -> [dim, K_scale]
        SCALE_KWIDTH: tl.constexpr = 4 if BLOCK_K_SCALE >= 4 else BLOCK_K_SCALE
        scale_a = tl.reshape(
            scale_a_raw, (BLOCK_M_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, PRESHUFFLE_FACTOR // 4, 4, SCALE_KWIDTH))
        scale_a = tl.permute(scale_a, (0, 3, 2, 1, 4))
        scale_a = tl.reshape(scale_a, (BLOCK_M, BLOCK_K_SCALE))

        scale_b = tl.reshape(
            scale_b_raw, (BLOCK_N_PRESHUFFLED, BLOCK_K_SCALE // SCALE_KWIDTH, PRESHUFFLE_FACTOR // 4, 4, SCALE_KWIDTH))
        scale_b = tl.permute(scale_b, (0, 3, 2, 1, 4))
        scale_b = tl.reshape(scale_b, (BLOCK_N, BLOCK_K_SCALE))

        # Load A and B matrices via descriptor load
        a = a_desc.load([0, k * (BLOCK_K // DIV_FACTOR_A)])
        b = b_desc.load([k * (BLOCK_K // DIV_FACTOR_B), 0])

        # Scaled matrix multiply using tl.dot_scaled
        accumulator = tl.dot_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator)

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
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype_a: str,
    dtype_b: str,
    num_warps: int,
    group_size_m: int,
):
    """
    Run MXFP GEMM kernel with specified configuration.

    Uses descriptor loads for both matrices and scales, with pre-shuffled scale layout.

    Args:
        M, N, K: Matrix dimensions
        BLOCK_M, BLOCK_N, BLOCK_K: Tile sizes
        dtype_a, dtype_b: Data types for A and B ('float4', 'float8_e5m2', 'float8_e4m3')
        num_warps: Number of warps per block
        group_size_m: Number of programs per group for L2 cache efficiency

    Returns:
        Tuple of (output_tensor, reference_tensor, max_diff)
    """
    # Validate configuration
    if BLOCK_M < PRESHUFFLE_FACTOR or BLOCK_N < PRESHUFFLE_FACTOR:
        raise ValueError(f"BLOCK_M and BLOCK_N must be >= {PRESHUFFLE_FACTOR} for scale preshuffling")

    torch.manual_seed(42)

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

    # Pre-shuffle scales (always enabled)
    a_scale_input = pack_scale(a_scale, PRESHUFFLE_FACTOR)
    b_scale_input = pack_scale(b_scale, PRESHUFFLE_FACTOR)

    # Move to GPU
    a_d = a.data.contiguous().cuda()
    b_d = b.data.contiguous().cuda()
    a_scale_d = a_scale_input.cuda()
    b_scale_d = b_scale_input.cuda()
    c_d = torch.zeros(M, N, dtype=torch.float32, device='cuda')

    # Launch kernel
    num_blocks = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (num_blocks, )

    mxgemm_kernel[grid](
        a_d, b_d, c_d,  #
        a_scale_d, b_scale_d,  #
        M, N, K,  #
        a_scale_d.stride(0),  #
        a_d.stride(0), a_d.stride(1),  #
        b_d.stride(0), b_d.stride(1),  #
        c_d.stride(0), c_d.stride(1),  #
        dtype_to_triton_type[dtype_a],  #
        dtype_to_triton_type[dtype_b],  #
        SCALE_BLOCK,  #
        BLOCK_M, BLOCK_N, BLOCK_K,  #
        group_size_m,  #
        num_warps=num_warps,  #
    )

    torch.cuda.synchronize()

    # Verify
    c_d_cpu = c_d.cpu()
    max_diff = (c_d_cpu - c_ref).abs().max().item()

    return c_d_cpu, c_ref, max_diff


# ============================================================================
# Pytest Test Functions
# ============================================================================


@pytest.mark.parametrize("M, N, K", [(256, 256, 512)])
@pytest.mark.parametrize("BM, BN, BK", [(128, 128, 128)])
@pytest.mark.parametrize("dtype_a", ['float8_e5m2', 'float8_e4m3', 'float4'])
@pytest.mark.parametrize("dtype_b", ['float8_e5m2', 'float8_e4m3', 'float4'])
@pytest.mark.skipif(not is_hip_gfx1250(), reason="Scaled dot with TDM is only tested on gfx1250.")
def test_mxgemm(M, N, K, BM, BN, BK, dtype_a, dtype_b):
    """Test MXFP GEMM with descriptor loads and pre-shuffled scales."""

    output, ref, max_diff = run_mxfp_gemm(M, N, K,  #
                                          BM, BN, BK,  #
                                          dtype_a, dtype_b,  #
                                          num_warps=4,  #
                                          group_size_m=1)

    torch.testing.assert_close(output, ref, rtol=1e-5, atol=0.02)


def main():
    parser = argparse.ArgumentParser(description='MXFP GEMM Kernel')

    parser.add_argument('-M', type=int, default=1024, help='M dimension')
    parser.add_argument('-N', type=int, default=1024, help='N dimension')
    parser.add_argument('-K', type=int, default=512, help='K dimension')
    parser.add_argument('-block_m', type=int, default=128, help='BLOCK_M')
    parser.add_argument('-block_n', type=int, default=128, help='BLOCK_N')
    parser.add_argument('-block_k', type=int, default=128, help='BLOCK_K')
    parser.add_argument('--dtype_a', type=str, default='float8_e5m2', choices=['float4', 'float8_e5m2', 'float8_e4m3'])
    parser.add_argument('--dtype_b', type=str, default='float8_e5m2', choices=['float4', 'float8_e5m2', 'float8_e4m3'])
    parser.add_argument('--num_warps', type=int, default=4, help='Number of warps')
    parser.add_argument('--group_size_m', type=int, default=1, help='Number of programs per group')

    args = parser.parse_args()

    print(f"Running MXFP GEMM: {args.dtype_a} x {args.dtype_b}")
    print(f"  Dimensions: M={args.M}, N={args.N}, K={args.K}")
    print(f"  Block sizes: block_m={args.block_m}, block_n={args.block_n}, block_k={args.block_k}")
    print("  Mode: Descriptor loads with pre-shuffled scales")

    output, ref, max_diff = run_mxfp_gemm(args.M, args.N, args.K,  #
                                          args.block_m, args.block_n, args.block_k,  #
                                          args.dtype_a, args.dtype_b,  #
                                          num_warps=args.num_warps,  #
                                          group_size_m=args.group_size_m,  #
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
