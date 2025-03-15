"""
Block Scaled Matrix Multiplication
==================================
This tutorial demonstrates a Triton implementation of block scaled matrix multiplication
which is generic over FP4 and FP8 formats. The formats supported in the tutorial are the OCP microscaling
formats, including mxfp4 and mxfp8, as well as NVIDIA's nvfp4 format. These matrix multiplications
are accelerated by fifth generation tensor core instructions on CUDA devices with compute capability 10.

Users can run the tutorial with each of the supported formats by passing the `--format`
argument and can benchmark the performance of each by specifying matrix dimensions
and iteration steps.

.. code-block:: bash

    # FP4
    python 10-block-scaled-matmul.py --format nvfp4
    python 10-block-scaled-matmul.py --format mxfp4 --K_range 512 8192 --bench

    # FP8
    python 10-block-scaled-matmul.py --format mxfp8 --K_range 8192 16384 --K_step 2048 --bench

Future updates to this tutorial which support mixed precision block scaled matmul are planned.
"""

# %%
# Background
# ----------
#
# CUDA devices that support PTX 8.7 and later can utlize block scaled matrix multiply
# instructions. In order for low latency access to these scale factors in the fast
# inner loop over tensor core MMAs, it is important to ensure that the blocked
# scale factors are stored in a contiguous memory layout according to their access
# pattern.
#
# The block scaled matmul tensor core instructions compute the following product:
#
#     C = (A * scale_a) @ (B * scale_b)
#
# where scale_a and scale_b are the blocked scale factors for the A and B matrices.
# Under block scaled matmul, each scale factor is broadcast and multiplied across a
# vector of elements from the A and B matrices, usually along their respective K axes.
# The number of elements of A and B over which each scale factor is broadcast is herein
# refered to as the vector size (VEC_SIZE).
#
# In a linear row-major layout, the scale factors would take the shape
#
#     (M, K // VEC_SIZE) and (N, K // VEC_SIZE)   [1]
#
# in global memory. However, to avoid non-contiguous memory access, it is beneficial to
# instead store the scale factors in a packed block layout. For the LHS matrix this layout
# is given by
#
#     (M // 32 // 4, K // VEC_SIZE // 4, 32, 4, 4)   [2].
#
# In this way, each tensor core MMA in the fast inner loop over K blocks can achieve contiguous
# access of a block of 128 rows of scale factors along the M axis, for each BLOCK_M x BLOCK_K
# subtile of the matrix A.
#
# In order to conform with Triton's language semantics for dot_scaled, the scale factors
# are prepared in the above 5D layout [2], but are then logically transposed and reshaped into
# the 2D layout [1] expected by tl.dot_scaled.
#
# For more detailed information on the scale factor layout, see
#  1. https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-1x
#  2. https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout
#

import argparse

import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
import triton.profiler as proton
from triton.tools.experimental_descriptor import TmaDescKernelParam
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_block_scaling():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    kernel_name = kernel.name
    if "ELEM_PER_BYTE_A" and "ELEM_PER_BYTE_B" and "VEC_SIZE" in args:
        if args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 1:
            kernel_name += "_mxfp8"
        elif args["ELEM_PER_BYTE_A"] == 1 and args["ELEM_PER_BYTE_B"] == 2:
            kernel_name += "_mixed"
        elif args["ELEM_PER_BYTE_A"] == 2 and args["ELEM_PER_BYTE_B"] == 2:
            if args["VEC_SIZE"] == 16:
                kernel_name += "_nvfp4"
            elif args["VEC_SIZE"] == 32:
                kernel_name += "_mxfp4"
    ret["name"] = f"{kernel_name} [M={M}, N={N}, K={K}]"
    ret["flops"] = 2. * M * N * K
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def block_scaled_matmul_kernel(  #
        a_desc, a_scale,  #
        b_desc_or_tensor, b_scale,  #
        c_desc,  #
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
        stride_sk: tl.constexpr, stride_sb: tl.constexpr, stride_sc: tl.constexpr, stride_sd: tl.constexpr,
        output_type: tl.constexpr,  #
        ELEM_PER_BYTE_A: tl.constexpr,  #
        ELEM_PER_BYTE_B: tl.constexpr,  #
        VEC_SIZE: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr,  #
        USE_2D_SCALE_LOAD: tl.constexpr):  #

    if ELEM_PER_BYTE_A == 1:
        dtype_a = tl.float8e4nv
    elif ELEM_PER_BYTE_A == 2:
        dtype_a = tl.dtype("uint8")

    if ELEM_PER_BYTE_B == 1:
        dtype_b = tl.float8e4nv
    elif ELEM_PER_BYTE_B == 2:
        dtype_b = tl.dtype("uint8")

    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e4nv

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k_a = 0
    offs_k_b = 0

    ## block scale offsets
    offs_sm = (pid_m * (BLOCK_M // 128) + tl.arange(0, BLOCK_M // 128)) % M
    offs_sn = (pid_n * (BLOCK_N // 128) + tl.arange(0, BLOCK_N // 128)) % N

    MIXED_PREC: tl.constexpr = ELEM_PER_BYTE_A == 1 and ELEM_PER_BYTE_B == 2

    if MIXED_PREC:
        b_desc = tl.make_tensor_descriptor(
            b_desc_or_tensor,
            shape=[N, K // ELEM_PER_BYTE_B],
            strides=[K // ELEM_PER_BYTE_B, 1],
            block_shape=[BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B],
        )
    else:
        b_desc = b_desc_or_tensor
        tl.inline_asm_elementwise("prefetch.tensormap [$1]; // dummy $0", "=r,l", [b_desc], dtype=tl.int32,
                                  is_pure=False, pack=1)

    tl.inline_asm_elementwise("prefetch.tensormap [$1]; // dummy $0", "=r,l", [a_desc], dtype=tl.int32, is_pure=False,
                              pack=1)
    tl.inline_asm_elementwise("prefetch.tensormap [$1]; // dummy $0", "=r,l", [c_desc], dtype=tl.int32, is_pure=False,
                              pack=1)

    # For now it is recommended to use 2D scale loads for better performance.
    # In the future we will bring additional optimizations to either allow 5D loads,
    # the use of TMAs for scale factors, or both.
    if USE_2D_SCALE_LOAD:
        offs_inner = tl.arange(0, (BLOCK_K // VEC_SIZE // 4) * 32 * 4 * 4)
        a_scale_ptr = a_scale + offs_sm[:, None] * stride_sk + offs_inner[None, :]
        b_scale_ptr = b_scale + offs_sn[:, None] * stride_sk + offs_inner[None, :]
    else:
        offs_sk = tl.arange(0, (BLOCK_K // VEC_SIZE // 4))
        # MN spatial offsets for 32 element blocking
        offs_sc = tl.arange(0, 32)
        # offsets for both scale factor column ID (along K)
        # and spatial block column ID (along MN)
        offs_sd = tl.arange(0, 4)
        a_scale_ptr = a_scale + (offs_sm[:, None, None, None, None] * stride_sk + offs_sk[None, :, None, None, None] *
                                 stride_sb + offs_sc[None, None, :, None, None] * stride_sc +
                                 offs_sd[None, None, None, :, None] * stride_sd + offs_sd[None, None, None, None, :])
        b_scale_ptr = b_scale + (offs_sn[:, None, None, None, None] * stride_sk + offs_sk[None, :, None, None, None] *
                                 stride_sb + offs_sc[None, None, :, None, None] * stride_sc +
                                 offs_sd[None, None, None, :, None] * stride_sd + offs_sd[None, None, None, None, :])

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl._experimental_descriptor_load(a_desc, [offs_am, offs_k_a], [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A],
                                             dtype_a)

        if MIXED_PREC:
            b = b_desc.load([offs_bn, offs_k_b])
        else:
            b = tl._experimental_descriptor_load(b_desc, [offs_bn, offs_k_b], [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B],
                                                 dtype_b)

        scale_a = tl.load(a_scale_ptr)
        scale_b = tl.load(b_scale_ptr)
        if USE_2D_SCALE_LOAD:
            scale_a = scale_a.reshape(BLOCK_M // 128, BLOCK_K // VEC_SIZE // 4, 32, 4, 4)
            scale_b = scale_b.reshape(BLOCK_N // 128, BLOCK_K // VEC_SIZE // 4, 32, 4, 4)
        scale_a = scale_a.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        if MIXED_PREC:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e2m1", accumulator)
        elif ELEM_PER_BYTE_A == 2 and ELEM_PER_BYTE_B == 2:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        else:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)

        offs_k_a += BLOCK_K // ELEM_PER_BYTE_A
        offs_k_b += BLOCK_K // ELEM_PER_BYTE_B
        a_scale_ptr += (BLOCK_K // VEC_SIZE // 4) * stride_sb
        b_scale_ptr += (BLOCK_K // VEC_SIZE // 4) * stride_sb

    tl._experimental_descriptor_store(c_desc, accumulator.to(output_dtype), [offs_am, offs_bn])


def block_scaled_matmul(a_desc, a_scale, b_desc_or_tensor, b_scale, dtype_dst, M, N, K, configs):
    output = torch.empty((M, N), dtype=dtype_dst, device="cuda")
    if dtype_dst == torch.float32:
        dtype_dst = 0
    elif dtype_dst == torch.float16:
        dtype_dst = 1
    elif dtype_dst == torch.float8_e4m3fn:
        dtype_dst = 2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_dst}")

    c_desc = TmaDescKernelParam(output.data_ptr(), output.shape, [configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"]],
                                output.element_size())

    grid = (triton.cdiv(M, configs["BLOCK_SIZE_M"]) * triton.cdiv(N, configs["BLOCK_SIZE_N"]), 1)
    block_scaled_matmul_kernel[grid](a_desc, a_scale, b_desc_or_tensor, b_scale, c_desc, M, N, K, a_scale.stride(0),
                                     a_scale.stride(1), a_scale.stride(2), a_scale.stride(3), dtype_dst,
                                     configs["ELEM_PER_BYTE_A"], configs["ELEM_PER_BYTE_B"], configs["VEC_SIZE"],
                                     configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"], configs["BLOCK_SIZE_K"],
                                     configs["num_stages"], USE_2D_SCALE_LOAD=True)
    return output


def initialize_block_scaled(M, N, K, block_scale_type="nvfp4", compute_reference=False):
    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 256 if "fp4" in block_scale_type else 128
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    assert block_scale_type in ["nvfp4", "mxfp4", "mxfp8", "mixed"], f"Invalid block scale type: {block_scale_type}"
    ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1
    ELEM_PER_BYTE_B = 1 if block_scale_type == "mxfp8" else 2

    device = "cuda"
    a_ref = MXFP4Tensor(size=(M, K), device=device).random()
    # Similar to Hopper's wgmma symmetric fp8 instruction, the RHS is expected
    # to be in col-major layout for Blackwell's tcgen05.mma when using fp4 operands.
    # To conform to the expected semantics of tl.dot_scaled, (M, K) x (K, N),
    # the data is generated in col-major layout, packed along K for fp4, and then
    # logically transposed. Note that if one operand is of fp8 precision, unlike Hopper,
    # Blackwell supports both row-major and col-major layouts for the RHS matrix.
    # For the mixed-precision case, the fp4 RHS can be either in row or col-major layout.
    # But for performance reason, it is recommended to use col-major layout. If TMA is used
    # for the fp4 RHS operand load in mixed-precision dot, as in this tutorial, it must be
    # in col-major layout.
    b_ref = MXFP4Tensor(size=(N, K), device=device).random()
    if block_scale_type in ["mxfp8", "mixed"]:
        a_ref = a_ref.to(torch.float32)
        a = a_ref.to(torch.float8_e4m3fn)
    else:
        # Pack two fp4 elements per byte along K
        a = a_ref.to_packed_tensor(dim=1)

    if block_scale_type == "mxfp8":
        b_ref = b_ref.to(torch.float32)
        b = b_ref.to(torch.float8_e4m3fn)
    else:
        b = b_ref.to_packed_tensor(dim=1)

    b_ref = b_ref.to(torch.float32).T

    a_desc = TmaDescKernelParam(a.data_ptr(), a.shape, [BLOCK_M, BLOCK_K // ELEM_PER_BYTE_A], 1)

    if block_scale_type == "mixed":
        b_desc_or_tensor = b
    else:
        b_desc_or_tensor = TmaDescKernelParam(b.data_ptr(), b.shape, [BLOCK_N, BLOCK_K // ELEM_PER_BYTE_B], 1)

    epsilon = 1e-8
    a_scale = torch.rand((M // 128, K // VEC_SIZE // 4, 32, 4, 4), device=device) + epsilon
    b_scale = torch.rand((N // 128, K // VEC_SIZE // 4, 32, 4, 4), device=device) + epsilon
    if block_scale_type == "nvfp4":
        a_scale = a_scale.to(torch.float8_e4m3fn)
        b_scale = b_scale.to(torch.float8_e4m3fn)
        a_scale_ref = a_scale
        b_scale_ref = b_scale
    elif block_scale_type in ["mxfp4", "mxfp8", "mixed"]:
        a_scale_ref = MXScaleTensor(a_scale)
        b_scale_ref = MXScaleTensor(b_scale)
        a_scale = a_scale_ref.data
        b_scale = b_scale_ref.data

    reference = None
    if compute_reference:
        a_scale_ref = a_scale_ref.to(torch.float32)
        b_scale_ref = b_scale_ref.to(torch.float32)

        def unpack_scale(packed):
            num_chunk_m, num_chunk_k, _, _, _ = packed.shape
            return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()

        a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
        b_scale_ref = unpack_scale(b_scale_ref).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
        reference = torch.matmul(a_ref.to(torch.float32) * a_scale_ref, b_ref * b_scale_ref)

    configs = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "num_stages": 4,
        "ELEM_PER_BYTE_A": ELEM_PER_BYTE_A,
        "ELEM_PER_BYTE_B": ELEM_PER_BYTE_B,
        "VEC_SIZE": VEC_SIZE,
    }
    return a_desc, a_scale, b_desc_or_tensor, b_scale, configs, reference


def validate_block_scaled(M, N, K, block_scale_type="nvfp4"):

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    if block_scale_type == "mixed":
        # This is needed for TMA with the descriptor created on the device.
        # TMA load for mixed-precision fp4 is supported only by device TMA.
        triton.set_allocator(alloc_fn)

    a_desc, a_scale, b_desc_or_tensor, b_scale, configs, reference = initialize_block_scaled(
        M, N, K, block_scale_type, compute_reference=True)
    output = block_scaled_matmul(a_desc, a_scale, b_desc_or_tensor, b_scale, torch.float16, M, N, K, configs)
    torch.testing.assert_close(reference, output.to(torch.float32), atol=1e-3, rtol=1e-3)
    print(f"✅ (pass {block_scale_type})")


def bench_block_scaled(K, block_scale_type="nvfp4", reps=10):
    assert K % 128 == 0
    M = 8192
    N = 8192
    print(f"Problem Shape = {M}x{N}x{K}")

    def alloc_fn(size: int, align: int, _):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    if block_scale_type == "mixed":
        triton.set_allocator(alloc_fn)

    a_desc, a_scale, b_desc_or_tensor, b_scale, configs, _ = initialize_block_scaled(
        M, N, K, block_scale_type, compute_reference=False)
    _ = block_scaled_matmul(a_desc, a_scale, b_desc_or_tensor, b_scale, torch.float16, M, N, K, configs)

    proton.activate(0)
    for _ in range(reps):
        _ = block_scaled_matmul(a_desc, a_scale, b_desc_or_tensor, b_scale, torch.float16, M, N, K, configs)
    proton.deactivate(0)
    print("Done benchmarking")


def show_profile(profile_name):
    import triton.profiler.viewer as proton_viewer
    metric_names = ["time/ms"]
    metric_names = ["tflop/s"] + metric_names
    file_name = f"{profile_name}.hatchet"
    tree, metrics = proton_viewer.parse(metric_names, file_name)
    proton_viewer.print_tree(tree, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--K_range", type=int, nargs=2)
    parser.add_argument("--K_step", type=int, default=512)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--format", type=str, choices=["mxfp4", "nvfp4", "mxfp8", "mixed"], default="nvfp4")
    args = parser.parse_args()

    if not supports_block_scaling():
        print("⛔ This example requires GPU support for block scaled matmul")
    else:
        torch.manual_seed(42)

        validate_block_scaled(8192, 8192, 8192, block_scale_type=args.format)

        if args.bench:
            proton.start("block_scaled_matmul", hook="triton")
            for K in range(args.K_range[0], args.K_range[1] + 1, args.K_step):
                bench_block_scaled(K, reps=10000, block_scale_type=args.format)
            proton.finalize()
            show_profile("block_scaled_matmul")
