import pytest
import torch

import triton
import triton.language as tl
import triton.testing_autows as utils
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


@triton.jit
def block_scaled_matmul_kernel(
        a_ptr, a_scale,
        b_ptr, b_scale,
        output_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        stride_sk: tl.constexpr, stride_sb: tl.constexpr, stride_sc: tl.constexpr, stride_sd: tl.constexpr,
        output_type: tl.constexpr,
        ELEM_PER_BYTE: tl.constexpr,
        VEC_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        rep_m: tl.constexpr,
        rep_n: tl.constexpr,
        rep_k: tl.constexpr,
        NUM_STAGES: tl.constexpr,
        USE_DEVICE_TMA_DESC: tl.constexpr,
        ):

    if ELEM_PER_BYTE == 1:
        dtype = tl.float8e4nv
        scale_dtype = tl.dtype("uint8")
    elif ELEM_PER_BYTE == 2:
        dtype = tl.dtype("uint8")
        if VEC_SIZE == 16:
            scale_dtype = tl.float8e4nv
        else:
            scale_dtype = tl.dtype("uint8")

    if output_type == 0:
        output_dtype = tl.float32
    elif output_type == 1:
        output_dtype = tl.float16
    elif output_type == 2:
        output_dtype = tl.float8e4nv

    if USE_DEVICE_TMA_DESC:
        a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K // ELEM_PER_BYTE], strides=[K // ELEM_PER_BYTE, 1], block_shape=[BLOCK_M, BLOCK_K // ELEM_PER_BYTE])
        b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K // ELEM_PER_BYTE], strides=[K // ELEM_PER_BYTE, 1], block_shape=[BLOCK_N, BLOCK_K // ELEM_PER_BYTE])
        c_desc = tl.make_tensor_descriptor(output_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    else:
        a_desc = a_ptr
        b_desc = b_ptr
        c_desc = output_ptr
        # FIXME: borken after rebaser
        #tl.inline_asm_elementwise("prefetch.tensormap [$1]; // dummy $0", "=r,l", [a_desc], dtype=tl.int32, is_pure=False, pack=1)
        #tl.inline_asm_elementwise("prefetch.tensormap [$1]; // dummy $0", "=r,l", [b_desc], dtype=tl.int32, is_pure=False, pack=1)
        #tl.inline_asm_elementwise("prefetch.tensormap [$1]; // dummy $0", "=r,l", [c_desc], dtype=tl.int32, is_pure=False, pack=1)

    if USE_DEVICE_TMA_DESC:
        a_scale_desc = tl.make_tensor_descriptor(a_scale, shape=[M // 128, K // VEC_SIZE // 4, 32, 16], strides=[stride_sk, stride_sb, 16, 1], block_shape=[rep_m, rep_k, 32, 16])
        b_scale_desc = tl.make_tensor_descriptor(b_scale, shape=[N // 128, K // VEC_SIZE // 4, 32, 16], strides=[stride_sk, stride_sb, 16, 1], block_shape=[rep_n, rep_k, 32, 16])
    else:
        # FIXME: borken after rebase
        a_scale_desc = a_scale
        b_scale_desc = b_scale
        #tl.inline_asm_elementwise("prefetch.tensormap [$1]; // dummy $0", "=r,l", [a_scale_desc], dtype=tl.int32, is_pure=False, pack=1)
        #tl.inline_asm_elementwise("prefetch.tensormap [$1]; // dummy $0", "=r,l", [b_scale_desc], dtype=tl.int32, is_pure=False, pack=1)

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = pid_m * BLOCK_M
    offs_bn = pid_n * BLOCK_N
    offs_k = 0

    offs_scale_m = pid_m * rep_m
    offs_scale_n = pid_n * rep_n
    offs_scale_k = 0

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])

        scale_a = a_scale_desc.load([offs_scale_m, offs_scale_k, 0, 0])
        scale_b = b_scale_desc.load([offs_scale_n, offs_scale_k, 0, 0])

        scale_a = scale_a.reshape(rep_m, rep_k, 32, 4, 4)
        scale_b = scale_b.reshape(rep_n, rep_k, 32, 4, 4)

        scale_a = scale_a.trans(0, 3, 2, 1, 4).reshape(BLOCK_M, BLOCK_K // VEC_SIZE)
        scale_b = scale_b.trans(0, 3, 2, 1, 4).reshape(BLOCK_N, BLOCK_K // VEC_SIZE)

        if ELEM_PER_BYTE == 2:
            accumulator = tl.dot_scaled(a, scale_a, "e2m1", b.T, scale_b, "e2m1", accumulator)
        else:
            accumulator = tl.dot_scaled(a, scale_a, "e4m3", b.T, scale_b, "e4m3", accumulator)
        offs_k += BLOCK_K // ELEM_PER_BYTE
        offs_scale_k += rep_k

    c_desc.store([offs_am, offs_bn], accumulator.to(output_dtype))


def prepare_inputs_and_reference(M, N, K, block_scale_type, use_device_tma_desc, num_stages):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    ELEM_PER_BYTE = 2 if "fp4" in block_scale_type else 1

    device = "cuda"
    a_ref = MXFP4Tensor(size=(M, K), device=device).random()
    b_ref = MXFP4Tensor(size=(N, K), device=device).random()
    if block_scale_type == "mxfp8":
        a_ref = a_ref.to(torch.float32)
        b_ref = b_ref.to(torch.float32)
        a = a_ref.to(torch.float8_e4m3fn)
        b = b_ref.to(torch.float8_e4m3fn)
    else:
        a = a_ref.to_packed_tensor(dim=1)
        b = b_ref.to_packed_tensor(dim=1)
    b_ref = b_ref.to(torch.float32).T

    a_scale_shape = [M // 128, K // VEC_SIZE // 4, 32, 16]
    b_scale_shape = [N // 128, K // VEC_SIZE // 4, 32, 16]

    epsilon = 1e-8
    a_scale = torch.rand(a_scale_shape, device=device) + epsilon
    b_scale = torch.rand(b_scale_shape, device=device) + epsilon
    if block_scale_type == "nvfp4":
        a_scale = a_scale.to(torch.float8_e4m3fn)
        b_scale = b_scale.to(torch.float8_e4m3fn)
        a_scale_ref = a_scale
        b_scale_ref = b_scale
    elif block_scale_type in ["mxfp4", "mxfp8"]:
        a_scale_ref = MXScaleTensor(a_scale)
        b_scale_ref = MXScaleTensor(b_scale)
        a_scale = a_scale_ref.data
        b_scale = b_scale_ref.data

    rep_m = BLOCK_M // 128
    rep_n = BLOCK_N // 128
    rep_k = BLOCK_K // VEC_SIZE // 4

    if not use_device_tma_desc:
        a_desc = TensorDescriptor(a, a.shape, a.stride(), [BLOCK_M, BLOCK_K // ELEM_PER_BYTE])
        b_desc = TensorDescriptor(b, b.shape, b.stride(), [BLOCK_N, BLOCK_K // ELEM_PER_BYTE])
        a_scale_desc = TensorDescriptor(a_scale, a_scale.shape, a_scale.stride(), [rep_m, rep_k] + list(a_scale_shape)[-2:])
        b_scale_desc = TensorDescriptor(b_scale, b_scale.shape, b_scale.stride(), [rep_n, rep_k] + list(b_scale_shape)[-2:])


    a_scale_ref = a_scale_ref.to(torch.float32)
    b_scale_ref = b_scale_ref.to(torch.float32)
    def unpack_scale(packed):
        packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)
        num_chunk_m, num_chunk_k, _, _, _ = packed.shape
        return packed.permute(0, 3, 2, 1, 4).reshape(num_chunk_m * 128, num_chunk_k * 4).contiguous()
    a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=1)[:M, :K]
    b_scale_ref = unpack_scale(b_scale_ref).repeat_interleave(VEC_SIZE, dim=1).T.contiguous()[:K, :N]
    ref_output = torch.matmul(a_ref.to(torch.float32) * a_scale_ref, b_ref * b_scale_ref)

    configs = {
        "BLOCK_SIZE_M": BLOCK_M,
        "BLOCK_SIZE_N": BLOCK_N,
        "BLOCK_SIZE_K": BLOCK_K,
        "ELEM_PER_BYTE": ELEM_PER_BYTE,
        "VEC_SIZE": VEC_SIZE,
        "num_stages": num_stages,
    }
    if use_device_tma_desc:
        return a, a_scale, b, b_scale, ref_output, rep_m, rep_n, rep_k, configs
    else:
        return a_desc, a_scale_desc, b_desc, b_scale_desc, ref_output, rep_m, rep_n, rep_k, configs


@pytest.mark.skipif(not (triton.runtime.driver.active.get_current_target().backend == "cuda" and
                         torch.cuda.get_device_capability()[0] == 10),
                   reason="Requires CUDA with compute capability 10.x")
@pytest.mark.parametrize("M", [8192])
@pytest.mark.parametrize("N", [8192])
@pytest.mark.parametrize("K", [512, 8192])
@pytest.mark.parametrize("output_dtype", [torch.float32, torch.float16], ids=["float32", "float16"])
@pytest.mark.parametrize("block_scale_type", ["nvfp4", "mxfp4", "mxfp8"])
@pytest.mark.parametrize("use_device_tma_desc", [True, False], ids=["device_tma", "host_tma"])
@pytest.mark.parametrize("ENABLE_WARP_SPECIALIZATION", [True, False], ids=["aws", "swp"])
@pytest.mark.parametrize("NUM_WARPS", [4])
@pytest.mark.parametrize("NUM_STAGES", [3])
def test_block_scaled_matmul(
    M,
    N,
    K,
    output_dtype,
    block_scale_type,
    use_device_tma_desc,
    ENABLE_WARP_SPECIALIZATION,
    NUM_WARPS,
    NUM_STAGES,
):
    VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
    if K % VEC_SIZE != 0 and "fp4" in block_scale_type:
        pytest.skip(f"K dimension ({K}) must be multiple of vector size ({VEC_SIZE}) for {block_scale_type}")

    utils.common_test_setup(ENABLE_WARP_SPECIALIZATION, NUM_WARPS)

    if use_device_tma_desc:
        def alloc_fn(size: int, align: int, stream=None):
            return torch.empty(size, dtype=torch.uint8, device="cuda")
        triton.set_allocator(alloc_fn)

    a, a_scale, b, b_scale, ref_output, rep_m, rep_n, rep_k, configs = prepare_inputs_and_reference(
        M, N, K, block_scale_type, use_device_tma_desc, NUM_STAGES
    )

    output = torch.empty((M, N), dtype=output_dtype, device="cuda")
    if output_dtype == torch.float32:
        output_dtype = 0
    elif output_dtype == torch.float16:
        output_dtype = 1
    elif output_dtype == torch.float8_e4m3fn:
        output_dtype = 2
    else:
        raise ValueError(f"Unsupported output dtype: {output_dtype}")


    if not use_device_tma_desc:
        c_desc = TensorDescriptor(output, output.shape, output.stride(), [configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"]])

    if use_device_tma_desc:
        scale_strides = [a_scale.stride(0), a_scale.stride(1), a_scale.stride(2), a_scale.stride(3)]
    else:
        scale_strides = [0] * 4

    grid = (triton.cdiv(M, configs["BLOCK_SIZE_M"]) * triton.cdiv(N, configs["BLOCK_SIZE_N"]), 1, 1)

    block_scaled_matmul_kernel[grid](
        a, a_scale, b, b_scale,
        output if use_device_tma_desc else c_desc,
        M, N, K,
        *scale_strides, output_dtype,
        configs["ELEM_PER_BYTE"], configs["VEC_SIZE"],
        configs["BLOCK_SIZE_M"], configs["BLOCK_SIZE_N"], configs["BLOCK_SIZE_K"],
        rep_m, rep_n, rep_k, NUM_STAGES,
        USE_DEVICE_TMA_DESC=use_device_tma_desc,
        num_warps=NUM_WARPS,
        enable_warp_specialization=ENABLE_WARP_SPECIALIZATION,
    )
    torch.testing.assert_close(ref_output, output.to(torch.float32), atol=1e-3, rtol=1e-3)
