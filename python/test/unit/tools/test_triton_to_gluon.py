import sys
import importlib.util
import torch
import triton
import triton.language as tl
import pytest
from triton.tools.tensor_descriptor import TensorDescriptor

from triton.tools.triton_to_gluon_translater.translator import convert_triton_to_gluon
from triton.tools.triton_to_gluon_translater.translator_helpers import convert_host_descriptor
from triton._internal_testing import (
    is_blackwell, )


def convert_kernel(kernel, kernel_name, tmp_path):
    converted = convert_triton_to_gluon(kernel)

    # Write converted kernel to a file so @gluon.jit can retrieve source
    mod_path = tmp_path / "converted_kernel.py"
    mod_path.write_text(converted)

    spec = importlib.util.spec_from_file_location("converted_kernel", mod_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["converted_kernel"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    kernel = getattr(module, kernel_name)
    return kernel


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    y = tl.load(y_ptr + offsets)
    tl.store(out_ptr + offsets, x + y)


@pytest.mark.skipif(not (is_blackwell()), reason="Requires Blackwell")
def test_simple_kernel(tmp_path):
    kernel = convert_kernel(add_kernel, "add_kernel", tmp_path)

    n = 1024
    BLOCK = 128
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (n // BLOCK, )
    kernel[grid](x, y, out, n, BLOCK)

    ref = torch.empty_like(x)
    add_kernel[grid](x, y, ref, n, BLOCK)

    torch.testing.assert_close(out, ref)


@triton.jit
def impl_matmul_tile_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)[:, None]
    offs_n = tl.arange(0, N)[None, :]
    acc = tl.zeros((M, N), dtype=tl.float32)
    a = tl.load(a_ptr + offs_m * K + (tl.arange(0, K))[None, :])
    b = tl.load(b_ptr + (tl.arange(0, K))[:, None] * N + offs_n)
    acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m * N + offs_n, acc)


@triton.jit
def matmul_tile_kernel(a_ptr, b_ptr, c_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    impl_matmul_tile_kernel(a_ptr, b_ptr, c_ptr, BLOCK_M, BLOCK_N, BLOCK_K)


@pytest.mark.skipif(not (is_blackwell()), reason="Requires Blackwell")
def test_triton_to_gluon_dot_minimal(tmp_path):
    # Convert directly from the Triton kernel object
    kernel = convert_kernel(matmul_tile_kernel, "matmul_tile_kernel", tmp_path)
    M, N, K = 128, 128, 128
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    grid = (1, )

    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    kernel[grid](a, b, c, M, N, K, num_warps=8)

    ref = torch.empty_like(c)
    matmul_tile_kernel[grid](a, b, ref, M, N, K, num_warps=8)
    torch.testing.assert_close(c, ref)


@triton.jit
def matmul_kernel(  #
    a_ptr,
    b_ptr,
    output_ptr,  #
    M,
    N,
    K,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  #
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), step=1, num_stages=4):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, acc=accumulator, out_dtype=output_ptr.dtype.element_ty)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(output_ptrs, accumulator)


@pytest.mark.parametrize("dtype_src_str", ["float16"])
@pytest.mark.parametrize("dtype_dst_str", ["float32"])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES", [(128, 128, 64, 1)])
@pytest.mark.parametrize("NUM_WARPS", [4])
@pytest.mark.skipif(not (is_blackwell()), reason="Requires Blackwell")
def test_simple_matmul(dtype_src_str, dtype_dst_str, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, NUM_WARPS, tmp_path):
    device = "cuda"
    M, N, K = 1024, 512, 256
    torch.manual_seed(42)
    dtype_src_str = "float32" if dtype_src_str == "tensorfloat32" else dtype_src_str
    dtype_src = getattr(torch, dtype_src_str)

    kernel = convert_kernel(matmul_kernel, "matmul_kernel", tmp_path)

    a = torch.randn(M, K, dtype=dtype_src, device=device)
    b = torch.randn(K, N, dtype=dtype_src, device=device)
    dtype_dst = getattr(torch, dtype_dst_str)
    output = torch.empty((M, N), dtype=dtype_dst, device=device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    kernel[grid](a, b, output, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), output.stride(0),
                 output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K)

    ref = torch.empty_like(output)
    matmul_kernel[grid](a, b, ref, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), output.stride(0),
                        output.stride(1), BLOCK_M, BLOCK_N, BLOCK_K)
    torch.testing.assert_close(output, ref)


@triton.jit
def descriptor_store_kernel(desc, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, V: tl.constexpr):
    tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16) + V
    desc.store([0, 0], tile)


@pytest.mark.skipif(not (is_blackwell()), reason="Requires Blackwell")
def test_triton_to_gluon_descriptor_roundtrip(tmp_path):
    kernel = convert_kernel(descriptor_store_kernel, "descriptor_store_kernel", tmp_path)

    M = N = 64
    y = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    grid = (1, )
    block_shape = [M, N]
    desc = TensorDescriptor(y, y.shape, y.stride(), block_shape)
    gluon_desc = convert_host_descriptor(desc)
    kernel[grid](gluon_desc, M, N, 1.0)

    y_ref = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    desc_ref = TensorDescriptor(y_ref, y_ref.shape, y_ref.stride(), block_shape)
    descriptor_store_kernel[grid](desc_ref, M, N, 1.0)
    torch.testing.assert_close(y, y_ref)


@triton.jit
def descriptor_copy_kernel(in_desc, out_desc, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    tile = in_desc.load([0, 0])
    out_desc.store([0, 0], tile)


@pytest.mark.skipif(not (is_blackwell()), reason="Requires Blackwell")
def test_triton_to_gluon_descriptor_load_roundtrip(tmp_path):
    kernel = convert_kernel(descriptor_copy_kernel, "descriptor_copy_kernel", tmp_path)

    M = N = 64
    x = torch.ones((M, N), device="cuda", dtype=torch.float16) * 3.0
    y = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    grid = (1, )
    block_shape = [M, N]

    in_desc = TensorDescriptor(x, x.shape, x.stride(), block_shape)
    gluon_desc = convert_host_descriptor(in_desc)
    out_desc = convert_host_descriptor(TensorDescriptor(y, y.shape, y.stride(), block_shape))
    kernel[grid](gluon_desc, out_desc, M, N)

    y_ref = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    desc_ref = TensorDescriptor(y_ref, y_ref.shape, y_ref.stride(), block_shape)
    descriptor_copy_kernel[grid](in_desc, desc_ref, M, N)
    torch.testing.assert_close(y, y_ref)


@triton.jit
def reshape_trans_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)

    x = tl.reshape(tl.load(x_ptr + offsets), 16, 16)
    y = tl.load(y_ptr + offsets).reshape(16, 16)
    a = x + y.trans(1, 0)
    a = a.reshape(256)
    tl.store(out_ptr + offsets, a)


@pytest.mark.skipif(not (is_blackwell()), reason="Requires Blackwell")
def test_triton_reshape_trans(tmp_path):
    kernel = convert_kernel(reshape_trans_kernel, "reshape_trans_kernel", tmp_path)

    n = 1024
    BLOCK = 256
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (n // BLOCK, )
    kernel[grid](x, y, out, n, BLOCK)
    ref = torch.empty_like(x)
    reshape_trans_kernel[grid](x, y, ref, n, BLOCK)
    torch.testing.assert_close(out, ref)


BLOCK_SPLIT = tl.constexpr(256)


@triton.jit
def split_kernel(x_ptr, out_ptr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SPLIT + tl.arange(0, BLOCK_SPLIT)
    offsets2 = pid * BLOCK_SPLIT + tl.arange(0, 2 * BLOCK_SPLIT)

    s0, s1 = tl.reshape(tl.load(x_ptr + offsets2), BLOCK_SPLIT, 2).split()
    a = s0 + s1
    p = out_ptr + offsets
    tl.store(p, a)


@pytest.mark.skipif(not (is_blackwell()), reason="Requires Blackwell")
def test_split(tmp_path):
    kernel = convert_kernel(split_kernel, "split_kernel", tmp_path)

    n = 1024
    x = torch.randn(2 * n, device="cuda", dtype=torch.float32)
    grid = (n // BLOCK_SPLIT, )

    out = torch.empty_like(x[:n])
    kernel[grid](x, out)
    ref = torch.empty_like(x[:n])
    split_kernel[grid](x, ref)
    torch.testing.assert_close(out, ref)


@triton.jit
def reduce_to_scalar_kernel(out_ptr):
    x = tl.arange(0, 16)
    x = tl.sum(x)
    tl.store(out_ptr, x)


@pytest.mark.skipif(not (is_blackwell()), reason="Requires Blackwell")
def test_reduce_to_scalar(tmp_path):
    kernel = convert_kernel(reduce_to_scalar_kernel, "reduce_to_scalar_kernel", tmp_path)
    grid = (1, )

    out = torch.empty((1, ), device="cuda", dtype=torch.int32)
    kernel[grid](out)
    ref = torch.empty_like(out)
    reduce_to_scalar_kernel[grid](ref)
    torch.testing.assert_close(out, ref)
