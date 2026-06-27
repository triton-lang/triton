import pytest
import torch

import triton
import triton.language as tl
from test_core import check_type_supported


@triton.jit
def block_copy_kernel(a_ptr, b_ptr, N, BLOCK_SIZE: tl.constexpr, PADDING_OPTION: tl.constexpr,
                      TEST_LOWER_BOUND: tl.constexpr, TEST_UPPER_BOUND: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE
    if TEST_LOWER_BOUND:
        offset = -N
    elif TEST_UPPER_BOUND:
        offset = N
    # We only copy half of the data to see if the padding works
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(N // 2, ), strides=(1, ), offsets=(offset, ),
                                    block_shape=(BLOCK_SIZE, ), order=(0, ))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(N, ), strides=(1, ), offsets=(offset, ),
                                    block_shape=(BLOCK_SIZE, ), order=(0, ))
    if PADDING_OPTION is None:
        a = tl.load(a_block_ptr, boundary_check=(0, ))
    else:
        a = tl.load(a_block_ptr, boundary_check=(0, ), padding_option=PADDING_OPTION)
    tl.store(b_block_ptr, a, boundary_check=(0, ))


@pytest.mark.interpreter
@pytest.mark.parametrize("dtypes_str, n, padding_option, boundary_check", [  #
    (dtypes_str, n, padding, boundary_check)  #
    for dtypes_str in (("bool", "bool"), ("int16", "int16"), ("int32", "int32"), ("float16", "float16"),
                       ("float32", "float32"), ("bfloat16", "bfloat16"))
    for n in (64, 128, 256, 512, 1024)
    for padding in (None, "zero", "nan")  #
    for boundary_check in (None, "lower", "upper")
])
def test_block_copy(dtypes_str, n, padding_option, boundary_check, device):
    src_dtype_str = dtypes_str[0]
    dst_dtype_str = dtypes_str[1]
    src_dtype = getattr(torch, src_dtype_str)
    dst_dtype = getattr(torch, dst_dtype_str)
    check_type_supported(src_dtype, device)
    check_type_supported(dst_dtype, device)
    if src_dtype_str in ("bool", "int16", "int32"):
        if padding_option == "nan":
            pytest.skip("Padding with NaN is not supported for integer types")
        a = torch.randint(0, 2, (n, ), device=device, dtype=src_dtype)
    else:
        a = torch.randn((n, ), device=device, dtype=src_dtype)
    b = torch.zeros((n, ), device=device, dtype=dst_dtype)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
    block_copy_kernel[grid](a_ptr=a, b_ptr=b, N=n, BLOCK_SIZE=64, PADDING_OPTION=padding_option,
                            TEST_LOWER_BOUND=boundary_check == "lower", TEST_UPPER_BOUND=boundary_check == "upper")
    a.to(dst_dtype)
    if (boundary_check == "lower") or (boundary_check == "upper"):
        assert torch.all(b == 0)
    else:
        assert torch.all(a[0:n // 2] == b[0:n // 2])
        if padding_option == "zero":
            assert torch.all(b[n // 2:n] == 0)
        elif padding_option == "nan":
            assert torch.all(torch.isnan(b[n // 2:n]))


@triton.jit
def matmul_no_scf_with_advance_kernel(  #
        a_ptr, b_ptr, c_ptr,  #
        M, N, K,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr  #
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak), offsets=(0, 0),
                                    block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn), offsets=(0, 0),
                                    block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))
    # Below two lines are just for testing negative offsets for the `advance` API, which could be removed
    a_block_ptr = tl.advance(a_block_ptr, (BLOCK_M, -BLOCK_K))
    a_block_ptr = tl.advance(a_block_ptr, (-BLOCK_M, BLOCK_K))
    a = tl.load(a_block_ptr, boundary_check=(1, ), padding_option="zero")
    b = tl.load(b_block_ptr, boundary_check=(0, ), padding_option="zero")

    c = tl.dot(a, b)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c)


@pytest.mark.interpreter
@pytest.mark.parametrize("shape, num_warps", [  #
    (shape, num_warps) for shape in [
        [64, 64, 16],
        [64, 64, 32],
        [64, 64, 64],
    ] for num_warps in [4, 8]
])
def test_block_ptr_matmul_no_scf(shape, num_warps, device):
    m, n, k = shape
    a = torch.randn((m, k), device=device, dtype=torch.float16)
    b = torch.randn((k, n), device=device, dtype=torch.float16)
    c = torch.empty((m, n), device=device, dtype=torch.float32)

    grid = lambda META: (1, )
    matmul_no_scf_with_advance_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,  #
        M=m, N=n, K=k,  #
        stride_am=a.stride(0), stride_ak=a.stride(1),  #
        stride_bk=b.stride(0), stride_bn=b.stride(1),  #
        stride_cm=c.stride(0), stride_cn=c.stride(1),  #
        BLOCK_M=m, BLOCK_N=n, BLOCK_K=k,  #
        num_warps=num_warps)
    golden = torch.matmul(a, b)
    torch.testing.assert_close(c, golden, check_dtype=False)


@triton.jit
def block_copy_3d_kernel(src_ptr, dst_ptr, dim0, dim1, dim2, stride0, stride1, stride2, BLOCK0: tl.constexpr,
                         BLOCK1: tl.constexpr, BLOCK2: tl.constexpr):
    src_block_ptr = tl.make_block_ptr(base=src_ptr, shape=(dim0, dim1, dim2), strides=(stride0, stride1, stride2),
                                      offsets=(dim0 - 1, 0, dim2 - 2), block_shape=(BLOCK0, BLOCK1, BLOCK2),
                                      order=(2, 1, 0))
    dst_block_ptr = tl.make_block_ptr(base=dst_ptr, shape=(dim0, dim1, dim2), strides=(stride0, stride1, stride2),
                                      offsets=(dim0 - 1, 0, dim2 - 2), block_shape=(BLOCK0, BLOCK1, BLOCK2),
                                      order=(2, 1, 0))
    values = tl.load(src_block_ptr, boundary_check=(0, 2), padding_option="zero")
    tl.store(dst_block_ptr, values, boundary_check=(0, 2))


@pytest.mark.interpreter
def test_block_ptr_rank_3(device):
    src = torch.arange(3 * 4 * 5, device=device, dtype=torch.float32).reshape(3, 4, 5)
    dst = torch.zeros_like(src)

    block_copy_3d_kernel[(1, )](src, dst, src.shape[0], src.shape[1], src.shape[2], src.stride(0), src.stride(1),
                                src.stride(2), BLOCK0=2, BLOCK1=4, BLOCK2=4)

    expected = torch.zeros_like(src)
    expected[2:3, 0:4, 3:5] = src[2:3, 0:4, 3:5]
    torch.testing.assert_close(dst, expected)


@triton.jit
def block_ptr_dtype_kernel(src_ptr, dst_ptr, N, BLOCK_SIZE: tl.constexpr):
    block_ptr = tl.make_block_ptr(base=src_ptr, shape=(N, ), strides=(1, ), offsets=(0, ), block_shape=(BLOCK_SIZE, ),
                                  order=(0, ))
    values = tl.load(block_ptr, boundary_check=(0, ), padding_option="zero").to(block_ptr.dtype.element_ty)
    tl.store(dst_ptr + tl.arange(0, BLOCK_SIZE), values)


@pytest.mark.interpreter
def test_block_ptr_dtype_element_ty(device):
    src = torch.arange(16, device=device, dtype=torch.float16)
    dst = torch.zeros_like(src)

    block_ptr_dtype_kernel[(1, )](src, dst, src.numel(), BLOCK_SIZE=16)

    torch.testing.assert_close(dst, src)
