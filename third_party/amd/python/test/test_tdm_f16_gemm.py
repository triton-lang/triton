import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_hip_gfx1250, is_hip_cdna4
from triton.tools.tensor_descriptor import TensorDescriptor

def supports_tensor_descriptor():
    # AMD GPUs with tensor ops support
    return (is_hip_gfx1250() or is_hip_cdna4()) and hasattr(tl, "make_tensor_descriptor")


HAS_TENSOR_DESC = supports_tensor_descriptor()
# Todo: Enable this kernel when host tensor descriptor lowering is fully implemented
HAS_HOST_TENSOR_DESC = supports_tensor_descriptor() and False


@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                USE_TENSOR_DESC: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    if USE_TENSOR_DESC:
        a_desc = tl.make_tensor_descriptor(
            base=a_ptr + pid_m * BLOCK_M * K,
            shape=(M, K),
            strides=(K, 1),
            block_shape=(BLOCK_M, BLOCK_K),
        )
        b_desc = tl.make_tensor_descriptor(
            base=b_ptr + pid_n * BLOCK_N,
            shape=(K, N),
            strides=(N, 1),
            block_shape=(BLOCK_K, BLOCK_N),
        )
        c_desc = tl.make_tensor_descriptor(
            base=c_ptr,
            shape=(M, N),
            strides=(N, 1),
            block_shape=(BLOCK_M, BLOCK_N),
        )
    else:
        offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + (offs_am[:, None] * K + offs_k[None, :])
        b_ptrs = b_ptr + (offs_k[:, None] * N + offs_bn[None, :])

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        if USE_TENSOR_DESC:
            a = a_desc.load([0, k])
            b = b_desc.load([k, 0])
        else:
            # Apply mask for valid K
            mask_k = (offs_k + k) < K
            a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * N
        accumulator = tl.dot(a, b, acc=accumulator)

    if USE_TENSOR_DESC:
        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], accumulator)
    else:
        offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        c_ptrs = c_ptr + (offs_cm[:, None] * N + offs_cn[None, :])
        tl.store(c_ptrs, accumulator)


@triton.jit
def gemm_tdm_kernel(a_desc, b_desc, c_desc, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    BLOCK_K: tl.constexpr):
    # GEMM with tensor descriptors as arguments
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = a_desc.load([0, k])
        b = b_desc.load([k, 0])
        accumulator = tl.dot(a, b, acc=accumulator)

    offs_cm = pid_m * BLOCK_M
    offs_cn = pid_n * BLOCK_N
    c_desc.store([offs_cm, offs_cn], accumulator)


def _run_kernel(x, y, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, device_tensor_desc=False, host_tensor_desc=False):
    z = torch.empty(M, N, dtype=torch.float32).cuda()
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    if host_tensor_desc:
        a_desc = TensorDescriptor.from_tensor(x, (BLOCK_M, BLOCK_K))
        b_desc = TensorDescriptor.from_tensor(y, (BLOCK_K, BLOCK_N))
        c_desc = TensorDescriptor.from_tensor(z, (BLOCK_M, BLOCK_N))
        kernel = gemm_tdm_kernel[grid](a_desc, b_desc, c_desc, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    else:
        kernel = gemm_kernel[grid](x, y, z, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, device_tensor_desc)
    return z


@pytest.mark.parametrize("M,N,K", [(256, 256, 512), [256, 256, 455]])
@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(32, 32, 64)])
@pytest.mark.parametrize("device_tensor_desc, host_tensor_desc", [(True, False), (False, True), (False, False)])
def test_gemm_fp16(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, device_tensor_desc, host_tensor_desc):
    x = torch.randn(M, K, dtype=torch.float16).cuda()
    y = torch.randn(K, N, dtype=torch.float16).cuda()

    z_ref = torch.matmul(x.to(torch.float32), y.to(torch.float32))

    if device_tensor_desc and not HAS_TENSOR_DESC:
        pytest.skip("Skip unsupported test with device tensor descriptor")

    if host_tensor_desc and not HAS_HOST_TENSOR_DESC:
        pytest.skip("Skip unsupported test with host tensor descriptor")

    z = _run_kernel(x, y, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, device_tensor_desc, host_tensor_desc)

    assert torch.allclose(z_ref, z, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=256, help='problem M size')
    parser.add_argument("-N", type=int, default=256, help='problem N size')
    parser.add_argument("-K", type=int, default=1024, help='problem K size')
    parser.add_argument("--block_m", type=int, default=32, help='Block M size')
    parser.add_argument("--block_n", type=int, default=32, help='Block N size')
    parser.add_argument("--block_k", type=int, default=64, help='Block K size')
    parser.add_argument("--ptr_loads", action="store_true", help='Use device pointer')
    parser.add_argument("--tensor_desc_device", action="store_true", help='Use tensor descriptor on device')
    parser.add_argument("--tensor_desc_host", action="store_true", help='Use tensor descriptor on host')
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    BLOCK_M, BLOCK_N, BLOCK_K = args.block_m, args.block_n, args.block_k
    dev_desc = args.tensor_desc_device
    host_desc = args.tensor_desc_host

    if dev_desc and HAS_TENSOR_DESC:
        test_gemm_fp16(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, dev_desc, False)
    elif host_desc and HAS_HOST_TENSOR_DESC:
        test_gemm_fp16(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, False, host_desc)
    else:
        test_gemm_fp16(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, False, False)
