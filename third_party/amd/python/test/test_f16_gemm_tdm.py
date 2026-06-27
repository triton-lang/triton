import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_hip_gfx1250
from triton.tools.tensor_descriptor import TensorDescriptor


def supports_tensor_descriptor():
    # AMD GPUs with tensor ops support
    return is_hip_gfx1250() and hasattr(tl, "make_tensor_descriptor")


@triton.jit
def gemm_device_tdm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                           BLOCK_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

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

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = a_desc.load([0, k])
        b = b_desc.load([k, 0])
        accumulator = tl.dot(a, b, acc=accumulator)

    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], accumulator)


@triton.jit
def gemm_host_tdm_kernel(a_desc, b_desc, c_desc, M, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr):
    # GEMM with tensor descriptors as arguments
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_cm = pid_m * BLOCK_M
    offs_cn = pid_n * BLOCK_N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = a_desc.load([offs_cm, k])
        b = b_desc.load([k, offs_cn])
        accumulator = tl.dot(a, b, acc=accumulator)

    c_desc.store([offs_cm, offs_cn], accumulator)


def _run_kernel(x, y, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, use_tdm='device'):
    z = torch.empty(M, N, dtype=torch.float32).cuda()
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    if use_tdm == 'host':
        a_desc = TensorDescriptor.from_tensor(x, (BLOCK_M, BLOCK_K))
        b_desc = TensorDescriptor.from_tensor(y, (BLOCK_K, BLOCK_N))
        c_desc = TensorDescriptor.from_tensor(z, (BLOCK_M, BLOCK_N))
        gemm_host_tdm_kernel[grid](a_desc, b_desc, c_desc, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    else:  # use_tdm == 'device'
        gemm_device_tdm_kernel[grid](x, y, z, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    return z


@pytest.mark.parametrize("M,N,K", [(256, 256, 512)])
@pytest.mark.parametrize("BLOCK_M,BLOCK_N,BLOCK_K", [(32, 32, 64), (128, 128, 128)])
@pytest.mark.parametrize("use_tdm", ['device', 'host'])
@pytest.mark.skipif(not is_hip_gfx1250(), reason="TDM is only tested on gfx1250.")
def test_gemm_fp16(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, use_tdm):
    x = torch.randn(M, K, dtype=torch.float16).cuda()
    y = torch.randn(K, N, dtype=torch.float16).cuda()

    z_ref = torch.matmul(x.to(torch.float32), y.to(torch.float32))
    z = _run_kernel(x, y, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, use_tdm)

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
    parser.add_argument("--tdm", type=str, default='device', choices=['device', 'host'],
                        help='Tensor descriptor mode: "device" or "host" (default: device)')
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K
    BLOCK_M, BLOCK_N, BLOCK_K = args.block_m, args.block_n, args.block_k
    use_tdm = args.tdm

    test_gemm_fp16(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, use_tdm)
