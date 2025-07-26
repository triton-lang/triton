import triton.profiler as proton

import torch
import sys

from helper_kernels import custom_add, matmul_kernel


def main():
    a = torch.zeros(1, device="cuda")
    with proton.scope("test"):
        custom_add[(1, )](a)


def test_main():
    main()


def matmul():
    a = torch.randn((32, 32), device="cuda", dtype=torch.float16)
    b = torch.randn((32, 32), device="cuda", dtype=torch.float16)
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    matmul_kernel[(1, )](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        128, 256, 64, 8)
    return c


if __name__ == "__main__":
    if sys.argv[1] == "test":
        main()
    elif sys.argv[1] == "test_matmul":
        matmul()
