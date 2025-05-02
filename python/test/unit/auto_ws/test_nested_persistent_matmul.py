import pytest
import torch
import argparse

import triton
import triton.language as tl
import triton.testing_autows as utils
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def matmul_persistent_tma_ws_cooperative_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    c_raw,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """

    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    for pid in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N
        offs_k = 0

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = a_ptr.load([offs_am, offs_k])
            b = b_ptr.load([offs_k, offs_bn])

            accumulator += tl.dot(a, b)
            offs_k += BLOCK_SIZE_K

        c = accumulator.to(tl.float16)

        if 1:
            c_ptr.store([offs_am, offs_bn],c)
        else:
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_raw + offs_cm[:, None] * N + offs_cn[None, :]
            mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(c_ptrs, c, mask)


@pytest.mark.parametrize("M", [128, 256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("N", [256, 512, 1024, 2048, 4096, 8192])
# FIXME: Fails accuracy check with K = 8192
@pytest.mark.parametrize("K", [128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("DTYPE", ["fp16"])
@pytest.mark.parametrize("BLOCK_M", [128])
@pytest.mark.parametrize("BLOCK_N", [256])
@pytest.mark.parametrize("BLOCK_K", [64])
@pytest.mark.parametrize("GROUP_M", [8])
@pytest.mark.parametrize(
    ("ENABLE_WARP_SPECIALIZATION", "NUM_WARPS", "NUM_STAGES", "MMA_DEPTH"),
    [(True, 4 if utils.is_sm10x() else 8, 3, 2)],
)
def test_matmul(
    M,
    N,
    K,
    DTYPE,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    GROUP_M,
    NUM_WARPS,
    NUM_STAGES,
    MMA_DEPTH,
    ENABLE_WARP_SPECIALIZATION,
    USE_TTG_WS=False,
    cmd_args=None,
    bench=False,
):
    utils.common_test_setup(ENABLE_WARP_SPECIALIZATION, NUM_WARPS)

    dtype = utils.torch_dtype(DTYPE)
    A = utils.generate_input((M, K), dtype)
    B = utils.generate_input((K, N), dtype)

    NUM_SMS = utils.get_num_sms()

    C = torch.empty((M, N), dtype=dtype, device="cuda")

    desc_a = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_M, BLOCK_K])
    desc_b = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_K, BLOCK_N])
    desc_c = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_M, BLOCK_N])

    def grid(META):
        return (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )

    def run():
        bytes_per_elem = A.element_size()
        flops_str = f"flops{bytes_per_elem * 8}"
        with proton.scope(
            f"matmul: [M={M}, N={N}, K={K}]",
            {
                "bytes": bytes_per_elem * (M * K + N * K + M * N),
                flops_str: 2.0 * M * N * K,
            },
        ):
            matmul_persistent_tma_ws_cooperative_kernel[grid](
                desc_a,
                desc_b,
                desc_c,
                C,
                M,
                N,
                K,
                BLOCK_SIZE_M=BLOCK_M,
                BLOCK_SIZE_N=BLOCK_N,
                BLOCK_SIZE_K=BLOCK_K,
                GROUP_SIZE_M=GROUP_M,
                num_stages=NUM_STAGES,
                num_warps=NUM_WARPS,
                mma_depth=MMA_DEPTH,
                enable_warp_specialization=ENABLE_WARP_SPECIALIZATION,
                use_ttg_ws=USE_TTG_WS,
            )

    if bench:
        utils.bench_fn(cmd_args, run)
    else:
        run()
        utils.verify_matmul(A, B.T.contiguous(), C)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    utils.common_bench_options(parser)

    parser.add_argument("-M", type=int, default=8192)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("--BLOCK_M", type=int, default=128)
    parser.add_argument("--BLOCK_N", type=int, default=256)
    parser.add_argument("--BLOCK_K", type=int, default=64)
    parser.add_argument("--GROUP_M", type=int, default=8)
    parser.add_argument("--MMA_DEPTH", type=int, default=2)
    args = parser.parse_args()

    def run(bench):
        test_matmul(
            args.M,
            args.N,
            args.K,
            args.prec,
            args.BLOCK_M,
            args.BLOCK_N,
            args.BLOCK_K,
            args.GROUP_M,
            args.NUM_WARPS,
            args.NUM_STAGES,
            args.MMA_DEPTH,
            args.auto_ws,
            args.ttg_ws,
            args,
            bench,
        )

    # verify correctness
    run(False)

    # run benchmark
    utils.run_bench("matmul", args, run)
