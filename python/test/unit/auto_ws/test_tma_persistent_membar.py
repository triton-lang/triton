import pytest
import torch
import argparse

import triton
import triton.language as tl
import triton.testing_autows as utils
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def matmul_kernel_tma_persistent(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    k_tiles = tl.cdiv(K, BLOCK_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    tile0 = tile_id
    ki = -1
    kj = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kk in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N

            offs_am = tl.multiple_of(offs_am, BLOCK_M)
            offs_bn = tl.multiple_of(offs_bn, BLOCK_N)

        offs_k = ki * BLOCK_K

        a = a_desc_ptr.load([offs_am, offs_k])
        b = b_desc_ptr.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        kj = tl.where(kj == k_tiles - 1, 0, kj + 1)
        if kj == k_tiles - 1:
            tile_id0 = tile0 + (kk + 1) // k_tiles * NUM_SMS
            group_id0 = tile_id0 // num_pid_in_group
            first_pid_m0 = group_id0 * GROUP_SIZE_M
            group_size_m0 = min(num_pid_m - first_pid_m0, GROUP_SIZE_M)
            pid_m0 = first_pid_m0 + (tile_id0 % group_size_m0)
            pid_n0 = (tile_id0 % num_pid_in_group) // group_size_m0

            offs_am0 = pid_m0 * BLOCK_M
            offs_bn0 = pid_n0 * BLOCK_N

            offs_am0 = tl.multiple_of(offs_am0, BLOCK_M)
            offs_bn0 = tl.multiple_of(offs_bn0, BLOCK_N)
            c = accumulator.to(dtype)

            c_desc_ptr.store([offs_am0, offs_bn0], c)
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)


@pytest.mark.parametrize("M", [128, 256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("N", [256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("K", [128, 256, 512, 1024, 2048])
@pytest.mark.parametrize("BLOCK_M", [128])
@pytest.mark.parametrize("BLOCK_N", [128, 256])
@pytest.mark.parametrize("BLOCK_K", [64, 128])
@pytest.mark.parametrize("DTYPE", ["fp8", "fp16"])
@pytest.mark.parametrize(
    (
        "ENABLE_WARP_SPECIALIZATION",
        "NUM_WARPS",
        "NUM_STAGES",
        "MMA_DEPTH",
        "WG_SPEC",
    ),
    [
         (True, 4, 2, 2, None),
         (True, 8, 2, 2, None),
    ],
)
# Disabled, because when it is enabled test_tma_peristent_regular_store.py fails
# with incorrect smem allocation:
#  "assert expected_shared_memory == kernel.metadata.share" in testing_autows.py
#  is triggered
def test_experimental_matmul(
    M,
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    DTYPE,
    NUM_WARPS,
    NUM_STAGES,
    MMA_DEPTH,
    ENABLE_WARP_SPECIALIZATION,
    WG_SPEC,
    cmd_args=None,
    bench=False,
):
    utils.common_test_setup(ENABLE_WARP_SPECIALIZATION, NUM_WARPS)
    NUM_SMS = utils.get_num_sms()

    if (
        triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) > NUM_SMS
        and ENABLE_WARP_SPECIALIZATION
        and NUM_WARPS == 12
    ):
        pytest.skip("FIXME: test fails when #CTA <= NUM_SMS")

    # also see skipping verification in benchmarking below
    if not bench and not ENABLE_WARP_SPECIALIZATION and (M >= 4096 or N >= 4096):
        pytest.skip("FIXME: test fails verification without autows for large inputs")

    dtype = utils.torch_dtype(DTYPE)
    A = utils.generate_input((M, K), dtype)
    B = utils.generate_input((N, K), dtype)
    C = torch.empty((M, N), dtype=dtype, device="cuda")

    desc_a = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_M, BLOCK_K])
    desc_b = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_N, BLOCK_K])
    desc_c = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_M, BLOCK_N])

    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        ),
    )
    if WG_SPEC == "tma_load_first":
        WG_SPEC = (("tma_load", 0, 4), ("mma", 4, NUM_WARPS))
    elif WG_SPEC == "mma_first":
        WG_SPEC = (("tma_load", NUM_WARPS, 4), ("mma", 0, NUM_WARPS))
    else:
        WG_SPEC = ()

    # XXX: Need to clear hook, otherwise subsquent test fail with smem check
    utils.clear_check_shared_memory_hook()
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
            matmul_kernel_tma_persistent[grid](
                desc_a,
                desc_b,
                desc_c,
                M,
                N,
                K,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                GROUP_SIZE_M=8,
                FP8_OUTPUT=dtype == torch.float8_e4m3fn,
                NUM_SMS=NUM_SMS,
                num_stages=NUM_STAGES,
                num_warps=NUM_WARPS,
                mma_depth=MMA_DEPTH,
                enable_warp_specialization=ENABLE_WARP_SPECIALIZATION,
                force_membar=True,
                wg_spec_override=WG_SPEC,
            )

    if bench:
        utils.bench_fn(cmd_args, run)
        return

    if ENABLE_WARP_SPECIALIZATION:
        smem_buffers = [
            NUM_STAGES * BLOCK_M * BLOCK_K * utils.dtype_size(dtype),
            NUM_STAGES * BLOCK_N * BLOCK_K * utils.dtype_size(dtype),
            BLOCK_M * BLOCK_N * utils.dtype_size(dtype),
            NUM_STAGES * 8,
            NUM_STAGES * 8,
        ]
        if utils.is_sm10x():
            smem_buffers.append(8)
            smem_buffers.append(8)
    else:
        smem_buffers = [
            NUM_STAGES * BLOCK_M * BLOCK_K * utils.dtype_size(dtype),
            NUM_STAGES * BLOCK_N * BLOCK_K * utils.dtype_size(dtype),
            BLOCK_M * BLOCK_N * utils.dtype_size(dtype),
            NUM_STAGES * 8,
        ]
        if utils.is_sm10x():
            smem_buffers.append(MMA_DEPTH * 8)
    shared_memory = utils.compute_shared_memory(smem_buffers)

    utils.init_check_shared_memory_hook(matmul_kernel_tma_persistent, shared_memory)

    try:
        run()
    except triton.runtime.errors.OutOfResources:
        assert shared_memory > utils.get_shared_memory()
        return
    finally:
        pass
    assert shared_memory <= utils.get_shared_memory()

     if utils.is_sm10x() and (
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        DTYPE,
        NUM_WARPS,
        NUM_STAGES,
        MMA_DEPTH,
        ENABLE_WARP_SPECIALIZATION,
    ) == (128, 128, 128, "fp16", 8, 3, 2, False):
        pytest.skip("FIXME: test fails verification - precision issues?")

    utils.verify_matmul(A, B, C)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    utils.common_bench_options(parser)

    parser.add_argument("-M", type=int, default=8192)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("--BLOCK_M", type=int, default=128)
    parser.add_argument("--BLOCK_N", type=int, default=256)
    parser.add_argument("--BLOCK_K", type=int, default=64)
    parser.add_argument("--MMA_DEPTH", type=int, default=2)
    args = parser.parse_args()

    def run(bench):
        test_experimental_matmul(
            args.M,
            args.N,
            args.K,
            args.BLOCK_M,
            args.BLOCK_N,
            args.BLOCK_K,
            args.prec,
            args.NUM_WARPS,
            args.NUM_STAGES,
            args.MMA_DEPTH,
            args.auto_ws,
            args.wg_spec,
            args,
            bench,
        )

    # verify correctness
    # Disabled for some configs due to skipping verification for large inputs. See TODO above
    if not (not args.auto_ws and (args.M >= 4096 or args.N >= 4096)):
        run(False)
    else:
        print("WARNING: skipping verification step")

    # run benchmark
    utils.run_bench("matmul", args, run)
