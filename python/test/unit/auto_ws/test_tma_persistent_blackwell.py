import pytest
import torch
import argparse

import triton
import triton.language as tl
import triton.testing_autows as utils
import triton.profiler as proton
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _compute_tile_and_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    tile_id += NUM_SMS
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return tile_id, pid_m, pid_n


@triton.jit
def matmul_kernel_tma_persistent(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,  #
    M,
    N,
    K,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,
):  #
    mma_group = tl.group(name='mma', start=0, size=4)
    load_group = tl.group(name='tma_load', start=4, size=4)

    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    tile_id_c = start_pid - NUM_SMS

    ki = -1

    offs_am = 0
    offs_bn = 0

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id, pid_m, pid_n = _compute_tile_and_pid(
                tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
            )
            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

        offs_k = ki * BLOCK_SIZE_K

        with load_group:
            a = a_desc_ptr.load([offs_am, offs_k])
            b = b_desc_ptr.load([offs_bn, offs_k])

        with mma_group:
            accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            tile_id_c, pid_m, pid_n = _compute_tile_and_pid(
                tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
            )

            offs_am_c = pid_m * BLOCK_SIZE_M
            offs_bn_c = pid_n * BLOCK_SIZE_N

            # Epilogue subtiling is a technique to break our computation and stores into multiple pieces
            # By subtiling we can reduce shared memory consumption by the epilogue and instead use that
            # memory to increase our stage count.
            # In this case we partition the accumulator into 2 BLOCK_SIZE_M x BLOCK_SIZE_N // 2 tensors
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(dtype)
                with mma_group:
                    c_desc_ptr.store([offs_am_c, offs_bn_c], c0)
                c1 = acc1.to(dtype)
                with mma_group:
                    c_desc_ptr.store([offs_am_c, offs_bn_c + BLOCK_SIZE_N // 2], c1)
            else:
                # accumulator = accumulator * 0
                # accumulator = accumulator + 2
                accumulator = accumulator.to(dtype)
                with mma_group:
                    c_desc_ptr.store([offs_am_c, offs_bn_c], accumulator)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def matmul_kernel_tma_persistent_nested(
    a_desc_ptr,
    b_desc_ptr,
    c_desc_ptr,  #
    M,
    N,
    K,  #
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    FP8_OUTPUT: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_SMS: tl.constexpr,
):  #
    dtype = tl.float8e4nv if FP8_OUTPUT else tl.float16
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, tl.num_programs(0)):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        offs_k = 0

        for _ in range(0, k_tiles):
            a = a_desc_ptr.load([offs_am, offs_k])
            b = b_desc_ptr.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)
            offs_k += BLOCK_SIZE_K

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc_ptr.store([offs_am, offs_bn], c0)
            c1 = acc1.to(dtype)
            c_desc_ptr.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
        else:
            accumulator = accumulator.to(dtype)
            c_desc_ptr.store([offs_am, offs_bn], accumulator)


@pytest.mark.parametrize("M", [128, 256, 512, 1024, 2048, 4096, 8192])
@pytest.mark.parametrize("N", [256, 512, 1024, 2048, 4096, 8192])
# FIXME: Fails accuracy check with K = 8192
@pytest.mark.parametrize("K", [128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("DTYPE", ["fp8", "fp16"])
@pytest.mark.parametrize("BLOCK_SIZE_M", [128, 256])
@pytest.mark.parametrize("BLOCK_SIZE_N", [128, 256])
@pytest.mark.parametrize("BLOCK_SIZE_K", [32, 64, 128])
@pytest.mark.parametrize("GROUP_SIZE_M", [8])
@pytest.mark.parametrize("NUM_STAGES", [4, 3])
@pytest.mark.parametrize("EPILOGUE_SUBTILE", [True, False])
@pytest.mark.parametrize(
    ("ENABLE_WARP_SPECIALIZATION", "USE_TTG_WS", "NUM_WARPS"),
    [
        (True, False, 4),
        (True, False, 8),
        # (False, 4),  FIXME: fails verification with auto-ws disabled for many configs
    ],
)
@pytest.mark.parametrize("NESTED", [True, False])
def test_matmul_tma_persistent_blackwell(
    M,
    N,
    K,
    DTYPE,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    GROUP_SIZE_M,
    NUM_STAGES,
    NUM_WARPS,
    EPILOGUE_SUBTILE,
    ENABLE_WARP_SPECIALIZATION,
    USE_TTG_WS,
    NESTED,
    IGNORE_MANUAL_GROUPS=True,
    cmd_args=None,
    bench=False,
):
    utils.common_test_setup(ENABLE_WARP_SPECIALIZATION, NUM_WARPS)
    if not utils.is_sm10x():
        pytest.skip("requires Blackwell")

    dtype = utils.torch_dtype(DTYPE)
    A = utils.generate_input((M, K), dtype)
    B = utils.generate_input((N, K), dtype)

    # Check constraints.
    assert A.shape[1] == B.shape[1], "Incompatible dimensions"  # B is transposed
    assert A.dtype == B.dtype, "Incompatible dtypes"

    C = torch.empty((M, N), device=A.device, dtype=dtype)

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    store_block_n = BLOCK_SIZE_N
    if EPILOGUE_SUBTILE:
        store_block_n = store_block_n // 2

    def grid(META):
        result = (
            min(
                NUM_SMS,
                triton.cdiv(M, META["BLOCK_SIZE_M"])
                * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            ),
        )
        return result

    desc_a = TensorDescriptor(A, A.shape, A.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_K])
    desc_b = TensorDescriptor(B, B.shape, B.stride(), [BLOCK_SIZE_N, BLOCK_SIZE_K])
    desc_c = TensorDescriptor(C, C.shape, C.stride(), [BLOCK_SIZE_M, store_block_n])

    kernel = matmul_kernel_tma_persistent_nested if NESTED else matmul_kernel_tma_persistent

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
            out = kernel[grid](
                desc_a,
                desc_b,
                desc_c,  #
                M,
                N,
                K,  #
                FP8_OUTPUT=dtype == torch.float8_e4m3fn,  #
                NUM_SMS=NUM_SMS,  #
                BLOCK_SIZE_M=BLOCK_SIZE_M,
                BLOCK_SIZE_N=BLOCK_SIZE_N,
                BLOCK_SIZE_K=BLOCK_SIZE_K,
                GROUP_SIZE_M=GROUP_SIZE_M,
                num_stages=NUM_STAGES,
                num_warps=NUM_WARPS,
                EPILOGUE_SUBTILE=EPILOGUE_SUBTILE,
                enable_warp_specialization=ENABLE_WARP_SPECIALIZATION,
                use_ttg_ws=USE_TTG_WS,
                ignore_manual_groups=IGNORE_MANUAL_GROUPS,
            )
            # print(out.asm["ttgir"], flush=True)

    if bench:
        utils.bench_fn(cmd_args, run)
        return

    num_subtiles = 2 if EPILOGUE_SUBTILE else 1
    if ENABLE_WARP_SPECIALIZATION:
        smem_buffers = [
            NUM_STAGES * BLOCK_SIZE_M * BLOCK_SIZE_K * utils.dtype_size(dtype),
            NUM_STAGES * BLOCK_SIZE_N * BLOCK_SIZE_K * utils.dtype_size(dtype),
            BLOCK_SIZE_M * BLOCK_SIZE_N * utils.dtype_size(dtype) // num_subtiles,
            NUM_STAGES * 8,
            NUM_STAGES * 8,
        ]
        if NESTED:
            # Epilogue group with double buffered TMEM
            if BLOCK_SIZE_M * BLOCK_SIZE_N > 128 * 256:
                num_tmem_buf = 1
            else:
                num_tmem_buf = 2
            smem_buffers.append(num_tmem_buf * 8) # TMEM full
            smem_buffers.append(num_tmem_buf * 8) # TMEM empty
        else:
            # No epilogue group, but still two "completion" barriers
            smem_buffers.append(8)
            smem_buffers.append(8)
    else:
        smem_buffers = [
            NUM_STAGES * BLOCK_SIZE_M * BLOCK_SIZE_K * utils.dtype_size(dtype),
            NUM_STAGES * BLOCK_SIZE_N * BLOCK_SIZE_K * utils.dtype_size(dtype),
            BLOCK_SIZE_M * BLOCK_SIZE_N * utils.dtype_size(dtype) // num_subtiles,
            NUM_STAGES * 8,
            2 * 8,
        ]
    shared_memory = utils.compute_shared_memory(smem_buffers)

    utils.init_check_shared_memory_hook(matmul_kernel_tma_persistent, shared_memory)

    try:
        run()
    except triton.runtime.errors.OutOfResources:
        assert shared_memory > utils.get_shared_memory()
        return
    finally:
        if not USE_TTG_WS:
            utils.clear_check_shared_memory_hook()
    assert shared_memory <= utils.get_shared_memory()

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
    parser.add_argument("--GROUP_SIZE_M", type=int, default=8)
    parser.add_argument("--EPILOGUE_SUBTILE", action="store_true")
    parser.add_argument("--NESTED", action="store_true")
    args = parser.parse_args()

    def run(bench):
        test_matmul_tma_persistent_blackwell(
            args.M,
            args.N,
            args.K,
            args.prec,
            args.BLOCK_M,
            args.BLOCK_N,
            args.BLOCK_K,
            args.GROUP_SIZE_M,
            args.NUM_STAGES,
            args.NUM_WARPS,
            args.EPILOGUE_SUBTILE,
            args.auto_ws,
            args.ttg_ws,
            args.NESTED,
            not args.manual_groups,
            args,
            bench,
        )

    # verify correctness
    run(False)

    # run benchmark
    utils.run_bench("matmul", args, run)
