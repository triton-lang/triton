"""
Multi-CTA Matrix Multiplication
===============================

This tutorial shows how to launch a standard Triton matmul with multiple CTAs
per program. On Blackwell, Triton can automatically lower eligible
``tl.dot`` operations to MMAv5 ``cta_group::2`` instructions when the RHS
operand can be shared across the two CTAs.

You will learn:

* How to request a two-CTA launch from the Triton frontend.
* How to inspect TTGIR/PTX to confirm that MMAv5 two-CTA MMA was selected.
* How to compare Triton single-CTA, multi-CTA, and persistent WS variants.
"""

# %%
# Setup
# -----

import argparse

import torch

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def is_blackwell():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "cuda" and torch.cuda.get_device_capability()[0] == 10


def is_cuda():
    target = triton.runtime.driver.active.get_current_target()
    return target is not None and target.backend == "cuda"


if torch.cuda.is_available():
    from triton._C.libtriton import nvidia

    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None

TORCH_HAS_FP8 = hasattr(torch, "float8_e4m3fn")

# %%
# Kernel
# ------
#
# This is intentionally close to the basic matmul tutorial. The only launch-time
# difference is that the wrapper below passes ``num_ctas=2``. The compiler is
# responsible for deciding whether the dot can use MMAv5 two-CTA mode.


def proton_autotune_do_bench(kernel_call, quantiles):
    return triton.testing.do_bench_proton(kernel_call, warmup=2, rep=20, quantiles=quantiles)


def matmul_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_M"]
    block_n = nargs["BLOCK_N"]
    block_k = nargs["BLOCK_K"]
    nargs["a_desc"].block_shape = [block_m, block_k]
    nargs["b_desc"].block_shape = [block_k, block_n]
    epilogue_subtile = nargs.get("EPILOGUE_SUBTILE", 1)
    if epilogue_subtile > 1:
        nargs["c_desc"].block_shape = [block_m, block_n // epilogue_subtile]
    else:
        nargs["c_desc"].block_shape = [block_m, block_n]


def make_matmul_configs(configs, num_ctas):
    return [
        triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GRID_MINOR_DIM": grid_minor_dim,
                "GRID_TILE_WIDTH": grid_tile_width,
                "NUM_STAGES": num_stages,
            },
            num_stages=num_stages,
            num_warps=num_warps,
            num_ctas=num_ctas,
            pre_hook=matmul_set_block_size_hook,
        ) for block_m, block_n, block_k, grid_minor_dim, grid_tile_width, num_stages, num_warps in configs
    ]


MATMUL_CONFIGS = make_matmul_configs(
    [
        (128, 128, 64, 0, 8, 3, 4),
        (128, 256, 64, 0, 8, 4, 4),
        (256, 128, 64, 0, 8, 3, 4),
    ],
    num_ctas=1,
)

TWO_CTA_CONFIGS = make_matmul_configs(
    [
        (256, 128, 64, 0, 8, 3, 4),
        (256, 128, 64, 0, 8, 4, 4),
        (256, 128, 128, 0, 8, 4, 4),
        (256, 256, 64, 1, 8, 2, 4),
        (256, 256, 64, 1, 8, 3, 4),
        (256, 256, 64, 1, 8, 4, 4),
        (256, 256, 128, 1, 8, 2, 4),
        (256, 256, 128, 1, 8, 3, 4),
        (512, 256, 64, 1, 8, 3, 4),
    ],
    num_ctas=2,
)

FOUR_CTA_CONFIGS = make_matmul_configs(
    [
        (256, 128, 64, 0, 8, 4, 4),
        (256, 256, 64, 1, 8, 3, 4),
        (256, 256, 64, 1, 8, 4, 4),
    ],
    num_ctas=4,
)

PERSISTENT_TWO_CTA_CONFIGS = make_matmul_configs(
    [
        (256, 128, 64, 0, 8, 3, 4),
        (256, 128, 64, 0, 8, 4, 4),
        (256, 256, 64, 1, 8, 3, 4),
        (256, 256, 64, 1, 8, 4, 8),
        (256, 256, 64, 0, 16, 5, 4),
    ],
    num_ctas=2,
)

PERSISTENT_FOUR_CTA_CONFIGS = make_matmul_configs(
    [
        (256, 128, 64, 0, 8, 4, 4),
        (256, 256, 64, 1, 8, 3, 4),
        (256, 256, 64, 1, 8, 4, 4),
    ],
    num_ctas=4,
)

WS_CONFIGS = make_matmul_configs(
    [
        (256, 128, 64, 0, 8, 3, 4),
        (256, 128, 64, 0, 8, 4, 4),
        (512, 128, 64, 0, 8, 3, 4),
    ],
    num_ctas=2,
)


@triton.jit
def _planar_snake(tile_id, num_pid_m, num_pid_n, GRID_MINOR_DIM: tl.constexpr, GRID_TILE_WIDTH: tl.constexpr):
    major_size = num_pid_n if GRID_MINOR_DIM == 0 else num_pid_m
    minor_size = num_pid_m if GRID_MINOR_DIM == 0 else num_pid_n

    full_minor_tiles = minor_size // GRID_TILE_WIDTH
    full_minor_size = full_minor_tiles * GRID_TILE_WIDTH
    full_elements = full_minor_tiles * GRID_TILE_WIDTH * major_size

    minor_tile_idx = tile_id // (GRID_TILE_WIDTH * major_size)
    full_minor_within = tile_id % GRID_TILE_WIDTH
    full_major_within = (tile_id // GRID_TILE_WIDTH) % major_size
    full_minor = minor_tile_idx * GRID_TILE_WIDTH + full_minor_within
    full_major = tl.where((minor_tile_idx % 2) == 0, full_major_within, major_size - 1 - full_major_within)

    partial_width = minor_size - full_minor_size
    partial_width = tl.where(partial_width > 0, partial_width, 1)
    partial_tile_id = tile_id - full_elements
    partial_minor_within = partial_tile_id % partial_width
    partial_major_within = (partial_tile_id // partial_width) % major_size
    partial_minor = minor_tile_idx * GRID_TILE_WIDTH + partial_minor_within
    partial_major = tl.where((minor_tile_idx % 2) == 0, partial_major_within, major_size - 1 - partial_major_within)

    in_full_tile = tile_id < full_elements
    minor = tl.where(in_full_tile, full_minor, partial_minor)
    major = tl.where(in_full_tile, full_major, partial_major)
    if GRID_MINOR_DIM == 0:
        return minor, major
    return major, minor


@triton.jit
def _store_epilogue(c_desc, off_m, off_n, acc, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                    EPILOGUE_SUBTILE: tl.constexpr):
    if EPILOGUE_SUBTILE == 2:
        acc = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        c_desc.store([off_m, off_n], acc0.to(tl.float16))
        c_desc.store([off_m, off_n + BLOCK_N // 2], acc1.to(tl.float16))
    elif EPILOGUE_SUBTILE == 4:
        acc = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        acc0 = tl.reshape(acc0, (BLOCK_M, 2, BLOCK_N // 4))
        acc0 = tl.permute(acc0, (0, 2, 1))
        acc00, acc01 = tl.split(acc0)
        acc1 = tl.reshape(acc1, (BLOCK_M, 2, BLOCK_N // 4))
        acc1 = tl.permute(acc1, (0, 2, 1))
        acc10, acc11 = tl.split(acc1)
        c_desc.store([off_m, off_n], acc00.to(tl.float16))
        c_desc.store([off_m, off_n + BLOCK_N // 4], acc01.to(tl.float16))
        c_desc.store([off_m, off_n + BLOCK_N // 2], acc10.to(tl.float16))
        c_desc.store([off_m, off_n + BLOCK_N * 3 // 4], acc11.to(tl.float16))
    elif EPILOGUE_SUBTILE == 8:
        acc = tl.reshape(acc, (BLOCK_M, 2, BLOCK_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        acc0 = tl.reshape(acc0, (BLOCK_M, 2, BLOCK_N // 4))
        acc0 = tl.permute(acc0, (0, 2, 1))
        acc00, acc01 = tl.split(acc0)
        acc1 = tl.reshape(acc1, (BLOCK_M, 2, BLOCK_N // 4))
        acc1 = tl.permute(acc1, (0, 2, 1))
        acc10, acc11 = tl.split(acc1)
        acc00 = tl.reshape(acc00, (BLOCK_M, 2, BLOCK_N // 8))
        acc00 = tl.permute(acc00, (0, 2, 1))
        acc000, acc001 = tl.split(acc00)
        acc01 = tl.reshape(acc01, (BLOCK_M, 2, BLOCK_N // 8))
        acc01 = tl.permute(acc01, (0, 2, 1))
        acc010, acc011 = tl.split(acc01)
        acc10 = tl.reshape(acc10, (BLOCK_M, 2, BLOCK_N // 8))
        acc10 = tl.permute(acc10, (0, 2, 1))
        acc100, acc101 = tl.split(acc10)
        acc11 = tl.reshape(acc11, (BLOCK_M, 2, BLOCK_N // 8))
        acc11 = tl.permute(acc11, (0, 2, 1))
        acc110, acc111 = tl.split(acc11)
        c_desc.store([off_m, off_n], acc000.to(tl.float16))
        c_desc.store([off_m, off_n + BLOCK_N // 8], acc001.to(tl.float16))
        c_desc.store([off_m, off_n + BLOCK_N // 4], acc010.to(tl.float16))
        c_desc.store([off_m, off_n + BLOCK_N * 3 // 8], acc011.to(tl.float16))
        c_desc.store([off_m, off_n + BLOCK_N // 2], acc100.to(tl.float16))
        c_desc.store([off_m, off_n + BLOCK_N * 5 // 8], acc101.to(tl.float16))
        c_desc.store([off_m, off_n + BLOCK_N * 3 // 4], acc110.to(tl.float16))
        c_desc.store([off_m, off_n + BLOCK_N * 7 // 8], acc111.to(tl.float16))
    else:
        c_desc.store([off_m, off_n], acc.to(tl.float16))


@triton.jit
def matmul_kernel(a_desc, b_desc, c_desc,  #
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                  GRID_MINOR_DIM: tl.constexpr, GRID_TILE_WIDTH: tl.constexpr, NUM_STAGES: tl.constexpr,
                  FP8_INPUTS: tl.constexpr, WARP_SPECIALIZE: tl.constexpr, EPILOGUE_SUBTILE: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m, pid_n = _planar_snake(pid, num_pid_m, num_pid_n, GRID_MINOR_DIM, GRID_TILE_WIDTH)

    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
        off_k = k * BLOCK_K
        a = a_desc.load([off_m, off_k])
        b = b_desc.load([off_k, off_n])
        acc = tl.dot(a, b, acc)
    _store_epilogue(c_desc, off_m, off_n, acc, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE)


@triton.jit
def matmul_kernel_persistent(a_desc, b_desc, c_desc,  #
                             M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,  #
                             GRID_MINOR_DIM: tl.constexpr, GRID_TILE_WIDTH: tl.constexpr, NUM_STAGES: tl.constexpr,
                             FP8_INPUTS: tl.constexpr, WARP_SPECIALIZE: tl.constexpr, EPILOGUE_SUBTILE: tl.constexpr,
                             NUM_SMS: tl.constexpr):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_tiles = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BLOCK_K)

    tile_id_c = start_pid - NUM_SMS
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _planar_snake(tile_id, num_pid_m, num_pid_n, GRID_MINOR_DIM, GRID_TILE_WIDTH)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in tl.range(0, k_tiles, num_stages=NUM_STAGES, warp_specialize=WARP_SPECIALIZE):
            off_k = k * BLOCK_K
            a = a_desc.load([off_m, off_k])
            b = b_desc.load([off_k, off_n])
            acc = tl.dot(a, b, acc)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _planar_snake(tile_id_c, num_pid_m, num_pid_n, GRID_MINOR_DIM, GRID_TILE_WIDTH)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        _store_epilogue(c_desc, off_m, off_n, acc, BLOCK_M, BLOCK_N, EPILOGUE_SUBTILE)


AUTOTUNE_KEY = ["M", "N", "K", "FP8_INPUTS", "WARP_SPECIALIZE"]

matmul_kernel_1cta = triton.autotune(configs=MATMUL_CONFIGS, key=AUTOTUNE_KEY,
                                     do_bench=proton_autotune_do_bench)(matmul_kernel)
matmul_kernel_2cta = triton.autotune(configs=TWO_CTA_CONFIGS, key=AUTOTUNE_KEY,
                                     do_bench=proton_autotune_do_bench)(matmul_kernel)
matmul_kernel_4cta = triton.autotune(configs=FOUR_CTA_CONFIGS, key=AUTOTUNE_KEY,
                                     do_bench=proton_autotune_do_bench)(matmul_kernel)
matmul_kernel_2cta_ws = triton.autotune(configs=WS_CONFIGS, key=AUTOTUNE_KEY,
                                        do_bench=proton_autotune_do_bench)(matmul_kernel)
matmul_kernel_persistent_2cta = triton.autotune(configs=PERSISTENT_TWO_CTA_CONFIGS, key=AUTOTUNE_KEY,
                                                do_bench=proton_autotune_do_bench)(matmul_kernel_persistent)
matmul_kernel_persistent_4cta = triton.autotune(configs=PERSISTENT_FOUR_CTA_CONFIGS, key=AUTOTUNE_KEY,
                                                do_bench=proton_autotune_do_bench)(matmul_kernel_persistent)


def matmul(a, b, *, num_ctas, warp_specialize=False, out=None):
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.dtype == b.dtype, "matrix A and B must have the same dtype"
    supported_dtypes = [torch.float16]
    if TORCH_HAS_FP8:
        supported_dtypes.append(torch.float8_e4m3fn)
    assert a.dtype in supported_dtypes, "this tutorial uses fp16 or fp8 inputs"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16) if out is None else out
    a_desc = TensorDescriptor.from_tensor(a, [1, 1])
    b_desc = TensorDescriptor.from_tensor(b, [1, 1])
    c_desc = TensorDescriptor.from_tensor(c, [1, 1])
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
    if warp_specialize:
        kernel = matmul_kernel_2cta_ws
    elif num_ctas == 4:
        kernel = matmul_kernel_4cta
    elif num_ctas == 2:
        kernel = matmul_kernel_2cta
    else:
        kernel = matmul_kernel_1cta
    return kernel[grid](a_desc, b_desc, c_desc, M, N, K, FP8_INPUTS=(TORCH_HAS_FP8 and a.dtype == torch.float8_e4m3fn),
                        WARP_SPECIALIZE=warp_specialize, EPILOGUE_SUBTILE=1)


def matmul_persistent(a, b, *, num_ctas, out=None):
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    assert a.dtype == b.dtype, "matrix A and B must have the same dtype"
    assert a.is_contiguous(), "matrix A must be contiguous"
    assert b.is_contiguous(), "matrix B must be contiguous"

    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16) if out is None else out
    a_desc = TensorDescriptor.from_tensor(a, [1, 1])
    b_desc = TensorDescriptor.from_tensor(b, [1, 1])
    c_desc = TensorDescriptor.from_tensor(c, [1, 1])
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    grid = lambda META: (min(num_sms, triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"])), )
    kernel = matmul_kernel_persistent_4cta if num_ctas == 4 else matmul_kernel_persistent_2cta
    warp_specialize = num_ctas in {2, 4}
    epilogue_subtile = 8 if num_ctas == 2 else 1
    return kernel[grid](a_desc, b_desc, c_desc, M, N, K, FP8_INPUTS=(TORCH_HAS_FP8 and a.dtype == torch.float8_e4m3fn),
                        WARP_SPECIALIZE=warp_specialize, EPILOGUE_SUBTILE=epilogue_subtile, NUM_SMS=num_sms)


def tflops(ms, M, N, K):
    return 2.0 * M * N * K * 1e-12 / (ms * 1e-3)


def bench(fn):
    return triton.testing.do_bench_proton(fn, warmup=2, rep=20, return_mode="median")


BENCHMARK_SHAPES = list(range(1024, 4097, 512))


def fmt_config(config):
    return (f"{config.kwargs['BLOCK_M']}x{config.kwargs['BLOCK_N']}x{config.kwargs['BLOCK_K']}"
            f"g{config.kwargs['GRID_MINOR_DIM']}x{config.kwargs['GRID_TILE_WIDTH']}"
            f"s{config.num_stages}w{config.num_warps}")


# %%
# Correctness and IR Inspection
# -----------------------------


def validate_and_inspect(skip_ws=False):
    if not is_blackwell():
        raise RuntimeError("This tutorial requires an NVIDIA Blackwell GPU.")

    M, N, K = 1024, 1024, 1024
    torch.manual_seed(0)
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    out = torch.empty((M, N), device=DEVICE, dtype=torch.float16)

    ref = torch.matmul(a, b)
    compiled_2cta = matmul(a, b, num_ctas=2, out=out)
    torch.testing.assert_close(ref.to(torch.float32), out.to(torch.float32), atol=0.06, rtol=0.06)

    compiled_4cta = matmul(a, b, num_ctas=4, out=out)
    torch.testing.assert_close(ref.to(torch.float32), out.to(torch.float32), atol=0.06, rtol=0.06)

    matmul_persistent(a, b, num_ctas=2, out=out)
    torch.testing.assert_close(ref.to(torch.float32), out.to(torch.float32), atol=0.06, rtol=0.06)

    matmul_persistent(a, b, num_ctas=4, out=out)
    torch.testing.assert_close(ref.to(torch.float32), out.to(torch.float32), atol=0.06, rtol=0.06)

    a_desc = TensorDescriptor.from_tensor(a, [256, 64])
    b_desc = TensorDescriptor.from_tensor(b, [64, 128])
    c_desc = TensorDescriptor.from_tensor(out, [256, 128])
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    persistent_2cta_grid = (min(num_sms, triton.cdiv(M, 256) * triton.cdiv(N, 128)), )
    compiled_persistent_2cta_forced = matmul_kernel_persistent[persistent_2cta_grid](
        a_desc, b_desc, c_desc, M, N, K, BLOCK_M=256, BLOCK_N=128, BLOCK_K=64, GRID_MINOR_DIM=0, GRID_TILE_WIDTH=8,
        NUM_STAGES=3, FP8_INPUTS=False, WARP_SPECIALIZE=False, EPILOGUE_SUBTILE=1, NUM_SMS=num_sms, num_warps=4,
        num_stages=3, num_ctas=2)

    a_desc = TensorDescriptor.from_tensor(a, [256, 64])
    b_desc = TensorDescriptor.from_tensor(b, [64, 256])
    c_desc = TensorDescriptor.from_tensor(out, [256, 256])
    persistent_4cta_grid = (min(num_sms, triton.cdiv(M, 256) * triton.cdiv(N, 256)), )
    compiled_persistent_4cta_forced = matmul_kernel_persistent[persistent_4cta_grid](
        a_desc, b_desc, c_desc, M, N, K, BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, GRID_MINOR_DIM=1, GRID_TILE_WIDTH=8,
        NUM_STAGES=3, FP8_INPUTS=False, WARP_SPECIALIZE=True, EPILOGUE_SUBTILE=1, NUM_SMS=num_sms, num_warps=4,
        num_stages=3, num_ctas=4)

    ttgir = compiled_2cta.asm["ttgir"]
    ptx = compiled_2cta.asm["ptx"]
    four_cta_ttgir = compiled_4cta.asm["ttgir"]
    four_cta_ptx = compiled_4cta.asm["ptx"]
    persistent_2cta_ttgir = compiled_persistent_2cta_forced.asm["ttgir"]
    persistent_2cta_ptx = compiled_persistent_2cta_forced.asm["ptx"]
    persistent_4cta_ttgir = compiled_persistent_4cta_forced.asm["ttgir"]
    persistent_4cta_ptx = compiled_persistent_4cta_forced.asm["ptx"]
    print(f"TTGIR contains two_ctas: {'two_ctas' in ttgir}", flush=True)
    print(f"4CTA TTGIR contains two_ctas: {'two_ctas' in four_cta_ttgir}", flush=True)
    print(
        f"4CTA TTGIR contains TMA multicast: {'async_tma_copy_global_to_local' in four_cta_ttgir and 'multicast' in four_cta_ttgir}",
        flush=True)
    print(f"PTX contains cta_group::2: {'cta_group::2' in ptx}", flush=True)
    print(
        f"4CTA PTX contains TMA multicast: {'cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster' in four_cta_ptx}",
        flush=True)
    if not skip_ws:
        compiled_ws = matmul(a, b, num_ctas=2, warp_specialize=True, out=out)
        torch.testing.assert_close(ref.to(torch.float32), out.to(torch.float32), atol=0.06, rtol=0.06)
        ws_ttgir = compiled_ws.asm["ttgir"]
        ws_ptx = compiled_ws.asm["ptx"]
        print(f"WS TTGIR contains two_ctas: {'two_ctas' in ws_ttgir}", flush=True)
        print(f"WS TTGIR contains ttg.warp_specialize: {'ttg.warp_specialize' in ws_ttgir}", flush=True)
        print(f"WS PTX contains cta_group::2: {'cta_group::2' in ws_ptx}", flush=True)
    print(f"Persistent 2CTA TTGIR contains two_ctas: {'two_ctas' in persistent_2cta_ttgir}", flush=True)
    print(f"Persistent 2CTA PTX contains cta_group::2: {'cta_group::2' in persistent_2cta_ptx}", flush=True)
    print(f"Persistent 4CTA TTGIR contains two_ctas: {'two_ctas' in persistent_4cta_ttgir}", flush=True)
    print(f"Persistent 4CTA TTGIR contains ttg.warp_specialize: {'ttg.warp_specialize' in persistent_4cta_ttgir}",
          flush=True)
    print(f"Persistent 4CTA TTGIR contains TMA multicast: {'multicast' in persistent_4cta_ttgir}", flush=True)
    print(f"Persistent 4CTA PTX contains cta_group::2: {'cta_group::2' in persistent_4cta_ptx}", flush=True)


# %%
# Benchmark
# ---------
#
# Compare the Triton variants listed below with a cuBLAS reference.


def benchmark(shapes=BENCHMARK_SHAPES, precision="fp16"):
    if not is_blackwell():
        raise RuntimeError("This tutorial requires an NVIDIA Blackwell GPU.")
    if precision not in {"fp16", "fp8", "all"}:
        raise ValueError("precision must be one of: 'fp16', 'fp8', or 'all'")

    print("Benchmarking Triton multi-CTA matmul", flush=True)
    print("====================================", flush=True)
    if precision in {"fp16", "all"}:
        benchmark_precision(shapes, precision="fp16")
    if precision in {"fp8", "all"}:
        if not TORCH_HAS_FP8 or not is_cuda():
            raise RuntimeError("fp8 benchmarking requires CUDA and torch.float8_e4m3fn")
        benchmark_precision(shapes, precision="fp8")


def benchmark_precision(shapes, precision):
    fp8_inputs = precision == "fp8"
    print(f"\n{precision.upper()} square shapes", flush=True)
    print("    M=N=K       1CTA       2CTA       4CTA P2CTA+WS P4CTA+WS     cuBLAS      best shapes", flush=True)

    for size in shapes:
        M = N = K = int(size)
        device = str(DEVICE)
        a = torch.empty((M, K), device=device, dtype=torch.float16).normal_()
        b = torch.empty((K, N), device=device, dtype=torch.float16).normal_()
        if fp8_inputs:
            a = a.to(torch.float8_e4m3fn)
            b = b.to(torch.float8_e4m3fn)
        b_trans = b.T.contiguous()
        c_triton = torch.empty((M, N), device=str(DEVICE), dtype=torch.float16)

        matmul(a, b, num_ctas=1, out=c_triton)
        ms_1cta = bench(lambda: matmul(a, b, num_ctas=1, out=c_triton))
        cfg_1cta = matmul_kernel_1cta.best_config

        compiled_2cta = matmul(a, b, num_ctas=2, out=c_triton)
        if "two_ctas" not in compiled_2cta.asm["ttgir"] or "cta_group::2" not in compiled_2cta.asm["ptx"]:
            raise RuntimeError("2CTA autotune selected a kernel without MMAv5 cta_group::2")
        ms_2cta = bench(lambda: matmul(a, b, num_ctas=2, out=c_triton))
        cfg_2cta = matmul_kernel_2cta.best_config

        compiled_4cta = matmul(a, b, num_ctas=4, out=c_triton)
        if "multicast" not in compiled_4cta.asm["ttgir"]:
            raise RuntimeError("4CTA autotune selected a kernel without TMA multicast")
        ms_4cta = bench(lambda: matmul(a, b, num_ctas=4, out=c_triton))
        cfg_4cta = matmul_kernel_4cta.best_config

        compiled_persistent_2cta = matmul_persistent(a, b, num_ctas=2, out=c_triton)
        ms_persistent_2cta = bench(lambda: matmul_persistent(a, b, num_ctas=2, out=c_triton))
        cfg_persistent_2cta = matmul_kernel_persistent_2cta.best_config
        persistent_2cta_kind = "mma2" if "cta_group::2" in compiled_persistent_2cta.asm["ptx"] else "no-mma2"

        compiled_persistent_4cta = matmul_persistent(a, b, num_ctas=4, out=c_triton)
        ms_persistent_4cta = bench(lambda: matmul_persistent(a, b, num_ctas=4, out=c_triton))
        cfg_persistent_4cta = matmul_kernel_persistent_4cta.best_config
        persistent_4cta_kind = ("ws+multicast" if "ttg.warp_specialize" in compiled_persistent_4cta.asm["ttgir"]
                                and "multicast" in compiled_persistent_4cta.asm["ttgir"] else "missing-ws-or-multicast")

        prefix = (f"{size:>9} {tflops(ms_1cta, M, N, K):>10.2f}"
                  f" {tflops(ms_2cta, M, N, K):>10.2f} {tflops(ms_4cta, M, N, K):>10.2f}"
                  f" {tflops(ms_persistent_2cta, M, N, K):>8.2f}"
                  f" {tflops(ms_persistent_4cta, M, N, K):>8.2f}")
        shape_text = (f"{fmt_config(cfg_1cta)} / {fmt_config(cfg_2cta)} / {fmt_config(cfg_4cta)} / "
                      f"P2 {fmt_config(cfg_persistent_2cta)} {persistent_2cta_kind} / "
                      f"P4 {fmt_config(cfg_persistent_4cta)} {persistent_4cta_kind}")
        if fp8_inputs:
            c_ref = torch.empty((M, N), device=str(DEVICE), dtype=torch.float8_e4m3fn)
        else:
            c_ref = torch.empty_like(c_triton)
        if cublas is not None:
            ms_cublas = bench(lambda: cublas.matmul(a, b_trans, c_ref))
        else:
            ms_cublas = bench(lambda: torch.matmul(a, b, out=c_ref))
        print(f"{prefix} {tflops(ms_cublas, M, N, K):>10.2f}      {shape_text}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ws", action="store_true", help="skip warp-specialized validation")
    args = parser.parse_args()
    validate_and_inspect(skip_ws=args.skip_ws)
    benchmark()
