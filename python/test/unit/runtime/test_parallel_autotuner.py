"""Tests for parallel autotuner compilation.

Verifies correctness, restore_value, and exception handling across all
code paths: sequential (workers=1), parallel with overlap disabled, and
parallel with overlap enabled.

Each test uses a unique SCALE constexpr per mode to force fresh compilations
and avoid hitting the disk cache from a previous mode.
"""

import os

import torch
import pytest

import triton
import triton.language as tl


def do_bench(kernel_call, quantiles, use_cuda_graph=False):
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)


MODES = [
    (0, 1, True),
    (1, 2, False),
    (2, 2, True),
]
MODE_IDS = ["sequential", "parallel-no-overlap", "parallel-overlap"]

CONFIGS = [
    triton.Config(kwargs={"BLOCK_SIZE": 32}),
    triton.Config(kwargs={"BLOCK_SIZE": 64}),
    triton.Config(kwargs={"BLOCK_SIZE": 128}),
    triton.Config(kwargs={"BLOCK_SIZE": 256}),
]


def _set_env(workers, overlap):
    os.environ["TRITON_AUTOTUNING_COMPILE_WORKERS"] = str(workers)
    os.environ["TRITON_AUTOTUNING_OVERLAP_BENCH"] = "1" if overlap else "0"


def _clear_env():
    os.environ.pop("TRITON_AUTOTUNING_COMPILE_WORKERS", None)
    os.environ.pop("TRITON_AUTOTUNING_OVERLAP_BENCH", None)


@pytest.mark.parametrize("mode_idx, workers, overlap", MODES, ids=MODE_IDS)
def test_correctness(mode_idx, workers, overlap, device):
    """Each code path produces correct results and selects a config."""
    N = 4096
    scale = 100 + mode_idx
    src = torch.randn(N, device=device)
    dst = torch.empty(N, device=device)

    @triton.autotune(configs=list(CONFIGS), key=["N", "SCALE"])
    @triton.jit
    def copy_kernel(src_ptr, dst_ptr, N, SCALE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offs = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask) * SCALE, mask=mask)

    _set_env(workers, overlap)
    try:
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
        copy_kernel[grid](src, dst, N, scale)
        torch.testing.assert_close(src * scale, dst)
        assert copy_kernel.best_config is not None
        assert copy_kernel.bench_time > 0
    finally:
        _clear_env()


@pytest.mark.parametrize("mode_idx, workers, overlap", MODES, ids=MODE_IDS)
def test_restore(mode_idx, workers, overlap, device):
    """restore_value correctly restores tensor state between config benchmarks."""
    N = 1024
    scale = 200 + mode_idx
    src = torch.zeros(N, device=device)

    configs = [triton.Config(kwargs={"BLOCK_SIZE": 32}), triton.Config(kwargs={"BLOCK_SIZE": 128})]

    @triton.autotune(configs=configs, key=["N", "SCALE"], restore_value=["src"], do_bench=do_bench)
    @triton.jit
    def _kernel(src, N, SCALE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + SCALE
        tl.store(src + offsets, x, mask=offsets < N)

    _set_env(workers, overlap)
    try:
        grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)
        _kernel[grid](src, N, scale)
        expected = torch.full_like(src, scale, dtype=src.dtype)
        triton.testing.assert_close(src, expected)
    finally:
        _clear_env()


@pytest.mark.parametrize("mode_idx, workers, overlap", MODES, ids=MODE_IDS)
def test_exceed_threads(mode_idx, workers, overlap, device):
    """Configs that exceed hardware limits are handled gracefully."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    scale = 300 + mode_idx
    x = torch.empty(1024, device=device, dtype=torch.float32)
    y = torch.empty_like(x)
    output = torch.empty_like(x)

    configs = [
        triton.Config({}, num_warps=128),
        triton.Config({}, num_warps=4),
    ]

    exception_out_of_resource = None

    def _post_hook(*args, exception):
        nonlocal exception_out_of_resource
        if exception is not None:
            exception_out_of_resource = exception

    @triton.autotune(configs=configs, key=["BLOCK_SIZE", "SCALE"], do_bench=do_bench, post_hook=_post_hook)
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, SCALE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        xv = tl.load(x_ptr + offsets, mask=mask)
        yv = tl.load(y_ptr + offsets, mask=mask)
        out = (xv + yv) * SCALE
        tl.store(output_ptr + offsets, out, mask=mask)

    _set_env(workers, overlap)
    try:
        def grid(meta):
            return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

        add_kernel[grid](x, y, output, x.numel(), scale, BLOCK_SIZE=128)

        warp_size = triton.runtime.driver.active.get_current_target().warp_size
        assert exception_out_of_resource is not None and f"out of resource: threads, Required: {128 * warp_size}" in str(
            exception_out_of_resource)
    finally:
        _clear_env()
