import torch

import triton
import triton.language as tl
import pytest


def do_bench(kernel_call, quantiles, use_cuda_graph=False):
    if use_cuda_graph:
        return triton.testing.do_bench_cudagraph(kernel_call, quantiles=quantiles)
    return triton.testing.do_bench(kernel_call, quantiles=quantiles, warmup=1, rep=1)


@pytest.mark.parametrize('use_cuda_graph', [False, True])
def test_kwargs(use_cuda_graph: bool, device: str):
    if use_cuda_graph and not torch.cuda.is_available():
        pytest.xfail("CUDA is not available")

    M, N = 1024, 16
    src = torch.randn(M * N, device=device)
    dst = torch.empty(M * N, device=device)

    configs = [triton.Config(kwargs={'BLOCK_SIZE_M': 32}), triton.Config(kwargs={'BLOCK_SIZE_M': 128})]

    @triton.autotune(configs=configs, key=["M"],
                     do_bench=lambda kernel, quantiles: do_bench(kernel, quantiles, use_cuda_graph))
    @triton.jit
    def _kernel(dst, src, stride_m: tl.constexpr, M, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
        offsets_m = tl.program_id(0) * stride_m + tl.arange(0, BLOCK_SIZE_M)
        offsets_n = tl.arange(0, BLOCK_SIZE_N)
        x = tl.load(src + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :])
        tl.store(dst + offsets_m[:, None] * BLOCK_SIZE_N + offsets_n[None, :], x)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_M']), )
    _kernel[grid](dst, src, N, M, N)
    # the key word args could be in arbitrary order.
    _kernel[grid](dst=dst, src=src, M=M // 2, stride_m=N, BLOCK_SIZE_N=N)
    assert len(_kernel.cache) == 2


@pytest.mark.parametrize('pass_kwargs_to_kernel', [False, True])
def test_restore(pass_kwargs_to_kernel, device):
    N = 1024
    src = torch.zeros(N, device=device)

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.autotune(configs=configs, key=['N'], restore_value=['src'], do_bench=do_bench)
    @triton.jit
    def _kernel(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    if pass_kwargs_to_kernel:
        _kernel[grid](src=src, N=N)
    else:
        _kernel[grid](src, N)
    triton.testing.assert_close(src, torch.ones_like(src))


def test_hooks(device):
    # Autotuner's pre- and post- hooks should be called the same number of times
    N = 4096
    src = torch.zeros(N, device=device)

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 4096}), triton.Config(kwargs={'BLOCK_SIZE': 32})]

    values = {"counter": 0, "has_exception": False}

    def _pre_hook(*args, **kwargs):
        values["counter"] += 1

    def _post_hook(*args, exception):
        values["counter"] -= 1
        if exception is not None:
            values["has_exception"] = True
        assert values["counter"] == 0

    @triton.autotune(configs=configs, key=['N'], do_bench=do_bench, pre_hook=_pre_hook, post_hook=_post_hook)
    @triton.heuristics({"N_STAGES": lambda nargs: 100 if nargs['N'] == 4096 else 4})
    @triton.jit
    def _kernel(src, N, N_STAGES: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        max_iters = tl.cdiv(N, BLOCK_SIZE)
        for _ in tl.range(max_iters, num_stages=N_STAGES):
            x = tl.load(src + offsets, mask=offsets < N)
            tl.store(src + offsets, x, mask=offsets < N)
            offsets += BLOCK_SIZE

    _kernel[(1, )](src, N)

    # On NVIDIA GPUs:
    # The tuning knob `num_stages` can be set by users.
    # This will cause out of resources when N_STAGES = 100
    # shared memory bytes = N_STAGES * BLOCK_SIZE * sizeof(float)
    # On AMD GPUs:
    # `num_stages` is a fixed value of 2, so it won't cause out of resources
    if triton.runtime.driver.active.get_current_target().backend == "cuda":
        assert values["has_exception"] is True
    else:
        assert values["has_exception"] is False


@pytest.mark.parametrize('with_perf_model', [False, True])
def test_prune_configs(with_perf_model: bool, device: str):
    N = 1024
    src = torch.randn(N, device=device)
    dst = torch.empty(N, device=device)
    records = {}

    def early_config_prune(configs, named_args, **kwargs):
        records['run_early_config_prune'] = True
        if "N" in kwargs and kwargs["N"] == 1024:
            records['capture_kwargs'] = True
        if "dst" in named_args and "src" in named_args and len(named_args) == 2:
            records['capture_named_args'] = True
        return [configs[0]]

    def perf_model(*args, **kwargs):
        records['run_perf_model'] = True
        return kwargs['BLOCK_SIZE']

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    if with_perf_model:
        prune_configs_by = {'perf_model': perf_model, 'top_k': 1}
    else:
        prune_configs_by = {'early_config_prune': early_config_prune}

    @triton.autotune(configs=configs, key=['N'], prune_configs_by=prune_configs_by, do_bench=do_bench)
    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel[grid](dst, src, N=N)
    torch.testing.assert_close(src, dst)
    if with_perf_model:
        assert len(records) == 1
        assert records['run_perf_model']
    else:
        assert len(records) == 3
        assert records['run_early_config_prune']
        assert records['capture_kwargs']
        assert records['capture_named_args']


def test_exceed_tmem(device):
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] == 10:
        pytest.skip("Test requires tensor memory.")
    N = 512
    dst = torch.empty((N, ), device=device, dtype=torch.float32)
    configs = [triton.Config(kwargs={'BLOCK_SIZE': 128}), triton.Config(kwargs={'BLOCK_SIZE': 32})]
    exception_out_of_resource = None

    def _post_hook(*args, exception):
        nonlocal exception_out_of_resource
        if exception is not None:
            exception_out_of_resource = exception

    @triton.autotune(configs=configs, key=['N'], do_bench=do_bench, pre_hook=None, post_hook=_post_hook)
    @triton.jit
    def dot_kernel(dst, BLOCK_SIZE: tl.constexpr):
        a = tl.full((BLOCK_SIZE, BLOCK_SIZE), 0.0, tl.float16)
        b = tl.full((BLOCK_SIZE, BLOCK_SIZE), 0.0, tl.float16)
        c0 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        c1 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        c2 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        c3 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        c4 = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
        for i in range(0, 100):
            c0 = tl.dot(a, b, c0)
            c1 = tl.dot(a, b, c1)
            c2 = tl.dot(a, b, c2)
            c3 = tl.dot(a, b, c3)
            c4 = tl.dot(a, b, c4)
        c = c4 + c3 + c2 + c1 + c0
        c = c.reshape([BLOCK_SIZE * BLOCK_SIZE])
        tl.store(dst + tl.arange(0, BLOCK_SIZE * BLOCK_SIZE), c)

    dot_kernel[(1, )](dst)
    assert exception_out_of_resource is not None and str(
        exception_out_of_resource
    ) == "out of resource: tensor memory, Required: 640, Hardware limit: 512. Reducing block sizes or `num_stages` may help."
