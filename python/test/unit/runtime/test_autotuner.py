import torch

import triton
import triton.language as tl
import pytest


@pytest.mark.parametrize('use_cuda_graph', [False, True])
def test_kwargs(use_cuda_graph: bool):
    N = 1024
    src = torch.empty(N, device='cuda')
    dst = torch.empty(N, device='cuda')

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.autotune(configs=configs, key=['N'], warmup=1, rep=1, use_cuda_graph=use_cuda_graph)
    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N)
        tl.store(dst + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel[grid](dst, src, N)
    _kernel[grid](dst=dst, src=src, N=N)


def test_restore():
    N = 1024
    src = torch.zeros(N, device='cuda')

    configs = [triton.Config(kwargs={'BLOCK_SIZE': 32}), triton.Config(kwargs={'BLOCK_SIZE': 128})]

    @triton.autotune(configs=configs, key=['N'], restore_value=['src'], warmup=1, rep=1)
    @triton.jit
    def _kernel(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel[grid](src, N)
    triton.testing.assert_close(src, torch.ones_like(src))


@pytest.mark.parametrize('with_perf_model', [False, True])
def test_prune_configs(with_perf_model: bool):
    N = 1024
    src = torch.empty(N, device='cuda')
    dst = torch.empty(N, device='cuda')
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

    @triton.autotune(configs=configs, key=['N'], prune_configs_by=prune_configs_by, warmup=1, rep=1)
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
