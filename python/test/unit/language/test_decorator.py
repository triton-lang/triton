import torch

import triton
import triton.language as tl
import pytest


def test_decorator_with_def(device):

    def triton_heuristics_pointwise(**kwargs):

        def decorator(func):
            return func

        return decorator

    # "def" might appear in a decorator call, e.g. a hash string argument.
    # This test makes sure the compiler can find the right position of function
    # definition.
    @triton_heuristics_pointwise(inductor_meta={'backend_hash': 'def0aeffabe53b3f8'}, )
    @triton.jit
    def kernel():
        pass

    try:
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constexprs={}))
    except Exception as e:
        pytest.fail(f"triton compile failed with error: {e}")


def test_triton_heuristic(device):
    N = 1023
    src = torch.empty(N, device=device)
    dst = torch.zeros(N, device=device)

    do_bench = lambda kernel, quantiles: triton.testing.do_bench(kernel, quantiles=quantiles, warmup=1, rep=1)

    @triton.autotune(configs=[triton.Config(kwargs={'BLOCK_SIZE': 32})], key=['N'], do_bench=do_bench)
    @triton.heuristics({'EVEN_N': lambda nargs: nargs['N'] % 2 == 0})  # test kwargs
    @triton.heuristics({'EVEN_src': lambda nargs: nargs['src'].data_ptr() % 2 == 0})  # test args
    @triton.jit
    def _kernel(dst, src, N, BLOCK_SIZE: tl.constexpr, EVEN_N: tl.constexpr, EVEN_src: tl.constexpr):
        tl.store(dst, EVEN_N)
        tl.store(dst + 1, EVEN_src)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )
    _kernel[grid](dst, src, N=N)
    assert dst[0].item() == 0.0
    assert dst[1].item() == 1.0
    assert _kernel.base_fn.__name__ == "_kernel"
