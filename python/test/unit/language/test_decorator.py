import triton
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
        triton.compile(triton.compiler.ASTSource(fn=kernel, signature={}, constants={}))
    except Exception as e:
        pytest.fail(f"triton compile failed with error: {e}")
