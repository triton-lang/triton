import warnings
from contextlib import contextmanager

import torch

import triton


@contextmanager
def compile_warmup_only():
    """Compile intercepted Triton launches without allocating or running on the GPU."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    previous_getitem = triton.KernelInterface.__getitem__
    previous_assert_close = torch.testing.assert_close

    def getitem(kernel, grid):

        def warmup(*args, **kwargs):
            return kernel.warmup(*args, grid=grid, **kwargs)

        return warmup

    triton.KernelInterface.__getitem__ = getitem
    torch.testing.assert_close = lambda *args, **kwargs: None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Accessing the data pointer of FakeTensor.*",
            )
            with FakeTensorMode(allow_fallback_kernels=False, allow_non_fake_inputs=True):
                yield
    finally:
        torch.testing.assert_close = previous_assert_close
        triton.KernelInterface.__getitem__ = previous_getitem
