import sys
from concurrent.futures import ThreadPoolExecutor
import torch

import triton
import triton.language as tl


def test_is_lazy():
    from importlib import reload
    reload(sys.modules["triton.runtime.driver"])
    reload(sys.modules["triton.runtime"])
    mod = sys.modules[triton.runtime.driver.__module__]
    assert isinstance(triton.runtime.driver.active, getattr(mod, "LazyProxy"))
    assert triton.runtime.driver.active._obj is None
    utils = triton.runtime.driver.active.utils  # noqa: F841
    assert issubclass(triton.runtime.driver.active._obj.__class__, getattr(triton.backends.driver, "DriverBase"))


def test_kernel_in_thread(device):
    # Test calling in a new thread sets a valid device context
    buf = torch.zeros((38016 * 1024, ), dtype=torch.float32, device=device)

    @triton.jit
    def _kernel(P, BLOCK: tl.constexpr):
        pid = tl.program_id(0).to(tl.int64)
        offset = pid * BLOCK + tl.arange(0, BLOCK)

        p = tl.load(P + offset)
        tl.store(P + offset, p)

    def call_triton():
        N = buf.numel()
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]), )
        _kernel[grid](buf, BLOCK=1024)
        getattr(torch, device).synchronize()

    call_triton()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(call_triton)
        future.result()
