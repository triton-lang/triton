import triton
import triton.language as tl
import torch


@triton.jit
def _add1(val):
    return val + 1


@triton.jit
def _add2(val):
    return val + 2


@triton.jit
def _template(Ptr,
              # nothing
              val,  # value
              cond, fn: tl.constexpr):
    if cond:
        tl.store(Ptr, fn(val))
    else:
        tl.store(Ptr, 0)


def test_specialize():
    kernel_cond_add1 = triton.specialize(_template, {"cond": True, "fn": _add1}, name="kernel_cond_add1")
    kernel_cond_add2 = triton.specialize(_template, {"cond": True, "fn": _add2}, name="kernel_cond_add1")
    kernel_nocond_add1 = triton.specialize(_template, {"cond": False, "fn": _add1}, name="kernel_cond_add1")
    kernel_nocond_add2 = triton.specialize(_template, {"cond": False, "fn": _add2}, name="kernel_cond_add1")

    x = torch.zeros((1, ), dtype=torch.int32, device="cuda")
    kernel_cond_add1[(1, )](x, 1)
    assert x.item() == 2
    kernel_cond_add2[(1, )](x, 1)
    assert x.item() == 3
    kernel_nocond_add1[(1, )](x, 1)
    assert x.item() == 0
    kernel_nocond_add2[(1, )](x, 1)
    assert x.item() == 0
