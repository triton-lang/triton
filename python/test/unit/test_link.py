import pytest
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from triton._C.libtriton import llvm


@triton.jit(noinline=True)
def add_one(x_ptr, SQRT: tl.constexpr) -> None:
    x = tl.load(x_ptr)
    if SQRT:
        x = libdevice.sqrt(x)
    tl.store(x_ptr, x + 1.0)


@triton.jit
def add_one_indirect(x_ptr, SQRT: tl.constexpr) -> None:
    add_one(x_ptr, SQRT)


@pytest.mark.parametrize("use_libdevice", (False, True))
@pytest.mark.parametrize("kernel", (add_one, add_one_indirect))
def test_link_extern_libs(use_libdevice, kernel, monkeypatch):
    link_called: bool = False

    link_extern_libs = llvm.link_extern_libs

    def link_extern_libs_wrapper(*args, **kwargs):
        nonlocal link_called
        link_called = True
        return link_extern_libs(*args, **kwargs)

    monkeypatch.setattr(llvm, "link_extern_libs", link_extern_libs_wrapper)
    x = torch.ones((1, ), device="cuda")
    with (compilation := triton.knobs.compilation).scope():
        compilation.always_compile = True
        kernel[(1, )](x, SQRT=use_libdevice)

    assert (link_called == use_libdevice)
