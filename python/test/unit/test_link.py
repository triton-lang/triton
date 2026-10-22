import pytest
import subprocess
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


@pytest.mark.parametrize("constant", ["-1", "0.5", "-0.0", "1e+20", "inf", "nan"])
def test_aot_linker_constexpr_symbols(tmp_path, constant):
    from triton.tools.link import (
        HeaderParser,
        make_func_pointers,
        make_kernel_hints_dispatcher,
        make_kernel_meta_const_dispatcher,
    )

    parser = HeaderParser()
    for signature_hash, suffix, value in [("aaaa", "0d", constant), ("bbbb", "", constant), ("cccc", "", "2")]:
        parser.extract_linker_meta(f"// tt-linker: kernel_{signature_hash}_{suffix}:int32_t x:{value}_warps1xstages3")
    assert len(parser.kernels) == 2
    assert sorted(len(metas) for metas in parser.kernels.values()) == [1, 2]
    meta = next(iter(parser.kernels.values()))[0]
    dispatchers = "\n".join(make_kernel_hints_dispatcher(name, metas) for name, metas in parser.kernels.items())
    source = f"""
#include <stdint.h>
typedef void *TT_StreamTy;
typedef int TT_ResultTy;
#define TT_ERROR_INVALID_VALUE 777
int kernel_aaaa_0d(TT_StreamTy stream, int32_t x) {{ return 1; }}
int kernel_bbbb_(TT_StreamTy stream, int32_t x) {{ return 2; }}
int kernel_cccc_(TT_StreamTy stream, int32_t x) {{ return 3; }}
void load_kernel_aaaa_0d(void) {{}}
void unload_kernel_aaaa_0d(void) {{}}
void load_kernel_bbbb_(void) {{}}
void unload_kernel_bbbb_(void) {{}}
void load_kernel_cccc_(void) {{}}
void unload_kernel_cccc_(void) {{}}
{dispatchers}
{make_func_pointers(list(parser.kernels), meta)}
{make_kernel_meta_const_dispatcher(meta)}
int main(void) {{
    return kernel(0, 16, 0) != 1 || kernel(0, 3, 0) != 2 || kernel(0, 16, 1) != 3;
}}
"""
    c_path = tmp_path / "dispatch.c"
    exe_path = tmp_path / "dispatch"
    c_path.write_text(source)
    subprocess.run(["gcc", "-std=c99", str(c_path), "-o", str(exe_path)], check=True)
    subprocess.run([str(exe_path)], check=True)
