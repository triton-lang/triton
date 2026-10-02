from contextlib import contextmanager

import pytest
import torch
import triton
import triton.language as tl


@contextmanager
def enable_dump_context(pass_name="1"):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("MLIR_ENABLE_DUMP", pass_name)
        yield


def test_fn_dump(capfd, device, fresh_triton_cache):
    N = 1024
    src = torch.zeros(N, device=device)

    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]), )

    @triton.jit
    def _kernel(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    with enable_dump_context():
        BLOCK_SIZE = 16
        _kernel[grid](src, N, BLOCK_SIZE)
    captured = capfd.readouterr()
    print(captured.err)
    assert "IR Dump Before" in captured.err
    assert "tt.func public @_kernel" in captured.err

    with enable_dump_context("_kernel"):
        BLOCK_SIZE = 32
        _kernel[grid](src, N, BLOCK_SIZE)
    captured = capfd.readouterr()
    assert "IR Dump Before" in captured.err
    assert "tt.func public @_kernel" in captured.err

    with enable_dump_context("_kernel2"):
        BLOCK_SIZE = 64
        _kernel[grid](src, N, BLOCK_SIZE)
    captured = capfd.readouterr()
    assert "IR Dump Before" not in captured.err


def test_version_info(device, tmp_path, fresh_knobs):
    fresh_knobs.compilation.always_compile = True
    fresh_knobs.compilation.dump_ir = True
    fresh_knobs.cache.dump_dir = str(tmp_path)

    N = 1024
    src = torch.zeros(N, device=device)

    @triton.jit
    def _kernel(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    _kernel[(triton.cdiv(N, 16), )](src, N, 16)

    llir_files = list(tmp_path.rglob("*.llir"))
    assert len(llir_files) == 1
    llir = llir_files[0].read_text()

    # !llvm.ident = !{!0, !1}
    # !0 = !{!"Triton version <version>"}
    # !1 = !{!"LLVM version <version> (<revision>)"}
    assert "!llvm.ident = !{" in llir
    assert '!{!"Triton version ' in llir
    assert '!{!"LLVM version ' in llir
