import os
import subprocess
import sys
from contextlib import contextmanager

import torch
import triton
import triton.language as tl


@contextmanager
def enable_dump_context(pass_name="1"):
    try:
        os.environ["MLIR_ENABLE_DUMP"] = pass_name
        yield
    finally:
        os.environ["MLIR_ENABLE_DUMP"] = "0"


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


def test_mlir_dump_path_directory(device, tmp_path, fresh_triton_cache):
    # Regression test for #6548: MLIR_DUMP_PATH can point to a directory.
    script = r"""
import torch
import triton
import triton.language as tl

N = 128
x = torch.zeros(N, device="{device}")
grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE"]),)

@triton.jit
def _kernel(ptr, n, BLOCK_SIZE: tl.constexpr):
    off = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    v = tl.load(ptr + off, mask=off < n)
    tl.store(ptr + off, v, mask=off < n)

_kernel[grid](x, N, BLOCK_SIZE=32)
""".format(device=device)

    env = os.environ.copy()
    env["MLIR_ENABLE_DUMP"] = "1"
    env["MLIR_DUMP_PATH"] = str(tmp_path)
    script_path = tmp_path / "jit_dump_script.py"
    script_path.write_text(script)
    proc = subprocess.run([sys.executable, str(script_path)], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
    assert any(p.suffix == ".mlir" for p in tmp_path.iterdir())
