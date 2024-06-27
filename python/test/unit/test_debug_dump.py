import triton
import triton.language as tl
import os
import torch


def test_fn_dump(capfd):
    N = 1024
    src = torch.zeros(N, device='cuda')

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )

    @triton.jit
    def _kernel(src, N, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offsets, mask=offsets < N) + 1
        tl.store(src + offsets, x, mask=offsets < N)

    os.environ['MLIR_ENABLE_DUMP'] = '1'
    BLOCK_SIZE = 16
    _kernel[grid](src, N, BLOCK_SIZE)
    captured = capfd.readouterr()
    assert "IR Dump Before" in captured.err
    assert "tt.func public @_kernel" in captured.err

    os.environ['MLIR_ENABLE_DUMP'] = '_kernel'
    BLOCK_SIZE = 32
    _kernel[grid](src, N, BLOCK_SIZE)
    captured = capfd.readouterr()
    assert "IR Dump Before" in captured.err
    assert "tt.func public @_kernel" in captured.err

    os.environ['MLIR_ENABLE_DUMP'] = '_kernel2'
    BLOCK_SIZE = 64
    _kernel[grid](src, N, BLOCK_SIZE)
    captured = capfd.readouterr()
    assert "IR Dump Before" not in captured.err

    os.environ['MLIR_ENABLE_DUMP'] = '0'
