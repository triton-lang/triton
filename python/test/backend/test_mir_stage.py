import triton
import triton.language as tl

import pytest
import torch


def is_hip():
    try:
        return triton.runtime.driver.active.get_current_target().backend == "hip"
    except Exception:
        return False


# This applies to ALL tests in this file
pytestmark = pytest.mark.skipif(not is_hip(), reason="MIR tests require AMD/HIP backend")


def test_mir_dump(tmp_path, monkeypatch):
    mir_path = tmp_path / "add_kernel.txt"
    monkeypatch.setenv("TRITON_DUMP_MIR", str(tmp_path))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

    # Run kernel
    size = 128
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=128)

    # Verify kernel executed correctly
    expected = x + y
    torch.testing.assert_close(output, expected)

    # Check MIR was dumped
    assert mir_path.exists(), "MIR file should have been created"
    mir_content = mir_path.read_text()

    # Verify MIR format
    assert len(mir_content) > 0, "MIR should not be empty"
    assert mir_content.strip().startswith("---"), "MIR should start with YAML document marker"
    assert "name:" in mir_content, "MIR should contain function names"
    assert "body:" in mir_content, "MIR should contain machine basic blocks"
