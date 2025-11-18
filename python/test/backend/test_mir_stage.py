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


def verify_mir_content(mir_content, kernel_name):
    # Verify basic MIR format
    assert len(mir_content) > 0, f"MIR for {kernel_name} should not be empty"
    assert mir_content.strip().startswith("---"), f"MIR for {kernel_name} should start with YAML document marker"
    assert "name:" in mir_content, f"MIR for {kernel_name} should contain function names"
    assert "body:" in mir_content, f"MIR for {kernel_name} should contain machine basic blocks"

    # Verify presence of Scheduling Units (SU)
    import re
    su_pattern = r'SU\(\d+\):'
    su_matches = re.findall(su_pattern, mir_content)
    assert len(su_matches) > 0, \
        f"Scheduling DAG for {kernel_name} should contain Scheduling Units (SU)"

    # Verify scheduling DAG structure with specific patterns
    assert "# preds left" in mir_content, \
        f"Scheduling DAG for {kernel_name} should contain predecessor info"
    assert "# succs left" in mir_content, \
        f"Scheduling DAG for {kernel_name} should contain successor info"

    # Verify no sched DAG from post-RA scheduler
    assert "renamable" not in mir_content, \
        f"Scheduling DAG for {kernel_name} should not contain entries from post-RA scheduler"


def test_mir_dump(tmp_path, monkeypatch):
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

    @triton.jit
    def mul_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x * y
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

    # Run mul kernel
    output_mul = torch.empty_like(x)
    mul_kernel[grid](x, y, output_mul, size, BLOCK_SIZE=128)

    # Verify mul kernel executed correctly
    expected_mul = x * y
    torch.testing.assert_close(output_mul, expected_mul)

    # Check that both kernels generated separate MIR files
    add_mir_files = list(tmp_path.glob("add_kernel_*.txt"))
    mul_mir_files = list(tmp_path.glob("mul_kernel_*.txt"))

    assert len(add_mir_files) == 1, "Exactly one MIR file should exist for add_kernel"
    assert len(mul_mir_files) == 1, "Exactly one MIR file should exist for mul_kernel"

    add_mir_path = add_mir_files[0]
    mul_mir_path = mul_mir_files[0]

    # Verify add_kernel MIR content
    add_mir_content = add_mir_path.read_text()
    verify_mir_content(add_mir_content, "add_kernel")

    # Verify mul_kernel MIR content
    mul_mir_content = mul_mir_path.read_text()
    verify_mir_content(mul_mir_content, "mul_kernel")
