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


def test_mir_swap_pipeline(tmp_path, monkeypatch):
    """Test MIR swap functionality using a previously dumped MIR file"""

    # First, dump a MIR file to use for swapping
    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    swap_dir = tmp_path / "swap"
    swap_dir.mkdir()

    monkeypatch.setenv("TRITON_DUMP_MIR", str(dump_dir))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    @triton.jit
    def original_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # Simple copy operation
        tl.store(output_ptr + offsets, x, mask=mask)

    # Run kernel once to generate MIR file
    size = 128
    x = torch.randn(size, device='cuda')
    output1 = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )
    original_kernel[grid](x, output1, size, BLOCK_SIZE=128)

    # Verify first execution
    torch.testing.assert_close(output1, x)

    # Find the generated MIR file
    mir_files = list(dump_dir.glob("original_kernel_*.txt"))
    assert len(mir_files) == 1, "Exactly one MIR file should have been dumped"

    original_mir_path = mir_files[0]
    mir_content = original_mir_path.read_text()
    verify_mir_content(mir_content, "original_kernel")

    # Copy MIR file to swap directory with the same name
    swap_mir_path = swap_dir / original_mir_path.name
    swap_mir_path.write_text(mir_content)

    # Now test MIR swapping
    monkeypatch.setenv("TRITON_SWAP_MIR", str(swap_dir))
    # Remove TRITON_DUMP_MIR to test pure swap functionality
    monkeypatch.delenv("TRITON_DUMP_MIR", raising=False)

    # Create a new kernel with same signature but different behavior to verify swap works
    @triton.jit
    def swap_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # Different operation: multiply by 2
        result = x * 2.0
        tl.store(output_ptr + offsets, result, mask=mask)

    # Run kernel with MIR swap - should use the swapped MIR instead of compiling swap_kernel
    output2 = torch.empty_like(x)
    swap_kernel[grid](x, output2, size, BLOCK_SIZE=128)

    # The behavior should be from the original_kernel (copy), not swap_kernel (multiply by 2)
    # This proves that MIR swap is working and bypassing the normal compilation pipeline
    torch.testing.assert_close(output2, x)

    # Verify that the result is NOT from the swap_kernel logic (which would be x * 2)
    expected_if_not_swapped = x * 2.0
    with pytest.raises(AssertionError):
        torch.testing.assert_close(output2, expected_if_not_swapped)


def test_mir_swap_with_nonexistent_file(tmp_path, monkeypatch):
    """Test MIR swap behavior when swap file doesn't exist"""

    # Set up swap directory but don't put any MIR files in it
    swap_dir = tmp_path / "empty_swap"
    swap_dir.mkdir()

    monkeypatch.setenv("TRITON_SWAP_MIR", str(swap_dir))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    @triton.jit
    def test_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x, mask=mask)

    size = 128
    x = torch.randn(size, device='cuda')
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )

    # This should either fail gracefully or fall back to normal compilation
    # The exact behavior depends on implementation but it shouldn't crash
    try:
        test_kernel[grid](x, output, size, BLOCK_SIZE=128)
        # If it succeeds, verify the kernel still works correctly
        torch.testing.assert_close(output, x)
    except (FileNotFoundError, RuntimeError) as e:
        # Expected behavior when MIR swap file doesn't exist
        assert "No such file or directory" in str(e) or "MIR" in str(e)


def test_mir_swap_disabled_when_dump_enabled(tmp_path, monkeypatch):
    """Test that MIR swap is disabled when TRITON_DUMP_MIR is also set"""

    dump_dir = tmp_path / "dump"
    swap_dir = tmp_path / "swap"
    dump_dir.mkdir()
    swap_dir.mkdir()

    # Create a dummy MIR file in swap directory
    dummy_mir = swap_dir / "test_kernel_dummy.txt"
    dummy_mir.write_text("---\nname: dummy\nbody: dummy\n")

    # Set both TRITON_DUMP_MIR and TRITON_SWAP_MIR
    monkeypatch.setenv("TRITON_DUMP_MIR", str(dump_dir))
    monkeypatch.setenv("TRITON_SWAP_MIR", str(swap_dir))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    @triton.jit
    def test_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # Multiply by 3 to distinguish from swap behavior
        result = x * 3.0
        tl.store(output_ptr + offsets, result, mask=mask)

    size = 128
    x = torch.randn(size, device='cuda')
    output = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )
    test_kernel[grid](x, output, size, BLOCK_SIZE=128)

    # Should execute normal kernel logic (multiply by 3), not swap
    expected = x * 3.0
    torch.testing.assert_close(output, expected)

    # Should have dumped a new MIR file
    mir_files = list(dump_dir.glob("test_kernel_*.txt"))
    assert len(mir_files) >= 1, "MIR file should have been dumped"
