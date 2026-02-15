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


def test_mir_dump_pipeline(tmp_path, monkeypatch):
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
    # First, dump a MIR file to use for swapping
    monkeypatch.setenv("TRITON_DUMP_MIR", str(tmp_path))
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    @triton.jit
    def copy_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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
    copy_kernel[grid](x, output1, size, BLOCK_SIZE=128)

    # Verify first execution
    torch.testing.assert_close(output1, x)

    # Find the generated MIR file
    mir_files = list(tmp_path.glob("copy_kernel_*.txt"))
    assert len(mir_files) == 1, "Exactly one MIR file should have been dumped"

    original_mir_path = mir_files[0]
    mir_content = original_mir_path.read_text()
    verify_mir_content(mir_content, "copy_kernel")

    # Now test MIR swapping
    monkeypatch.setenv("TRITON_SWAP_MIR", str(tmp_path))
    # Remove TRITON_DUMP_MIR to test pure swap functionality
    monkeypatch.delenv("TRITON_DUMP_MIR", raising=False)
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    # Run kernel with MIR swap
    output2 = torch.empty_like(x)
    copy_kernel[grid](x, output2, size, BLOCK_SIZE=128)

    torch.testing.assert_close(output2, x)


_SIMPLE_KERNEL_SCRIPT = '''
import triton
import triton.language as tl
import torch

@triton.jit
def simple_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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
simple_kernel[grid](x, output, size, BLOCK_SIZE=128)
'''


def test_mir_swap_pipeline_passes(tmp_path):
    """Test that MIR swap pipeline starts before machine-scheduler and disables schedulers."""
    import re
    import os
    import subprocess

    script_file = tmp_path / "test_kernel.py"
    script_file.write_text(_SIMPLE_KERNEL_SCRIPT)

    # Phase 1: Dump MIR
    env = os.environ.copy()
    env["TRITON_DUMP_MIR"] = str(tmp_path)
    env["TRITON_ALWAYS_COMPILE"] = "1"

    result = subprocess.run(["python", str(script_file)], capture_output=True, text=True, env=env, timeout=120)

    assert result.returncode == 0, \
        f"Dump phase should succeed. stderr: {result.stderr[:1000]}"

    # Verify MIR file was created
    mir_files = list(tmp_path.glob("simple_kernel_*.txt"))
    assert len(mir_files) == 1, "Exactly one MIR file should have been dumped"

    # Strip scheduling DAG and trailing "..." from MIR file (they break YAML parsing)
    mir_file = mir_files[0]
    mir_content = mir_file.read_text()
    dag_marker = "\n---\n=========="
    if dag_marker in mir_content:
        mir_content = mir_content.split(dag_marker)[0]
    # Remove trailing "..." which LLVM MIR parser doesn't accept
    if mir_content.rstrip().endswith("..."):
        mir_content = mir_content.rstrip()[:-3]
    mir_file.write_text(mir_content)

    # Phase 2: Swap MIR with LLVM_IR_ENABLE_DUMP to capture pass sequence
    env = os.environ.copy()
    env["TRITON_SWAP_MIR"] = str(tmp_path)
    env["TRITON_ALWAYS_COMPILE"] = "1"
    env["LLVM_IR_ENABLE_DUMP"] = "1"

    result = subprocess.run(["python", str(script_file)], capture_output=True, text=True, env=env, timeout=120)

    assert result.returncode == 0, \
        f"Swap phase should succeed. stderr: {result.stderr[:1000]}"

    all_output = result.stderr

    # Find the first "# Machine code for function" line and check the preceding IR Dump
    lines = all_output.split('\n')
    machine_code_indices = [i for i, line in enumerate(lines) if "# Machine code for function" in line]
    assert len(machine_code_indices) > 0, \
        f"Should find '# Machine code for function' in output. Stderr length: {len(all_output)}"

    first_machine_code_idx = machine_code_indices[0]

    # Find the immediately preceding "IR Dump After" line
    ir_dump_pattern = r"# \*\*\* IR Dump After (.+) \*\*\*"
    preceding_ir_dump = None
    for i in range(first_machine_code_idx - 1, -1, -1):
        match = re.search(ir_dump_pattern, lines[i])
        if match:
            preceding_ir_dump = match.group(1).strip()
            break

    assert preceding_ir_dump is not None, \
        f"Should find 'IR Dump After' before first Machine code. Lines before: {lines[max(0, first_machine_code_idx-10):first_machine_code_idx]}"

    assert "slotindexes" in preceding_ir_dump.lower() or "slot index" in preceding_ir_dump.lower(), \
        f"First MIR pass should be slotindexes, got: '{preceding_ir_dump}'"

    # Verify machine-scheduler pass does NOT modify MIR (disabled via enable-misched=false).
    # The scheduler passes still appear in the pipeline output but return early without
    # making changes when enable-misched=false is set. This is the expected LLVM behavior -
    # we verify the MIR is unchanged rather than checking for pass absence.
    dumps = re.split(r'# \*\*\* IR Dump After ([^*]+) \*\*\*', all_output)

    machine_sched_idx = None
    for i, part in enumerate(dumps):
        if 'Machine Instruction Scheduler' in part and 'PostRA' not in part:
            machine_sched_idx = i
            break

    if machine_sched_idx and machine_sched_idx >= 1 and machine_sched_idx + 1 < len(dumps):
        before_content = dumps[machine_sched_idx - 1]
        after_content = dumps[machine_sched_idx + 1]

        # Extract machine code sections
        def extract_machine_code(text):
            match = re.search(r'# Machine code for function.*', text, re.DOTALL)
            return match.group(0).strip() if match else text.strip()

        before_mc = extract_machine_code(before_content)
        after_mc = extract_machine_code(after_content)

        assert before_mc == after_mc, \
            "machine-scheduler should not modify MIR when disabled, but MIR changed"

    # Verify post-RA machine scheduler does NOT modify MIR (disabled via enable-post-misched=false).
    # Same as above - the pass appears but returns early without changes.
    post_ra_idx = None
    for i, part in enumerate(dumps):
        if 'PostRA Machine Instruction Scheduler' in part:
            post_ra_idx = i
            break

    if post_ra_idx and post_ra_idx >= 1 and post_ra_idx + 1 < len(dumps):
        before_content = dumps[post_ra_idx - 1]
        after_content = dumps[post_ra_idx + 1]

        def extract_machine_code(text):
            match = re.search(r'# Machine code for function.*', text, re.DOTALL)
            return match.group(0).strip() if match else text.strip()

        before_mc = extract_machine_code(before_content)
        after_mc = extract_machine_code(after_content)

        assert before_mc == after_mc, \
            "post-RA scheduler should not modify MIR when disabled, but MIR changed"


def _dump_and_prepare_mir(tmp_path, script_file):
    """Dump MIR for a kernel script and strip it for swapping. Returns the cleaned MIR file path."""
    import os
    import subprocess

    env = os.environ.copy()
    env["TRITON_DUMP_MIR"] = str(tmp_path)
    env["TRITON_ALWAYS_COMPILE"] = "1"

    result = subprocess.run(["python", str(script_file)], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, \
        f"Dump phase should succeed. stderr: {result.stderr[:1000]}"

    mir_files = list(tmp_path.glob("complex_kernel_*.txt"))
    assert len(mir_files) == 1, "Exactly one MIR file should have been dumped"

    mir_file = mir_files[0]
    mir_content = mir_file.read_text()
    dag_marker = "\n---\n=========="
    if dag_marker in mir_content:
        mir_content = mir_content.split(dag_marker)[0]
    if mir_content.rstrip().endswith("..."):
        mir_content = mir_content.rstrip()[:-3]
    mir_file.write_text(mir_content)
    return mir_file


def _swap_mir_and_get_output(tmp_path, script_file, enable_misched):
    """Swap MIR with LLVM_IR_ENABLE_DUMP and return stderr output."""
    import os
    import subprocess

    env = os.environ.copy()
    env["TRITON_SWAP_MIR"] = str(tmp_path)
    env["TRITON_ALWAYS_COMPILE"] = "1"
    env["LLVM_IR_ENABLE_DUMP"] = "1"
    if enable_misched:
        env["TRITON_SWAP_MIR_ENABLE_MISCHED"] = "1"

    result = subprocess.run(["python", str(script_file)], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode == 0, \
        f"Swap phase (misched={'enabled' if enable_misched else 'disabled'}) should succeed. stderr: {result.stderr[:1000]}"
    return result.stderr


def _extract_mc_around_sched(output_text):
    """Extract machine code before and after the Machine Instruction Scheduler pass."""
    import re

    dumps = re.split(r'# \*\*\* IR Dump After ([^*]+) \*\*\*', output_text)

    machine_sched_idx = None
    for i, part in enumerate(dumps):
        if 'Machine Instruction Scheduler' in part and 'PostRA' not in part:
            machine_sched_idx = i
            break

    if machine_sched_idx is None or machine_sched_idx < 1 or machine_sched_idx + 1 >= len(dumps):
        return None, None

    def extract_machine_code(text):
        match = re.search(r'# Machine code for function.*', text, re.DOTALL)
        return match.group(0).strip() if match else text.strip()

    before_mc = extract_machine_code(dumps[machine_sched_idx - 1])
    after_mc = extract_machine_code(dumps[machine_sched_idx + 1])
    return before_mc, after_mc


# Kernel script with enough independent operations for the scheduler to reorder
_COMPLEX_KERNEL_SCRIPT = '''
import triton
import triton.language as tl
import torch

@triton.jit
def complex_kernel(a_ptr, b_ptr, c_ptr, d_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Multiple independent loads
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    # Independent arithmetic chains
    ab = a * b + c
    cd = c * d + a
    bd = b + d
    ac = a - c
    # Merge results
    result = ab * cd + bd * ac
    tl.store(output_ptr + offsets, result, mask=mask)

size = 1024
a = torch.randn(size, device='cuda')
b = torch.randn(size, device='cuda')
c = torch.randn(size, device='cuda')
d = torch.randn(size, device='cuda')
output = torch.empty_like(a)
grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']), )
complex_kernel[grid](a, b, c, d, output, size, BLOCK_SIZE=256)

expected = (a * b + c) * (c * d + a) + (b + d) * (a - c)
torch.testing.assert_close(output, expected)
'''


def test_mir_swap_enable_misched(tmp_path):
    """Test that TRITON_SWAP_MIR_ENABLE_MISCHED=1 causes the machine scheduler to actually modify MIR."""
    script_file = tmp_path / "test_kernel.py"
    script_file.write_text(_COMPLEX_KERNEL_SCRIPT)

    # Phase 1: Dump and prepare MIR
    _dump_and_prepare_mir(tmp_path, script_file)

    # Phase 2: Swap with misched DISABLED (default) — scheduler should be a no-op
    disabled_output = _swap_mir_and_get_output(tmp_path, script_file, enable_misched=False)
    before_disabled, after_disabled = _extract_mc_around_sched(disabled_output)

    assert before_disabled is not None and after_disabled is not None, \
        "Should find machine code around scheduler pass (disabled case)"
    assert before_disabled == after_disabled, \
        "Scheduler should NOT modify MIR when misched is disabled"

    # Phase 3: Swap with misched ENABLED — scheduler should actually reschedule
    enabled_output = _swap_mir_and_get_output(tmp_path, script_file, enable_misched=True)
    before_enabled, after_enabled = _extract_mc_around_sched(enabled_output)

    assert before_enabled is not None and after_enabled is not None, \
        "Should find machine code around scheduler pass (enabled case)"
    assert before_enabled != after_enabled, \
        "Scheduler SHOULD modify MIR when misched is enabled"


def test_mir_swap_enable_misched_requires_swap_mir(tmp_path):
    """Test that TRITON_SWAP_MIR_ENABLE_MISCHED raises an error without TRITON_SWAP_MIR."""
    import os
    import subprocess

    script_file = tmp_path / "test_kernel.py"
    script_file.write_text(_SIMPLE_KERNEL_SCRIPT)

    env = os.environ.copy()
    env["TRITON_SWAP_MIR_ENABLE_MISCHED"] = "1"
    env["TRITON_ALWAYS_COMPILE"] = "1"
    # TRITON_SWAP_MIR is NOT set

    result = subprocess.run(["python", str(script_file)], capture_output=True, text=True, env=env, timeout=120)
    assert result.returncode != 0
    assert "TRITON_SWAP_MIR_ENABLE_MISCHED requires TRITON_SWAP_MIR" in result.stderr
