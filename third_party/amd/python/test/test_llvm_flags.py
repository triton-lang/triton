from pathlib import Path

import pytest
from triton._C.libtriton import amd
import triton

current_target = triton.runtime.driver.active.get_current_target()
TTIR_PATH = str(Path(__file__).parent / "attn_fwd.ttir")


# -- Unit tests for the C++ bindings (no GPU needed) --


def test_set_and_restore_bool_flag():
    """set_llvm_flags handles bare boolean flags and restore resets them."""
    modified = amd.set_llvm_flags(["amdgpu-early-inline-all"])
    assert modified == ["amdgpu-early-inline-all"]
    amd.restore_llvm_flags(modified)


def test_set_and_restore_value_flag():
    """set_llvm_flags handles key=value flags and restore resets them."""
    modified = amd.set_llvm_flags(["inline-threshold=500"])
    assert modified == ["inline-threshold"]
    amd.restore_llvm_flags(modified)


def test_unknown_flag_skipped():
    """Unknown flags are skipped and not included in the modified list."""
    modified = amd.set_llvm_flags(["this-flag-does-not-exist=42"])
    assert modified == []


def test_multiple_flags():
    """Multiple flags can be set and restored in one call."""
    flags = ["amdgpu-early-inline-all", "inline-threshold=500"]
    modified = amd.set_llvm_flags(flags)
    assert len(modified) == 2
    amd.restore_llvm_flags(modified)


def test_restore_empty_list():
    """Restoring an empty list is a no-op."""
    amd.restore_llvm_flags([])


# -- Integration test: compile attn_fwd.ttir with llvm_flags (GPU needed) --


@pytest.mark.skipif(current_target.backend != "hip", reason="requires HIP backend")
def test_compile_accepts_llvm_flags():
    """Compilation with llvm_flags succeeds and produces valid assembly."""
    kernel = triton.compile(
        TTIR_PATH,
        target=current_target,
        options={"llvm_flags": ("enable-misched=0",)},
    )
    assert "attn_fwd" in kernel.asm["amdgcn"]


@pytest.mark.skipif(current_target.backend != "hip", reason="requires HIP backend")
def test_llvm_flags_change_codegen():
    """llvm_flags produce different assembly than the default."""
    baseline = triton.compile(TTIR_PATH, target=current_target)
    with_flags = triton.compile(
        TTIR_PATH,
        target=current_target,
        options={"llvm_flags": ("enable-misched=0",)},
    )
    assert baseline.asm["amdgcn"] != with_flags.asm["amdgcn"], \
        "expected llvm_flags to change generated assembly"
