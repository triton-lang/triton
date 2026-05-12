from pathlib import Path

import pytest
from triton._C.libtriton import amd
import triton

current_target = triton.runtime.driver.active.get_current_target()
TTIR_PATH = str(Path(__file__).parent / "attn_fwd.ttir")

# -- Unit tests for the C++ bindings (no GPU needed) --


def test_set_bool_option():
    """set_llvm_options handles bare boolean flags."""
    amd.set_llvm_options(["amdgpu-early-inline-all"])


def test_set_value_option():
    """set_llvm_options handles key=value options."""
    amd.set_llvm_options(["inline-threshold=500"])


def test_unknown_option_raises():
    """Unknown options raise std::invalid_argument."""
    with pytest.raises(ValueError):
        amd.set_llvm_options(["this-flag-does-not-exist=42"])


def test_multiple_options():
    """Multiple options can be set in one call."""
    amd.set_llvm_options(["amdgpu-early-inline-all", "inline-threshold=500"])


# -- Integration test: compile attn_fwd.ttir with llvm_amdgpu_options (GPU needed) --


@pytest.mark.skipif(current_target.backend != "hip", reason="requires HIP backend")
def test_compile_accepts_llvm_amdgpu_options():
    """Compilation with llvm_amdgpu_options succeeds and produces valid assembly."""
    kernel = triton.compile(
        TTIR_PATH,
        target=current_target,
        options={"llvm_amdgpu_options": ("enable-misched=0", )},
    )
    assert "attn_fwd" in kernel.asm["amdgcn"]


@pytest.mark.skipif(current_target.backend != "hip", reason="requires HIP backend")
def test_llvm_amdgpu_options_change_codegen():
    """llvm_amdgpu_options produce different assembly than the default."""
    baseline = triton.compile(TTIR_PATH, target=current_target)
    with_options = triton.compile(
        TTIR_PATH,
        target=current_target,
        options={"llvm_amdgpu_options": ("enable-misched=0", )},
    )
    assert baseline.asm["amdgcn"] != with_options.asm["amdgcn"], \
        "expected llvm_amdgpu_options to change generated assembly"
