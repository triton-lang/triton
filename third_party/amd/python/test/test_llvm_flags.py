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


# -- Integration test: compile with AMDGCN_LLVM_OPTIONS env var (GPU needed) --


@pytest.mark.skipif(current_target.backend != "hip", reason="requires HIP backend")
def test_llvm_amdgpu_options_change_codegen(monkeypatch):
    """Setting AMDGCN_LLVM_OPTIONS env var produces different assembly."""
    monkeypatch.setattr(triton.knobs.compilation, "always_compile", True)

    baseline = triton.compile(TTIR_PATH, target=current_target)

    monkeypatch.setenv("AMDGCN_LLVM_OPTIONS", "enable-misched=0")

    with_options = triton.compile(TTIR_PATH, target=current_target)
    assert baseline.asm["amdgcn"] != with_options.asm["amdgcn"], \
        "expected AMDGCN_LLVM_OPTIONS to change generated assembly"
