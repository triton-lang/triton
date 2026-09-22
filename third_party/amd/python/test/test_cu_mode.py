from pathlib import Path

import pytest
import torch

import triton
import triton.language as tl

from triton import knobs
from triton._internal_testing import is_hip

if not is_hip():
    pytest.skip(allow_module_level=True)


def _has_wgp_mode():
    """Whether the target distinguishes WGP and CU wavefront execution mode.

    gfx10, gfx11 and gfx120x spread a work-group over both CUs of a work-group
    processor unless CU mode is requested. gfx9 and gfx1250 have no such mode,
    so cu_mode is a no-op there and these tests have nothing to observe. Note
    "gfx120" rather than "gfx12", to avoid matching gfx1250.
    """
    arch = triton.runtime.driver.active.get_current_target().arch
    return arch.startswith(("gfx10", "gfx11", "gfx120"))


pytestmark = pytest.mark.skipif(not _has_wgp_mode(), reason="target has no WGP/CU mode distinction")

ATTN_FWD_TTIR = str(Path(__file__).parent / "attn_fwd.ttir")

WGP_DIRECTIVE = ".amdhsa_workgroup_processor_mode"
WGP, CU = "1", "0"


def _compile(**options):
    target = triton.runtime.driver.active.get_current_target()
    return triton.compile(ATTN_FWD_TTIR, target=target, options=options)


def _wgp_mode(asm):
    for line in asm.splitlines():
        if WGP_DIRECTIVE in line:
            return line.split()[-1]
    return None


@pytest.fixture
def cu_mode_knob():
    """Override the TRITON_HIP_CU_MODE default for the duration of a test."""
    previous = knobs.amd.cu_mode
    yield lambda value: setattr(knobs.amd, "cu_mode", value)
    knobs.amd.cu_mode = previous


# Tests that are not about the knob pass cu_mode explicitly on both sides, so
# they do not depend on whether TRITON_HIP_CU_MODE is set in the environment.


def test_cu_mode_sets_workgroup_processor_mode():
    assert _wgp_mode(_compile(cu_mode=False).asm["amdgcn"]) == WGP
    assert _wgp_mode(_compile(cu_mode=True).asm["amdgcn"]) == CU


def test_cu_mode_sets_target_feature():
    assert "+cumode" not in _compile(cu_mode=False).asm["llir"]
    assert "+cumode" in _compile(cu_mode=True).asm["llir"]


def test_cu_mode_defaults_to_knob(cu_mode_knob):
    cu_mode_knob(True)
    assert _wgp_mode(_compile().asm["amdgcn"]) == CU
    cu_mode_knob(False)
    assert _wgp_mode(_compile().asm["amdgcn"]) == WGP


def test_cu_mode_option_overrides_knob(cu_mode_knob):
    cu_mode_knob(True)
    assert _wgp_mode(_compile(cu_mode=False).asm["amdgcn"]) == WGP
    cu_mode_knob(False)
    assert _wgp_mode(_compile(cu_mode=True).asm["amdgcn"]) == CU


def test_cu_mode_none_means_unspecified(cu_mode_knob):
    # Callers forwarding an optional value must not accidentally suppress the
    # knob; None means "not specified", not "off".
    cu_mode_knob(True)
    assert _wgp_mode(_compile(cu_mode=None).asm["amdgcn"]) == CU


def test_cu_mode_rejects_asan():
    """ASAN also writes "target-features"; one would silently replace the other."""
    previous = knobs.compilation.enable_asan
    knobs.compilation.enable_asan = True
    try:
        with pytest.raises(Exception, match="cu_mode"):
            _compile(cu_mode=True)
    finally:
        knobs.compilation.enable_asan = previous


def test_cu_mode_rejects_conflicting_target_features():
    with pytest.raises(Exception, match="target-features"):
        _compile(cu_mode=True, llvm_fn_attrs="target-features=+wavefrontsize32")


@triton.jit
def _add_one(x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs, tl.load(x_ptr + offs) + 1)


def test_cu_mode_as_launch_argument():
    """cu_mode is usable as a kernel launch flag, not just via triton.compile."""
    x = torch.zeros(64, device="cuda")

    baseline = _add_one[(1, )](x, BLOCK=64, cu_mode=False)
    with_cu = _add_one[(1, )](x, BLOCK=64, cu_mode=True)

    assert _wgp_mode(baseline.asm["amdgcn"]) == WGP
    assert _wgp_mode(with_cu.asm["amdgcn"]) == CU
    assert torch.equal(x, torch.full_like(x, 2))


@triton.jit
def _double(src, dst, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(dst + offs, tl.load(src + offs) * 2)


def test_cu_mode_autotune_configs_compile_and_launch():
    """Both configs must really build and run, not just round-trip through Config."""
    configs = [
        triton.Config({"BLOCK": 64, "cu_mode": False}, num_warps=1),
        triton.Config({"BLOCK": 64, "cu_mode": True}, num_warps=1),
    ]

    src = torch.arange(64, device="cuda", dtype=torch.float32)
    modes = []
    for config in configs:
        dst = torch.empty_like(src)
        handle = _double[(1, )](src, dst, 64, **config.all_kwargs())
        modes.append(_wgp_mode(handle.asm["amdgcn"]))
        torch.testing.assert_close(dst, src * 2)
    assert modes == [WGP, CU]

    # And the autotuner itself drives them end to end.
    @triton.autotune(configs=configs, key=["n"])
    @triton.jit
    def _autotuned_double(src, dst, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        tl.store(dst + offs, tl.load(src + offs) * 2)

    dst = torch.empty_like(src)
    _autotuned_double[(1, )](src, dst, 64)
    torch.testing.assert_close(dst, src * 2)


@triton.jit
def _cross_warp_sum(src, dst, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    total = tl.sum(tl.load(src + offs))
    tl.store(dst + offs, tl.zeros([BLOCK], tl.float32) + total)


@pytest.mark.parametrize("cu_mode", [False, True])
def test_cross_warp_reduction_is_correct(cu_mode):
    """Exercise the barrier lowering this change affects.

    A reduction wider than one wave exchanges partial results through LDS and
    synchronises with ttg.barrier, which is the op whose lowering cu_mode now
    switches on RDNA. Compiling is not enough here: the point is that waves
    still observe each other's data under the tagged lowering.
    """
    BLOCK = 1024
    src = torch.rand(BLOCK, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)

    handle = _cross_warp_sum[(1, )](src, dst, BLOCK=BLOCK, num_warps=8, cu_mode=cu_mode)
    asm = handle.asm["amdgcn"]

    # Guard the premise: without a barrier the test says nothing about waves
    # observing each other.
    assert "s_barrier" in asm
    assert _wgp_mode(asm) == (CU if cu_mode else WGP)
    torch.testing.assert_close(dst, src.sum().expand(BLOCK), rtol=1e-4, atol=1e-4)
