from pathlib import Path

import pytest
import triton

from triton._internal_testing import is_hip

if not is_hip():
    pytest.skip(allow_module_level=True)

ATTN_FWD_TTIR = str(Path(__file__).parent / "attn_fwd.ttir")


def test_llvm_fn_attrs_set_single_llir_attribute():
    target = triton.runtime.driver.active.get_current_target()

    baseline = triton.compile(ATTN_FWD_TTIR, target=target)
    with_attrs = triton.compile(
        ATTN_FWD_TTIR,
        target=target,
        options={"llvm_fn_attrs": "amdgpu-sched-strategy=iterative-ilp"},
    )

    assert '"amdgpu-sched-strategy"="iterative-ilp"' not in baseline.asm["llir"]
    assert '"amdgpu-sched-strategy"="iterative-ilp"' in with_attrs.asm["llir"]


def test_llvm_fn_attrs_set_multiple_llir_attributes():
    target = triton.runtime.driver.active.get_current_target()
    llvm_fn_attrs = ",".join([
        "amdgpu-sched-strategy=iterative-ilp",
        "triton-test-attr=enabled",
        "triton-bare-attr",
    ])

    with_attrs = triton.compile(
        ATTN_FWD_TTIR,
        target=target,
        options={"llvm_fn_attrs": llvm_fn_attrs},
    )

    assert '"amdgpu-sched-strategy"="iterative-ilp"' in with_attrs.asm["llir"]
    assert '"triton-test-attr"="enabled"' in with_attrs.asm["llir"]
    assert '"triton-bare-attr"' in with_attrs.asm["llir"]
    assert '"triton-bare-attr"=' not in with_attrs.asm["llir"]


def test_llvm_fn_attrs_change_amdgcn():
    target = triton.runtime.driver.active.get_current_target()

    baseline = triton.compile(ATTN_FWD_TTIR, target=target)
    with_sched_attr = triton.compile(
        ATTN_FWD_TTIR,
        target=target,
        options={"llvm_fn_attrs": "amdgpu-sched-strategy=iterative-ilp"},
    )

    assert baseline.asm["amdgcn"] != with_sched_attr.asm["amdgcn"]
