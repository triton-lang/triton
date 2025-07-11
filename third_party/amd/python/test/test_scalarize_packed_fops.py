import re
from pathlib import Path

import pytest
import triton

current_target = triton.runtime.driver.active.get_current_target()
if current_target.arch not in ("gfx950", "gfx942", "gfx90a", "gfx908"):
    pytest.skip(allow_module_level=True)


def get_func_body(llir):
    func_body = re.findall(r"define amdgpu_kernel void .*? \{(.* ret void.*?)}", llir, flags=re.DOTALL)
    assert len(func_body) == 1, "couldn't find kernel body"
    return func_body[0]


def get_func_body_asm(amdgcn):
    amdgcn = re.findall(r"^attn_fwd:(.*); -- End function", amdgcn, flags=re.DOTALL | re.MULTILINE)
    assert len(amdgcn) == 1, "couldn't find kernel body"
    return amdgcn[0]


# check there are actually instances of colliding/adjacent fops and mfma without scalarization
def test_check_not_scalarize():
    triton.knobs.amd.scalarize_packed_fops = False
    kernel = triton.compile(str(Path(__file__).parent / "attn_fwd.ttir"), target=current_target)
    llir = kernel.asm["llir"]
    func_body = get_func_body(llir)

    # check for specific patterns that we'll be rewriting in the pass
    def checked_packed_fops_ir_bbs():
        bbs = list(re.split(r"^\d+:\s+; preds = %.*?$", func_body, flags=re.MULTILINE))
        assert len(bbs) > 1, "didn't split func body correctly"
        found_colliding_packed_fop = False
        packed_fop = re.compile(r"= f(add|sub|mul) <")
        for bb in bbs:
            if ("mfma" in bb or "wmma" in bb) and packed_fop.search(bb):
                found_colliding_packed_fop = True
        assert found_colliding_packed_fop, "couldn't find adjacent packed fop and mfma"

    # check that the pattern has the pessimistic effect on the assembly
    amdgcn = get_func_body_asm(kernel.asm["amdgcn"])

    def checked_packed_fops_asm_bbs():
        bbs = list(re.split(r"^.L\w+:", amdgcn, flags=re.MULTILINE))
        assert len(bbs) > 1, "didn't split func body correctly"
        found_mfma = False
        found_colliding_packed_fop = False
        packed_fop = re.compile(r"v_pk_\w+")
        for bb in bbs:
            if "mfma" in bb or "wmma" in bb:
                found_mfma = True
            if packed_fop.search(bb) and ("mfma" in bb or "wmma" in bb):
                found_colliding_packed_fop = True

        assert (found_mfma and found_colliding_packed_fop
                ), f"couldn't find mfma or packed fop {found_mfma=} {found_colliding_packed_fop=}"

    checked_packed_fops_ir_bbs()
    checked_packed_fops_asm_bbs()


# check scalarization "fixes"
def test_check_scalarized():
    triton.knobs.amd.scalarize_packed_fops = True
    triton.knobs.amd.use_buffer_ops = True

    kernel = triton.compile(str(Path(__file__).parent / "attn_fwd.ttir"), target=current_target)

    # check the specific IR pattern was rewritten
    llir = kernel.asm["llir"]
    func_body = get_func_body(llir)

    def checked_packed_fops_ir_bbs():
        bbs = list(re.split(r"^\d+:\s+; preds = %.*?$", func_body, flags=re.MULTILINE))
        assert len(bbs) > 1, "didn't split func body correctly"
        found_mfma = False
        packed_fop = re.compile(r"= f(add|sub|mul) <")
        for bb in bbs:
            if "mfma" in bb or "wmma" in bb:
                assert not packed_fop.search(bb)
                found_mfma = True
            if packed_fop.search(bb):
                assert not ("mfma" in bb or "wmma" in bb)

        assert found_mfma, "couldn't find packed mfma"

    # check that it had the profitable effect on the assembly
    amdgcn = get_func_body_asm(kernel.asm["amdgcn"])

    def checked_packed_fops_asm_bbs():
        bbs = list(re.split(r"^.L\w+:", amdgcn, flags=re.MULTILINE))
        assert len(bbs) > 1, "couldn't split amdgcn bbs"
        found_mfma = False
        found_packed_fop = False
        packed_fop = re.compile(r"v_pk_\w+")
        for bb in bbs:
            if "mfma" in bb or "wmma" in bb:
                assert not packed_fop.search(bb)
                found_mfma = True
            if packed_fop.search(bb):
                assert not ("mfma" in bb or "wmma" in bb)
                found_packed_fop = True
            # we don't check for v_pk_add because for this kernel,
            # there are no remaining v_pk_adds (the remaining v_pk_muls are in the epilogue)

        assert found_mfma and found_packed_fop, f"couldn't find mfma or packed fop: {found_mfma=}, {found_packed_fop=}"

    checked_packed_fops_ir_bbs()
    checked_packed_fops_asm_bbs()
