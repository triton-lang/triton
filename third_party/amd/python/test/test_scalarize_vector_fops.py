import re
from pathlib import Path

import pytest
import triton

current_target = triton.runtime.driver.active.get_current_target()
if current_target.arch not in ("gfx950", "gfx940", "gfx941", "gfx942", "gfx90a", "gfx908"):
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
    kernel = triton.compile(
        str(Path(__file__).parent / "attn_fwd.ttir"),
        target=current_target,
    )
    # check for specific patterns that we'll be rewriting in the pass
    llir = kernel.asm["llir"]
    split_func_body = get_func_body(llir).splitlines()
    found_adjacent_fmul_mfma = False
    for i, l in enumerate(split_func_body):
        if "fmul <16" in l:
            if "mfma" in split_func_body[i + 1] or "wmma" in split_func_body[i + 1]:
                found_adjacent_fmul_mfma = True
                break
    assert found_adjacent_fmul_mfma, "didn't find adjacent fmul and mfma"

    # check that the pattern has the pessimistic effect on the assembly
    amdgcn = get_func_body_asm(kernel.asm["amdgcn"])
    bbs = list(re.split(r"^.L\w+:", amdgcn, flags=re.MULTILINE))
    found_mfma = False
    found_colliding_fmul = False
    found_colliding_fadd = False
    for bb in bbs:
        if "mfma" in bb or "wmma" in bb:
            found_mfma = True
        if "v_pk_mul" in bb and ("mfma" in bb or "wmma" in bb):
            found_colliding_fmul = True
        if "v_pk_add" in bb and ("mfma" in bb or "wmma" in bb):
            found_colliding_fadd = True

    assert (found_mfma and found_colliding_fmul and found_colliding_fadd), "couldn't find mfma or fmul or fadd"


# check scalarization "fixes"
def test_check_scalarized():
    kernel = triton.compile(
        str(Path(__file__).parent / "attn_fwd.ttir"),
        target=current_target,
        options={"scalarize_vector_fops": True},
    )

    # check the specific IR pattern was rewritten
    llir = kernel.asm["llir"]
    func_body = get_func_body(llir)
    bbs = list(re.split(r"^\d+:\s+; preds = %.*?$", func_body, flags=re.MULTILINE))
    assert len(bbs) > 1, "didn't split func body correctly"
    found_mfma = False
    found_fmul = False
    for bb in bbs:
        if "mfma" in bb or "wmma" in bb:
            assert "fmul <" not in bb
            assert "fadd <" not in bb
            found_mfma = True
        if "fmul <" in bb or "fadd <" in bb:
            assert not ("mfma" in bb or "wmma" in bb)
            found_fmul = True

    assert found_mfma and found_fmul, "couldn't find mfma or fmul"

    # check that it had the profitable effect on the assembly
    amdgcn = get_func_body_asm(kernel.asm["amdgcn"])
    assert "v_pk_add" not in amdgcn
    bbs = list(re.split(r"^.L\w+:", amdgcn, flags=re.MULTILINE))
    assert len(bbs) > 1, "couldn't split amdgcn bbs"
    found_mfma = False
    found_fmul = False
    for bb in bbs:
        if "mfma" in bb or "wmma" in bb:
            assert "v_pk_mul" not in bb
            found_mfma = True
        if "v_pk_mul" in bb:
            assert not ("mfma" in bb or "wmma" in bb)
            found_fmul = True
        # we don't check for v_pk_add because for this kernel,
        # there are no remaining v_pk_adds (the remaining v_pk_muls are in the epilogue)

    assert found_mfma and found_fmul, "couldn't find mfma or fmul"
