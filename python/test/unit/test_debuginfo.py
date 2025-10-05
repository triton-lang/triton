import os

import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def checkDbgInfo(llir, hasDbgInfo):
    assert hasDbgInfo == ('dbg_value' in llir)
    for name in ["offsets", "pid", "block_start", "mask", "x", "y", "output"]:
        assert hasDbgInfo == ('!DILocalVariable(name: \"' + name + '\"' in llir)


@pytest.mark.parametrize("lineInfoKey, diLocalVarKey, hasDbgInfo", [
    (None, None, False),
    # expect dbginfo based on parent proccess' TRITON_DISABLE_LINE_INFO
    (None, "1", "infer"),
    ("0", "1", True),
    ("1", "1", False),
    ("0", "0", False),
    ("1", "0", False),
])
def test_triton_debuginfo_on(lineInfoKey, diLocalVarKey, hasDbgInfo, device, monkeypatch):
    lineInfoKeyName = "TRITON_DISABLE_LINE_INFO"
    diLocalVarKeyName = "LLVM_EXTRACT_DI_LOCAL_VARIABLES"
    if lineInfoKey is not None:
        monkeypatch.setenv(lineInfoKeyName, lineInfoKey)
    if diLocalVarKey is not None:
        monkeypatch.setenv(diLocalVarKeyName, diLocalVarKey)

    isEnvSet = lambda env, str: env.get(str, None) is not None
    if hasDbgInfo == "infer":
        hasDbgInfo = (not isEnvSet(os.environ, lineInfoKeyName)
                      or os.environ[lineInfoKeyName].lower() not in ["on", "true", "1"])

    size = 98432
    torch.manual_seed(0)
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel.device_caches.clear()
    h = add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    checkDbgInfo(h.asm['llir'], hasDbgInfo)
