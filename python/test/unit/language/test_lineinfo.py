import os
import subprocess
import tempfile

import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def kernel_single(X,
                  Y,
                  BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def device_inline(x):
    return x + x


@triton.jit
def kernel_call(X,
                Y,
                BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = device_inline(x)
    tl.store(Y + tl.arange(0, BLOCK), y)


@triton.jit(noinline=True)
def device_noinline(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = x + x
    tl.store(Y + tl.arange(0, BLOCK), y)


@triton.jit
def kernel_call_noinline(X, Y, BLOCK: tl.constexpr):
    device_noinline(X, Y, BLOCK)


def extract_file_lines(asm):
    fd, path = tempfile.mkstemp()
    with open(fd, 'wb') as cubin:
        cubin.write(asm)
    asm = subprocess.check_output(["nvdisasm", "-g", path]).decode("utf-8")
    file_lines = []
    lines = asm.splitlines()
    for line in lines:
        if "## File" in line:
            entries = line[line.index("## File"):].split(",")
            file_lines.append((entries[0].strip(), entries[1].strip()))
    return file_lines


def check_file_lines(file_lines, lineno):
    for file, line in file_lines:
        if "## File" in file and str(lineno) in line:
            return True
    return False


func_types = ["single", "call", "call_noinline"]


@pytest.mark.parametrize("func", func_types)
def test_line_info(func: str):
    try:
        subprocess.check_output(["nvdisasm", "-h"])
    except BaseException:
        pytest.skip("nvdisasm is not available")
    os.environ["TRITON_LINEINFO"] = "1"

    shape = (128, )
    x = torch.arange(0, shape[0], dtype=torch.int32, device='cuda')
    y = torch.zeros(shape, dtype=x.dtype, device="cuda")
    kernel_info = {}
    if func == "single":
        kernel_info = kernel_single[(1,)](x, y, BLOCK=shape[0])
    elif func == "call":
        kernel_info = kernel_call[(1,)](x, y, BLOCK=shape[0])
    elif func == "call_noinline":
        kernel_info = kernel_call_noinline[(1,)](x, y, BLOCK=shape[0])

    file_lines = extract_file_lines(kernel_info.asm["cubin"])
    if func == "single":
        assert (check_file_lines(file_lines, 15))
        assert (check_file_lines(file_lines, 16))
    elif func == "call":
        assert (check_file_lines(file_lines, 28))
        assert (check_file_lines(file_lines, 21))
        assert (check_file_lines(file_lines, 30))
    elif func == "call_noinline":
        assert (check_file_lines(file_lines, 42))
        assert (check_file_lines(file_lines, 35))
        assert (check_file_lines(file_lines, 36))
        assert (check_file_lines(file_lines, 37))
