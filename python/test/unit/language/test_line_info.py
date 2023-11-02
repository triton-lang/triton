import subprocess
import tempfile

import pytest
import torch

import triton
import triton.language as tl
from triton.common.backend import path_to_nvdisasm


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


@triton.jit
def kernel_multi_files(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = tl.softmax(x)
    tl.store(Y + tl.arange(0, BLOCK), y)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=4),
    ],
    key=[],
)
@triton.jit
def kernel_autotune(X, Y, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    for i in range(0, SIZE, BLOCK):
        x = tl.load(X + i + tl.arange(0, BLOCK))
        tl.store(Y + i + tl.arange(0, BLOCK), x)


def extract_file_lines(asm):
    nvdisasm, _ = path_to_nvdisasm()
    fd, path = tempfile.mkstemp()
    with open(fd, 'wb') as cubin:
        cubin.write(asm)
    asm = subprocess.check_output([nvdisasm, "-g", path]).decode("utf-8")
    file_lines = []
    lines = asm.splitlines()
    for line in lines:
        if "## File" in line:
            entries = line[line.index("## File"):].split(",")
            file_lines.append((entries[0].strip(), entries[1].strip()))
    return file_lines


def check_file_lines(file_lines, file_name, lineno):
    for file, line in file_lines:
        # -1 means do not check line number
        if lineno == -1:
            if file_name in file:
                return True
        if file_name in file and str(lineno) in line:
            return True
    return False


func_types = ["single", "call", "call_noinline", "multi_files", "autotune"]


@pytest.mark.parametrize("func", func_types)
def test_line_info(func: str):
    try:
        _, _ = path_to_nvdisasm()
    except BaseException:
        pytest.skip("nvdisasm is not available")

    shape = (128, )
    x = torch.arange(0, shape[0], dtype=torch.float32, device='cuda')
    y = torch.zeros(shape, dtype=x.dtype, device="cuda")
    kernel_info = {}
    if func == "single":
        kernel_info = kernel_single[(1,)](x, y, BLOCK=shape[0])
    elif func == "call":
        kernel_info = kernel_call[(1,)](x, y, BLOCK=shape[0])
    elif func == "call_noinline":
        kernel_info = kernel_call_noinline[(1,)](x, y, BLOCK=shape[0])
    elif func == "multi_files":
        kernel_info = kernel_multi_files[(1,)](x, y, BLOCK=shape[0])
    elif func == "autotune":
        kernel_info = kernel_autotune[(1,)](x, y, SIZE=shape[0])

    file_lines = extract_file_lines(kernel_info.asm["cubin"])
    if func == "single":
        assert (check_file_lines(file_lines, "test_line_info.py", 16))
        assert (check_file_lines(file_lines, "test_line_info.py", 17))
    elif func == "call":
        assert (check_file_lines(file_lines, "test_line_info.py", 29))
        assert (check_file_lines(file_lines, "test_line_info.py", 22))
        assert (check_file_lines(file_lines, "test_line_info.py", 31))
    elif func == "call_noinline":
        assert (check_file_lines(file_lines, "test_line_info.py", 43))
        assert (check_file_lines(file_lines, "test_line_info.py", 36))
        assert (check_file_lines(file_lines, "test_line_info.py", 37))
        assert (check_file_lines(file_lines, "test_line_info.py", 38))
    elif func == "multi_files":
        assert (check_file_lines(file_lines, "test_line_info.py", 48))
        assert (check_file_lines(file_lines, "test_line_info.py", 50))
        assert (check_file_lines(file_lines, "standard.py", 33))
        assert (check_file_lines(file_lines, "standard.py", 34))
        assert (check_file_lines(file_lines, "standard.py", 36))
    elif func == "autotune":
        assert (check_file_lines(file_lines, "test_line_info.py", 60))
        assert (check_file_lines(file_lines, "test_line_info.py", 61))
        assert (check_file_lines(file_lines, "test_line_info.py", 62))
        assert (check_file_lines(file_lines, "test_line_info.py", 63))
