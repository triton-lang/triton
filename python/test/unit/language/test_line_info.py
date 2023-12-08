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


# AddIOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)
# Since the + symbol will take effect in the dot op after combination,
# it seems making sense to annotate with the same line as dot.
@triton.jit
def kernel_dot_combine(x):
    c = tl.full((32, 32), 4, dtype=tl.int8)
    a = (tl.arange(0, 32)[:, None] + tl.arange(0, 32)[None, :]).to(tl.int8)
    d = tl.dot(a, a)
    d = d + c
    tl.device_print("", d)


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


def check_file_lines(file_lines, file_name, lineno, should_contain=True):
    """
    Check if the file name and line number is in the file_lines

    Args:
        file_lines: list of (file_name, line_number)
        file_name: file name
        lineno: line number, -1 means do not check line number
        should_contain: whether the file name and line number should be in the file_lines
    """
    for file, line in file_lines:
        if lineno == -1:
            if file_name in file:
                return True
        if file_name in file and str(lineno) in line:
            return should_contain
    return not should_contain


func_types = ["single", "call", "call_noinline", "multi_files", "autotune", "dot_combine"]


@pytest.mark.parametrize("func", func_types)
def test_line_info(func: str):
    try:
        _, _ = path_to_nvdisasm()
    except BaseException:
        pytest.skip("nvdisasm is not available")

    shape = (128, )
    kernel_info = {}
    if func == "single":
        kernel_info = kernel_single.warmup(torch.float32, torch.float32, BLOCK=shape[0], grid=(1,))
    elif func == "call":
        kernel_info = kernel_call.warmup(torch.float32, torch.float32, BLOCK=shape[0], grid=(1,))
    elif func == "call_noinline":
        kernel_info = kernel_call_noinline.warmup(torch.float32, torch.float32, BLOCK=shape[0], grid=(1,))
    elif func == "multi_files":
        kernel_info = kernel_multi_files.warmup(torch.float32, torch.float32, BLOCK=shape[0], grid=(1,))
    elif func == "autotune":
        kernel_info = kernel_autotune.warmup(torch.float32, torch.float32, SIZE=shape[0], grid=(1,))[0]
    elif func == "dot_combine":
        kernel_info = kernel_dot_combine.warmup(20, grid=(1,))

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
        assert (check_file_lines(file_lines, "standard.py", 35))
        assert (check_file_lines(file_lines, "standard.py", 36))
        assert (check_file_lines(file_lines, "standard.py", 38))
    elif func == "autotune":
        assert (check_file_lines(file_lines, "test_line_info.py", 60))
        assert (check_file_lines(file_lines, "test_line_info.py", 61))
        assert (check_file_lines(file_lines, "test_line_info.py", 62))
        assert (check_file_lines(file_lines, "test_line_info.py", 63))
    elif func == "dot_combine":
        assert (check_file_lines(file_lines, "test_line_info.py", 73))
        assert (check_file_lines(file_lines, "test_line_info.py", 74, should_contain=False))
