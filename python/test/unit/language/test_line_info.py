import subprocess
import tempfile

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_interpreter


@triton.jit
def kernel_single(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def device_inline(x):
    return x + x


@triton.jit
def kernel_call(X, Y, BLOCK: tl.constexpr):
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


# Call another jit function (cdiv) not in this file
@triton.jit
def kernel_cdiv(x):
    c = tl.full((32, 32), 4, dtype=tl.int8)
    d = tl.cdiv(c, 4)
    tl.device_print("", d)


def get_disassembler_command_and_debug_line_format():
    """Gets backend specific disassembler information.

    Returns a tuple: (object file kind, disassembler tool command,
    debug line anchor, debug line file and line number separator).
    """
    backend = triton.runtime.driver.active.get_current_target().backend

    if backend == "cuda":
        nvdisasm = triton.knobs.nvidia.nvdisasm.path
        return ("cubin", [nvdisasm, "-g"], "## File", ",")

    if backend == "hip":
        import shutil
        # Try to find llvm-objdump from the current PATH to disassmble hsaco.
        tool = shutil.which("llvm-objdump")
        if tool is not None:
            return ("hsaco", [tool, "-D", "-l", "--arch=amdgcn"], ";", ":")
        raise RuntimeError("llvm-objdump not found in PATH")

    raise RuntimeError(f"unknown backend {backend}")


def extract_file_lines(command, anchor, separator, asm):
    fd, path = tempfile.mkstemp()
    with open(fd, 'wb') as cubin:
        cubin.write(asm)
    asm = subprocess.check_output(command + [path]).decode("utf-8")
    file_lines = []
    lines = asm.splitlines()
    for line in lines:
        # We are looking for an anchor string and a separator between the file name and line number.
        if anchor in line and separator in line:
            entries = line[line.index(anchor):].split(separator)
            if len(entries) == 2 and all(len(e) != 0 for e in entries):
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
        if lineno == -1 and file_name in file:
            return True
        if file_name in file and str(lineno) in line:
            return should_contain
    return not should_contain


func_types = ["single", "call", "call_noinline", "autotune", "dot_combine", "cdiv"]


@pytest.mark.parametrize("func", func_types)
def test_line_info(func: str):
    try:
        obj_kind, command, anchor, separator = get_disassembler_command_and_debug_line_format()
    except BaseException:
        pytest.skip("disassembler is not available")

    shape = (128, )
    kernel_info = {}
    if func == "single":
        kernel_info = kernel_single.warmup(torch.float32, torch.float32, BLOCK=shape[0], grid=(1, ))
    elif func == "call":
        kernel_info = kernel_call.warmup(torch.float32, torch.float32, BLOCK=shape[0], grid=(1, ))
    elif func == "call_noinline":
        kernel_info = kernel_call_noinline.warmup(torch.float32, torch.float32, BLOCK=shape[0], grid=(1, ))
    elif func == "autotune":
        kernel_info = kernel_autotune.warmup(torch.float32, torch.float32, SIZE=shape[0], grid=(1, ))[0]
    elif func == "dot_combine":
        kernel_info = kernel_dot_combine.warmup(20, grid=(1, ))
    elif func == "cdiv":
        kernel_info = kernel_cdiv.warmup(20, grid=(1, ))

    file_lines = extract_file_lines(command, anchor, separator, kernel_info.asm[obj_kind])
    backend = triton.runtime.driver.active.get_current_target().backend

    if func == "single":
        if backend != "hip":
            # removed for release/3.5 w/ turning off AMD buffer ops
            assert (check_file_lines(file_lines, "test_line_info.py", 14))
        assert (check_file_lines(file_lines, "test_line_info.py", 15))
    elif func == "call":
        assert (check_file_lines(file_lines, "test_line_info.py", 25))
        assert (check_file_lines(file_lines, "test_line_info.py", 27))
    elif func == "call_noinline":
        assert (check_file_lines(file_lines, "test_line_info.py", 39))
        assert (check_file_lines(file_lines, "test_line_info.py", 32))
        assert (check_file_lines(file_lines, "test_line_info.py", 32))
    elif func == "autotune":
        assert (check_file_lines(file_lines, "test_line_info.py", 50))
        if backend != "hip":
            # removed for release/3.5 w/ turning off AMD buffer ops
            assert (check_file_lines(file_lines, "test_line_info.py", 51))
        assert (check_file_lines(file_lines, "test_line_info.py", 52))
    elif func == "dot_combine":
        assert (check_file_lines(file_lines, "test_line_info.py", 62))
        assert (check_file_lines(file_lines, "test_line_info.py", 63, should_contain=False))
    elif func == "cdiv":
        assert (check_file_lines(file_lines, "test_line_info.py", 72))


@pytest.mark.interpreter
@pytest.mark.parametrize("func", func_types)
def test_line_info_interpreter(func: str):
    if not is_interpreter():
        pytest.skip("interpreter is not enabled")

    kernel = None
    expected_def_lineno = 0
    if func == "single":
        kernel = kernel_single
        expected_def_lineno = 13
    elif func == "call":
        kernel = kernel_call
        expected_def_lineno = 24
    elif func == "call_noinline":
        kernel = kernel_call_noinline
        expected_def_lineno = 38
    elif func == "autotune":
        kernel = kernel_autotune.fn
        expected_def_lineno = 49
    elif func == "dot_combine":
        kernel = kernel_dot_combine
        expected_def_lineno = 59
    elif func == "cdiv":
        kernel = kernel_cdiv
        expected_def_lineno = 69
    kernel.rewrite()
    assert kernel.rewriter.def_file_lineno == expected_def_lineno


@pytest.mark.parametrize("status", ["0", "1"])
def test_line_info_env(monkeypatch, status: str):
    try:
        obj_kind, command, anchor, separator = get_disassembler_command_and_debug_line_format()
    except BaseException:
        pytest.skip("disassembler is not available")

    shape = (128, )
    monkeypatch.setenv("TRITON_DISABLE_LINE_INFO", status)
    kernel_single.device_caches.clear()
    kernel_info = kernel_single.warmup(torch.float32, torch.float32, BLOCK=shape[0], grid=(1, ))
    file_lines = extract_file_lines(command, anchor, separator, kernel_info.asm[obj_kind])
    assert len(file_lines) == 0 if status == "1" else len(file_lines) > 0


@pytest.mark.parametrize("status", ["ttir", ""])
def test_line_info_ir_source(monkeypatch, status, tmp_path):
    try:
        obj_kind, command, anchor, separator = get_disassembler_command_and_debug_line_format()
    except BaseException:
        pytest.skip("disassembler is not available")

    src = """
    #loc = loc("/path/test.py":7:0)
    module {
    tt.func public @test(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/path/test.py":7:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/path/test.py":7:0)) attributes {noinline = false} {
        %0 = tt.load %arg0 : !tt.ptr<f32> loc(#loc1)
        tt.store %arg1, %0 : !tt.ptr<f32> loc(#loc2)
        tt.return loc(#loc3)
    } loc(#loc)
    } loc(#loc)
    #loc1 = loc("/path/test.py":8:16)
    #loc2 = loc("/path/test.py":9:20)
    #loc3 = loc("/path/test.py":9:4)
    """
    monkeypatch.setenv("USE_IR_LOC", status)
    temp_file = tmp_path / "test.ttir"
    temp_file.write_text(src)
    kernel_info = triton.compile(str(temp_file))
    file_lines = extract_file_lines(command, anchor, separator, kernel_info.asm[obj_kind])
    if status == "ttir":
        assert check_file_lines(file_lines, "/path/test.py", 8, should_contain=False)
        assert check_file_lines(file_lines, str(temp_file), -1, should_contain=True)
    else:
        assert check_file_lines(file_lines, "/path/test.py", 8, should_contain=True)


def test_use_name_loc_as_prefix(fresh_triton_cache):
    import inspect
    from triton._filecheck import run_filecheck

    @triton.jit
    def kernel_basic(src, N, BLOCK_SIZE: tl.constexpr):
        # CHECK: #loc = loc("{{.*}}":267:0)
        # CHECK-LABEL:  tt.func public @kernel_basic(
        # CHECK-SAME:                                %src: !tt.ptr<f32> loc("src"(#loc)), %N: i32 loc("N"(#loc)))
        # CHECK:          %x_plus_1 = arith.constant dense<1.000000e+00> : tensor<16xf32> loc(#loc14)
        # CHECK:          %c16_i32 = arith.constant 16 : i32 loc(#loc2)
        # CHECK:          %pid = tt.get_program_id x : i32 loc(#loc15)
        # CHECK:          %offset = arith.muli %pid, %c16_i32 : i32 loc(#loc16)
        # CHECK:          %offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc17)
        # CHECK:          %offsets_0 = tt.splat %offset : i32 -> tensor<16xi32> loc(#loc18)
        # CHECK:          %offsets_1 = arith.addi %offsets_0, %offsets : tensor<16xi32> loc(#loc18)
        # CHECK:          %load_src_store_dst = tt.splat %src : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>> loc(#loc19)
        # CHECK:          %load_src_store_dst_2 = tt.addptr %load_src_store_dst, %offsets_1 : tensor<16x!tt.ptr<f32>>, tensor<16xi32> loc(#loc19)
        # CHECK:          %mask = tt.splat %N : i32 -> tensor<16xi32> loc(#loc20)
        # CHECK:          %mask_3 = arith.cmpi slt, %offsets_1, %mask : tensor<16xi32> loc(#loc20)
        # CHECK:          %x_plus_1_4 = tt.load %load_src_store_dst_2, %mask_3 : tensor<16x!tt.ptr<f32>> loc(#loc21)
        # CHECK:          %x_plus_1_5 = arith.addf %x_plus_1_4, %x_plus_1 : tensor<16xf32> loc(#loc14)
        # CHECK:          tt.store %load_src_store_dst_2, %x_plus_1_5, %mask_3 : tensor<16x!tt.ptr<f32>> loc(#loc10)
        # CHECK:          tt.return loc(#loc11)
        # CHECK:          } loc(#loc)
        # CHECK:         } loc(#loc)

        # CHECK: #loc1 = loc({{.*}})
        # CHECK: #loc2 = loc(unknown)
        # CHECK: #loc3 = loc({{.*}})
        # CHECK: #loc4 = loc({{.*}})
        # CHECK: #loc5 = loc({{.*}})
        # CHECK: #loc6 = loc({{.*}})
        # CHECK: #loc7 = loc({{.*}})
        # CHECK: #loc8 = loc({{.*}})
        # CHECK: #loc9 = loc({{.*}})
        # CHECK: #loc10 = loc({{.*}})
        # CHECK: #loc11 = loc({{.*}})
        # CHECK: #loc14 = loc("x_plus_1"(#loc1))
        # CHECK: #loc15 = loc("pid"(#loc3))
        # CHECK: #loc16 = loc("offset"(#loc4))
        # CHECK: #loc17 = loc("offsets"(#loc5))
        # CHECK: #loc18 = loc("offsets"(#loc6))
        # CHECK: #loc19 = loc("load_src_store_dst"(#loc7))
        # CHECK: #loc20 = loc("mask"(#loc8))
        # CHECK: #loc21 = loc("x_plus_1"(#loc9))

        pid = tl.program_id(0)
        offset = pid * BLOCK_SIZE
        offsets = offset + tl.arange(0, BLOCK_SIZE)
        load_src_store_dst = src + offsets
        mask = offsets < N
        x_plus_1 = tl.load(load_src_store_dst, mask=mask) + 1
        tl.store(load_src_store_dst, x_plus_1, mask=mask)

    h = triton.compile(
        triton.compiler.ASTSource(fn=kernel_basic, signature={"src": "*fp32", "N": "i32", "BLOCK_SIZE": "constexpr"},
                                  constexprs={"BLOCK_SIZE": 16}))

    check_template = inspect.getsource(kernel_basic.fn)
    run_filecheck("placeholder", h.asm["ttir"], check_template)

    @triton.jit
    def kernel_basic_for_loop(N):
        # CHECK-LABEL: tt.func public @kernel_basic_for_loop

        # CHECK: scf.for %ivar = %c0_i32 to %N step %c1_i32
        for ivar in range(N):
            tl.device_print("", ivar)

    h = triton.compile(triton.compiler.ASTSource(fn=kernel_basic_for_loop, signature={"N": "i32"}, constexprs={}))

    check_template = inspect.getsource(kernel_basic_for_loop.fn)
    run_filecheck("placeholder", h.asm["ttir"], check_template)

    @triton.jit
    def kernel_basic_for_loop_with_block_args(N):
        # CHECK-LABEL: tt.func public @kernel_basic_for_loop_with_block_args

        # CHECK: %arange = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
        arange = tl.arange(0, 16)
        # CHECK: %arange_0 = scf.for %ivar = %c0_i32 to %N step %c1_i32 iter_args(%arange_1 = %arange) -> (tensor<16xi32>)
        for ivar in range(N):
            # CHECK: %arange_2 = arith.addi %arange_1, %arange_1 : tensor<16xi32>
            arange += arange
            # scf.yield %arange_2 : tensor<16xi32>

        tl.device_print("", arange)

    h = triton.compile(
        triton.compiler.ASTSource(fn=kernel_basic_for_loop_with_block_args, signature={"N": "i32"}, constexprs={}))

    check_template = inspect.getsource(kernel_basic_for_loop_with_block_args.fn)
    run_filecheck("placeholder", h.asm["ttir"], check_template)

    @triton.jit
    def kernel_basic_if(N):
        # CHECK-LABEL: tt.func public @kernel_basic_if

        # CHECK-DAG: %cst = arith.constant dense<4> : tensor<16xi32>
        # CHECK-DAG: %cst_0 = arith.constant dense<2> : tensor<16xi32>

        # CHECK: %arange = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
        arange = tl.arange(0, 16)

        if N > 2:
            # CHECK: %arange_1 = arith.muli %arange, %cst_0 : tensor<16xi32>
            arange *= 2
            # CHECK: scf.yield %arange_1 : tensor<16xi32>
        else:
            # CHECK: %arange_1 = arith.muli %arange, %cst : tensor<16xi32>
            arange *= 4
            # CHECK: scf.yield %arange_1 : tensor<16xi32>

        tl.device_print("", arange)

    h = triton.compile(triton.compiler.ASTSource(fn=kernel_basic_if, signature={"N": "i32"}, constexprs={}))

    check_template = inspect.getsource(kernel_basic_if.fn)
    run_filecheck("placeholder", h.asm["ttir"], check_template)

    @triton.jit
    def kernel_basic_if_top_level(N):
        # CHECK-LABEL: tt.func public @kernel_basic_if_top_level

        # CHECK: %arange = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
        arange = tl.arange(0, 16)
        if N == 0:
            # CHECK: %arange_0 = arith.addi %arange, %arange : tensor<16xi32>
            arange += tl.arange(0, 16)
            tl.device_print("", arange)
            return
        else:
            # CHECK: %new_arange = tt.make_range {end = 32 : i32, start = 16 : i32} : tensor<16xi32>
            new_arange = tl.arange(16, 32)
            # CHECK: %arange_1 = arith.addi %arange, %new_arange : tensor<16xi32>
            arange += new_arange
            tl.device_print("", arange)
            return

    h = triton.compile(triton.compiler.ASTSource(fn=kernel_basic_if_top_level, signature={"N": "i32"}, constexprs={}))

    check_template = inspect.getsource(kernel_basic_if_top_level.fn)
    run_filecheck("placeholder", h.asm["ttir"], check_template)

    @triton.jit
    def kernel_basic_while(N):
        # CHECK-LABEL: tt.func public @kernel_basic_while

        # CHECK: %arange = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
        arange = tl.arange(0, 16)
        ivar = 0
        # CHECK: %ivar_[[IV0:.+]]:2 = scf.while (%arange_[[AR0:.+]] = %arange, %ivar_[[IV1:.+]] = %ivar) : (tensor<16xi32>, i32) -> (tensor<16xi32>, i32)
        # CHECK: %[[COND:.*]] = arith.cmpi slt, %ivar_[[IV1]], %N : i32
        # CHECK: scf.condition(%[[COND]]) %arange_[[AR0]], %ivar_[[IV1]] : tensor<16xi32>, i32
        while ivar < N:
            # CHECK: ^bb0(%arange_[[AR0]]: tensor<16xi32> loc("arange"), %ivar_[[IV1]]: i32

            # CHECK: %ivar_[[IV2:.+]] = arith.addi %ivar_[[IV1]], %c1_i32 : i32
            ivar += 1
            # CHECK: %arange_[[AR1:.+]] = tt.splat %ivar_[[IV2]] : i32 -> tensor<16xi32>
            # CHECK: %arange_[[AR2:.+]] = arith.muli %arange_[[AR0]], %arange_[[AR1]] : tensor<16xi32>
            # CHECK: scf.yield %arange_[[AR2]], %ivar_[[IV2]] : tensor<16xi32>, i32
            arange *= ivar

        # CHECK: tt.print ": " {hex = false, isSigned = array<i32: 1>} : %ivar_[[IV0]]#0 : tensor<16xi32>
        tl.device_print("", arange)

    h = triton.compile(triton.compiler.ASTSource(fn=kernel_basic_while, signature={"N": "i32"}, constexprs={}))
    check_template = inspect.getsource(kernel_basic_while.fn)
    run_filecheck("placeholder", h.asm["ttir"], check_template)
