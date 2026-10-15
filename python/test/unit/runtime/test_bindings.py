import triton
import triton.language as tl

import torch
import math
import subprocess
import sys
import textwrap
import pytest

_BLOCK_SIZE = 16


def test_llvm_ir_to_bitcode():
    from triton._C.libtriton import llvm

    bitcode = llvm.to_bitcode("define void @kernel() { ret void }")
    assert isinstance(bitcode, bytes)
    assert bitcode.startswith(b"BC\xc0\xde")
    assert b"\x00" in bitcode


def test_llvm_ir_to_bitcode_reports_invalid_ir():
    from triton._C.libtriton import llvm

    with pytest.raises(RuntimeError, match="failed to parse LLVM IR.*expected top-level entity"):
        llvm.to_bitcode("invalid LLVM IR")


@pytest.mark.parametrize("dtype", ["i8", "i16", "i32", "i64", "f16", "f32", "f64"])
@pytest.mark.parametrize("num_elements", [2, 4])
@pytest.mark.parametrize("enable_nvptx_tti", [False, True])
@pytest.mark.parametrize("triple", ["", "nvptx64-nvidia-cuda", "amdgcn-amd-amdhsa"])
@pytest.mark.parametrize("dynamic_indices", [False, True])
def test_nvptx_tti_vector_load_scalarization(dtype, num_elements, enable_nvptx_tti, triple, dynamic_indices, tmp_path):
    from triton._C.libtriton import ir, llvm

    vector_type = f"vector<{num_elements}x{dtype}>"
    instructions = [f"%v = llvm.load %p {{alignment = 32 : i64}} : !llvm.ptr<1> -> {vector_type}"]
    if dynamic_indices:
        instructions.append(f"%mask = llvm.mlir.constant({num_elements - 1} : i32) : i32")
    for index in range(num_elements):
        if dynamic_indices:
            instructions.append(f"%i{index} = llvm.and %arg{index}, %mask : i32")
        else:
            instructions.append(f"%i{index} = llvm.mlir.constant({index} : i32) : i32")
        instructions.append(f"%v{index} = llvm.extractelement %v[%i{index} : i32] : {vector_type}")
    value = "%v0"
    opcode = "add" if dtype.startswith("i") else "fadd"
    for index in range(1, num_elements):
        instructions.append(f"%sum{index} = llvm.{opcode} {value}, %v{index} : {dtype}")
        value = f"%sum{index}"
    instructions.append(f"llvm.return {value} : {dtype}")
    arguments = ["%p: !llvm.ptr<1>"]
    if dynamic_indices:
        arguments.extend(f"%arg{index}: i32" for index in range(num_elements))
    body = "\n".join(instructions)
    source = tmp_path / "vector_load.mlir"
    source.write_text(f'''module attributes {{llvm.target_triple = "{triple}"}} {{
      llvm.func @unpack({", ".join(arguments)}) -> {dtype} {{
        {body}
      }}
    }}''')
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(source), context)
    llvm_context = llvm.context()
    llvm_module = llvm.to_module(module, llvm_context)
    llvm.optimize_module(llvm_module, llvm.OPTIMIZE_O3, disable_slp_vectorizer=True, enable_nvptx_tti=enable_nvptx_tti)
    result = str(llvm_module)
    llvm_type = {"f16": "half", "f32": "float", "f64": "double"}.get(dtype, dtype)
    # Whole-register extracts are free only in our NVIDIA TTI. Packed sub-word
    # elements, dynamic indices, and other targets retain the generic decision.
    preserve_vector = (enable_nvptx_tti and triple in ("", "nvptx64-nvidia-cuda") and not dynamic_indices
                       and dtype in ("i32", "i64", "f32", "f64"))
    if preserve_vector:
        assert f"load <{num_elements} x {llvm_type}>" in result
        assert f"load {llvm_type}," not in result
    else:
        assert f"load <{num_elements} x {llvm_type}>" not in result
        assert result.count(f"load {llvm_type},") == num_elements


def test_binding_classes_released_at_shutdown():
    script = textwrap.dedent("""
        import os
        from triton._C.libtriton import gluon_ir, ir
        from triton.experimental.gluon.language import BlockedLayout

        class ShutdownSentinel:
            def __init__(self, message):
                self.message = message

            def __del__(self, write=os.write):
                write(1, self.message)

        ir.builder._shutdown_sentinel = ShutdownSentinel(b"builder released\\n")
        context = ir.context()
        ir.load_dialects(context)
        builder = gluon_ir.GluonOpBuilder(context)
        layout = BlockedLayout([1], [32], [4], [0])._to_ir(builder)
        result = builder.to_linear_layout(layout, [128])
        type(result)._shutdown_sentinel = ShutdownSentinel(b"layout released\\n")
        del result, layout, builder, context
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert set(result.stdout.splitlines()) == {"builder released", "layout released"}, result.stderr
    assert "nanobind: leaked" not in result.stderr


@triton.jit
def add_helper(x, y):
    return x + y


@triton.jit
def add_kernel(
    in_ptr0,
    in_ptr1,
    n_elements,
    out_ptr,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    x2d = x[None, :]
    x1d = tl.reshape(x2d, [BLOCK_SIZE])
    output = add_helper(x1d, y)
    tl.store(out_ptr + offsets, output, mask=mask)


def test_module_walk(device):
    """
    Test the MLIR bindings exposed for the out-of-tree walk.
    """

    def walk_fn(op):
        name = op.get_name()
        for i in range(op.get_num_results()):
            op.get_result(i).id()
        for i in range(op.get_num_operands()):
            op.get_operand(i).id()
        for i in range(op.get_num_regions()):
            op.get_region(i).id()
        block = op.get_block()
        if block is not None:
            block.id()
            for i in range(block.get_num_arguments()):
                block.get_argument(i)
        if name == "tt.func":
            op.get_str_attr("sym_name")
        if name == "tt.call":
            op.get_flat_symbol_ref_attr("callee")
        if name == "tt.make_range":
            assert 0 == op.get_int_attr("start")
            assert _BLOCK_SIZE == op.get_int_attr("end")
        if name == "arith.constant":
            val = op.get_constant_value()
            assert isinstance(val, int)
        if name == "tt.expand_dims":
            shape = op.get_result(0).get_shape()
            assert shape == [1, _BLOCK_SIZE]
        if name == "tt.reshape":
            in_shape = op.get_operand(0).get_shape()
            out_shape = op.get_result(0).get_shape()
            assert in_shape == [1, _BLOCK_SIZE]
            assert out_shape == [_BLOCK_SIZE]

    kernel = add_kernel
    args = [
        torch.empty((32, 32), device=device),  # in_ptr0
        torch.empty((32, 32), device=device),  # in_ptr1
        1024,  # n_elements
        torch.empty((32, 32), device=device),  # out_ptr
        _BLOCK_SIZE,  # BLOCK_SIZE
    ]
    target = triton.runtime.driver.active.get_current_target()
    backend = triton.compiler.compiler.make_backend(target)
    src = triton.compiler.compiler.ASTSource(
        fn=kernel,
        signature={kernel.arg_names[i]: triton.runtime.jit.mangle_type(arg)
                   for i, arg in enumerate(args)},
        constexprs={kernel.arg_names[i]: arg
                    for i, arg in enumerate(args)
                    if not isinstance(arg, torch.Tensor)},
    )

    context = triton._C.libtriton.ir.context()
    options = backend.parse_options(dict())
    codegen_fns = dict()
    module_map = backend.get_module_map()
    triton._C.libtriton.ir.load_dialects(context)
    backend.load_dialects(context)

    ttir_module = src.make_ir(target, options, codegen_fns, module_map, context)
    ttir_module.walk(walk_fn)


def test_python_func_in_visit_call(device):

    @triton.jit
    def test_py_call_const_kernel(
        in_ptr0,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        log2e: tl.constexpr = math.log2(math.e)
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        output = x * log2e
        tl.store(out_ptr + offsets, output, mask=mask)

    x = torch.randn(4, device=device)
    out = torch.zeros_like(x)
    test_py_call_const_kernel[(4, )](x, out, 4, 4)
