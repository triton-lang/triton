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


@pytest.mark.parametrize("setup", [
    "import torch",
    "from triton.runtime.jit import mangle_type\nmangle_type(1)",
], ids=["torch", "specialization"])
def test_gluon_binding_class_released_at_shutdown(tmp_path, setup):
    # PyTorch and native argument specialization can retain JIT dependencies
    # through shutdown. Capture a Gluon builtin to exercise its builder references.
    script = textwrap.dedent("""
        import os
        from triton._C.libtriton import gluon_ir
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl
        from triton.experimental.gluon.language.nvidia.blackwell import (
            allocate_tensor_memory, TensorMemoryLayout,
        )

        class ShutdownSentinel:
            def __del__(self, write=os.write):
                write(1, b"builder released\\n")

        gluon_ir.GluonOpBuilder._shutdown_sentinel = ShutdownSentinel()

        def make_allocator(allocate):
            @gluon.jit
            def allocator():
                return allocate(gl.float32, [128, 128], TensorMemoryLayout([128, 128], col_stride=1))
            return allocator

        allocator = make_allocator(allocate_tensor_memory)
    """)
    script_path = tmp_path / "gluon_binding_shutdown.py"
    script_path.write_text(setup + "\n" + script)
    result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True, check=True, timeout=60)
    assert result.stdout.splitlines() == ["builder released"], result.stderr
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
