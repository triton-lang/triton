import triton
import triton.language as tl

import torch
import math
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


def test_nvidia_optimize_unsigned_multiply(tmp_path, monkeypatch):
    from triton._C.libtriton import ir, llvm
    nvidia = pytest.importorskip("triton._C.libtriton.nvidia")

    source = """
    module {
      llvm.func @low_first(%x: i32, %y: i32, %out: !llvm.ptr) -> i64 {
        %low = llvm.mul %x, %y : i32
        llvm.store %low, %out : i32, !llvm.ptr
        %x64 = llvm.zext %x : i32 to i64
        %y64 = llvm.zext %y : i32 to i64
        %wide = llvm.mul %x64, %y64 : i64
        llvm.return %wide : i64
      }
      llvm.func @high_first_constant(%x: i32, %out: !llvm.ptr) -> i32 {
        %x64 = llvm.zext %x : i32 to i64
        %c64 = llvm.mlir.constant(3528531795 : i64) : i64
        %wide = llvm.mul %c64, %x64 : i64
        llvm.store %wide, %out : i64, !llvm.ptr
        %c32 = llvm.mlir.constant(-766435501 : i32) : i32
        %low = llvm.mul %x, %c32 : i32
        llvm.return %low : i32
      }
      llvm.func @masked_alias(%x: i64, %y: i32, %out: !llvm.ptr) -> i64 {
        %x32 = llvm.trunc %x : i64 to i32
        %low = llvm.mul %x32, %y : i32
        llvm.store %low, %out : i32, !llvm.ptr
        %mask = llvm.mlir.constant(4294967295 : i64) : i64
        %masked = llvm.and %x, %mask : i64
        %y64 = llvm.zext %y : i32 to i64
        %wide = llvm.mul %y64, %masked : i64
        llvm.return %wide : i64
      }
      llvm.func @signed_extension(%x: i32, %y: i32, %out: !llvm.ptr) -> i64 {
        %low = llvm.mul %x, %y : i32
        llvm.store %low, %out : i32, !llvm.ptr
        %x64 = llvm.sext %x : i32 to i64
        %y64 = llvm.zext %y : i32 to i64
        %wide = llvm.mul %x64, %y64 : i64
        llvm.return %wide : i64
      }
      llvm.func @different_blocks(%x: i32, %y: i32, %cond: i1, %out: !llvm.ptr) -> i64 {
        %x64 = llvm.zext %x : i32 to i64
        %y64 = llvm.zext %y : i32 to i64
        %wide = llvm.mul %x64, %y64 : i64
        llvm.cond_br %cond, ^then, ^end
      ^then:
        %low = llvm.mul %x, %y : i32
        llvm.store %low, %out : i32, !llvm.ptr
        llvm.br ^end
      ^end:
        llvm.return %wide : i64
      }
    }
    """
    path = tmp_path / "multiply.mlir"
    path.write_text(source)
    context = ir.context()
    ir.load_dialects(context)
    module = ir.parse_mlir_module(str(path), context)
    llvm_module = llvm.to_module(module, llvm.context())
    monkeypatch.delenv("DISABLE_LLVM_OPT", raising=False)
    nvidia.optimize_unsigned_multiply(llvm_module)
    result = str(llvm_module)
    assert llvm.to_bitcode(result).startswith(b"BC\xc0\xde")

    for name in ["low_first", "high_first_constant", "masked_alias", "signed_extension", "different_blocks"]:
        body = result.split(f"@{name}(", 1)[1].split("\n}", 1)[0]
        if name in ["signed_extension", "different_blocks"]:
            assert "mul.wide.u32" not in body, name
            assert "mul i32" in body and "mul i64" in body, name
        else:
            assert body.count("mul.wide.u32") == 1, name
            assert "mul i32" not in body and "mul i64" not in body, name
            # Both early low uses and early high uses must follow the product.
            assert body.index("mul.wide.u32") < body.index("store "), name


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
