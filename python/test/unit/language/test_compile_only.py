import pytest
import re

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.errors import CompileTimeAssertionFailure
from triton.runtime.driver import driver


@triton.jit
def topk_kernel(K: tl.constexpr):
    x = tl.arange(0, 8)
    tl.topk(x, K)


@pytest.mark.parametrize(
    "k, error",
    [
        (0, "topk: k must be greater than 0"),
        (-1, "topk: k must be greater than 0"),
        (3, "topk: k must be a power of two"),
        (16, "topk: k must not exceed the size of the selected dimension"),
    ],
)
def test_topk_invalid_k(k, error):
    src = ASTSource(fn=topk_kernel, signature={"K": "constexpr"}, constexprs={"K": k})
    with pytest.raises(triton.CompilationError) as exc_info:
        triton.compile(src, target=GPUTarget("cuda", 90, 32))
    cause = exc_info.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert isinstance(cause, CompileTimeAssertionFailure)
    assert cause.error_message == error


@pytest.mark.parametrize("k", [1, 8])
def test_topk_valid_k(k):
    src = ASTSource(fn=topk_kernel, signature={"K": "constexpr"}, constexprs={"K": k})
    triton.compile(src, target=GPUTarget("cuda", 90, 32))


def test_compile_only_sort_keeps_comparisons_boolean() -> None:

    @triton.jit
    def sort_kernel(values, result):
        offsets = tl.arange(0, 128)
        loaded = tl.load(values + offsets)
        sorted_values = tl.sort(loaded, descending=False)
        tl.store(result + offsets, sorted_values)

    source = ASTSource(fn=sort_kernel, signature={"values": "*i32", "result": "*i32"})
    compiled = triton.compile(source, target=GPUTarget("cuda", 100, 32))
    assert "arith.extui" not in compiled.asm["ttgir"]


def _make_low_precision_test_ir(source, target=None):
    from triton._C.libtriton import ir
    from triton.compiler.compiler import make_backend

    target = target or GPUTarget("cuda", 100, 32)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 1, "ptx_version": 94})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    return module, backend, options


def _lower_low_precision_storage_types(module, context, target=None):
    from triton._C.libtriton import ir, passes

    pm = ir.pass_manager(context)
    if target is None:
        passes.ttir.add_lower_storage_types(pm)
    else:
        passes.ttir.add_lower_storage_types(pm, target)
    pm.run(module, "lower_storage_types_test")


def _low_precision_mixed_scale_ir(native_scale, module_target=None):
    scale_dtype = "f8E8M0FNU" if native_scale else "i8"
    attributes = f' attributes {{ttg.target = "{module_target}"}}' if module_target else ""
    return f"""module{attributes} {{
  tt.func public @mixed_scales(%a: tensor<16x32xf8E4M3FN>, %b: tensor<16x8xi8>,
                              %a_scale: tensor<16x1x{scale_dtype}>, %b_scale: tensor<8x1xf8E4M3FN>,
                              %c: tensor<16x8xf32>) -> tensor<16x8xf32> {{
    %d = tt.dot_scaled %a scale %a_scale, %b scale %b_scale, %c lhs = e4m3 rhs = e2m1 {{fastMath = false}}
      : tensor<16x32xf8E4M3FN>, tensor<16x1x{scale_dtype}> * tensor<16x8xi8>, tensor<8x1xf8E4M3FN> -> tensor<16x8xf32>
    tt.return %d : tensor<16x8xf32>
  }}
}}"""


@pytest.mark.parametrize("target, module_target, native_scale, allowed", [
    ("cuda:100", None, True, False),
    ("hip:gfx1250", None, True, True),
    (None, None, True, False),
    (None, "cuda:100", True, False),
    (None, "hip:gfx1250", True, True),
    ("cuda:100", "hip:gfx1250", True, False),
    ("hip:gfx1250", "cuda:100", True, True),
    ("cuda:100", None, False, True),
    ("hip:gfx1250", None, False, True),
])
def test_low_precision_mixed_scales_storage_lowering_target(tmp_path, capfd, target, module_target, native_scale,
                                                            allowed):
    from triton._C.libtriton import ir

    context = ir.context()
    ir.load_dialects(context)
    path = tmp_path / "mixed_scales.ttir"
    path.write_text(_low_precision_mixed_scale_ir(native_scale, module_target))
    module = ir.parse_mlir_module(str(path), context)
    before = module.str_nodebug()
    if not allowed:
        with pytest.raises(RuntimeError):
            _lower_low_precision_storage_types(module, context, target)
        assert "native E8M0" in capfd.readouterr().err
        assert module.str_nodebug() == before
    else:
        _lower_low_precision_storage_types(module, context, target)
        expected = before.replace("f8E8M0FNU", "i8") if native_scale else before
        assert module.str_nodebug() == expected


@pytest.mark.parametrize("entrypoint", ["source", "override"])
@pytest.mark.parametrize("target", [GPUTarget("cuda", 100, 32), GPUTarget("hip", "gfx1250", 32)])
@pytest.mark.parametrize("native_scale", [False, True])
def test_low_precision_textual_ir_mixed_scales_target(tmp_path, capfd, entrypoint, target, native_scale):
    from triton._C.libtriton import ir
    from triton.compiler.compiler import IRSource, make_backend, parse

    context = ir.context()
    ir.load_dialects(context)
    path = tmp_path / "mixed_scales.ttir"
    path.write_text(_low_precision_mixed_scale_ir(native_scale))
    before = ir.parse_mlir_module(str(path), context).str_nodebug()

    def import_ir():
        if entrypoint == "source":
            source = IRSource(str(path), context, make_backend(target))
            return source.make_ir(target, None, None, None, context)
        return parse(str(path), "ttir", context, target)

    if native_scale and target.backend == "cuda":
        with pytest.raises(RuntimeError):
            import_ir()
        assert "native E8M0" in capfd.readouterr().err
    else:
        module = import_ir()
        expected = before.replace("f8E8M0FNU", "i8") if native_scale else before
        assert module.str_nodebug() == expected


@pytest.mark.parametrize("extension", ["ttir", "ttgir", "glir"])
@pytest.mark.parametrize("entrypoint", ["source", "override"])
def test_low_precision_textual_ir_rejects_numeric_e8m0(tmp_path, capfd, extension, entrypoint):
    from triton._C.libtriton import ir
    from triton.compiler.compiler import IRSource, make_backend, parse

    context = ir.context()
    ir.load_dialects(context)
    path = tmp_path / f"unsupported.{extension}"
    path.write_text("""module {
  tt.func public @unsupported(%value: f8E8M0FNU) {
    %result = arith.addf %value, %value : f8E8M0FNU
    tt.return
  }
}""")
    with pytest.raises(RuntimeError):
        if entrypoint == "source":
            target = GPUTarget("cuda", 100, 32)
            source = IRSource(str(path), context, make_backend(target))
            source.make_ir(target, None, None, None, context)
        else:
            parse(str(path), extension, context)
    assert "does not support native low-precision storage types" in capfd.readouterr().err


@pytest.mark.parametrize("literal, bits", [("1.0", 127), ("0x00", 0), ("0xFF", -1)])
@pytest.mark.parametrize("tensor", [False, True])
def test_low_precision_e8m0_constants_lower_encoded_bits(tmp_path, literal, bits, tensor):
    from triton._C.libtriton import ir

    context = ir.context()
    ir.load_dialects(context)
    dtype = "tensor<4xf8E8M0FNU>" if tensor else "f8E8M0FNU"
    value = f"dense<{literal}>" if tensor else literal
    path = tmp_path / "constant.mlir"
    path.write_text(f"""module {{
  tt.func @constant() -> {dtype} {{
    %value = arith.constant {value} : {dtype}
    tt.return %value : {dtype}
  }}
}}""")
    module = ir.parse_mlir_module(str(path), context)
    _lower_low_precision_storage_types(module, context)
    expected_value = f"dense<{bits}>" if tensor else str(bits)
    expected_dtype = "tensor<4xi8>" if tensor else "i8"
    assert f"arith.constant {expected_value} : {expected_dtype}" in module.str_nodebug()


def test_low_precision_e8m0_dense_constants_preserve_all_encodings(tmp_path):
    from triton._C.libtriton import ir

    context = ir.context()
    ir.load_dialects(context)
    values = ", ".join(f"0x{value:02X}" for value in range(256))
    path = tmp_path / "all_encodings.mlir"
    path.write_text(f"""module {{
  tt.func @all_encodings() -> tensor<256xf8E8M0FNU> {{
    %value = arith.constant dense<[{values}]> : tensor<256xf8E8M0FNU>
    tt.return %value : tensor<256xf8E8M0FNU>
  }}
}}""")
    module = ir.parse_mlir_module(str(path), context)
    _lower_low_precision_storage_types(module, context)
    expected = ", ".join(str(value if value < 128 else value - 256) for value in range(256))
    expected_hex = "".join(f"{value:02x}" for value in range(256))
    result = module.str_nodebug().lower()
    assert (f"arith.constant dense<[{expected}]> : tensor<256xi8>" in result
            or f' arith.constant dense<"0x{expected_hex}"> : tensor<256xi8>' in result)


@pytest.mark.parametrize("dtype", ["i8", "!tt.fp4e2m1x2", "f8E8M0FNU"])
def test_low_precision_storage_lowering_is_explicit(tmp_path, dtype):
    from triton._C.libtriton import ir

    context = ir.context()
    ir.load_dialects(context)
    path = tmp_path / "copy.mlir"
    path.write_text(f"""module {{
  tt.func @copy(%source: !tt.ptr<{dtype}>, %destination: !tt.ptr<{dtype}>) {{
    %value = tt.load %source : !tt.ptr<{dtype}>
    tt.store %destination, %value : !tt.ptr<{dtype}>
    tt.return
  }}
}}""")
    module = ir.parse_mlir_module(str(path), context)
    before = module.str_nodebug()
    assert f"!tt.ptr<{dtype}>" in before
    _lower_low_precision_storage_types(module, context)
    after = module.str_nodebug()
    assert "!tt.ptr<i8>" in after
    if dtype == "i8":
        assert before == after
    else:
        assert dtype not in after


@pytest.mark.parametrize("dtype, operation", [
    ("f8E8M0FNU", "%result = arith.addf %arg, %arg : f8E8M0FNU"),
    ("f8E8M0FNU", '"tt.print"(%arg) {prefix = "", hex = false, isSigned = array<i32: 0>} : (f8E8M0FNU) -> ()'),
    ("!tt.fp4e2m1x2", '"tt.print"(%arg) {prefix = "", hex = false, isSigned = array<i32: 0>} : (!tt.fp4e2m1x2) -> ()'),
])
def test_low_precision_storage_lowering_rejects_unsupported_operations(tmp_path, capfd, dtype, operation):
    from triton._C.libtriton import ir

    context = ir.context()
    ir.load_dialects(context)
    path = tmp_path / "unsupported.mlir"
    path.write_text(f"module {{ tt.func @unsupported(%arg: {dtype}) {{ {operation}\n tt.return }} }}")
    module = ir.parse_mlir_module(str(path), context)
    before = module.str_nodebug()
    with pytest.raises(RuntimeError):
        _lower_low_precision_storage_types(module, context)
    assert "does not support native low-precision storage types" in capfd.readouterr().err
    assert module.str_nodebug() == before


@pytest.mark.parametrize("dtype", ["!tt.fp4e2m1x2", "f8E8M0FNU"])
def test_low_precision_ir_rejects_implicit_byte_store(tmp_path, capfd, dtype):
    from triton._C.libtriton import ir

    context = ir.context()
    ir.load_dialects(context)
    path = tmp_path / "invalid_store.mlir"
    path.write_text(f"""module {{
  tt.func @invalid_store(%destination: !tt.ptr<{dtype}>, %value: i8) {{
    "tt.store"(%destination, %value) : (!tt.ptr<{dtype}>, i8) -> ()
    tt.return
  }}
}}""")
    with pytest.raises(RuntimeError, match="Parse MLIR file failed"):
        ir.parse_mlir_module(str(path), context)
    assert "value type matches ptr type" in capfd.readouterr().err


@pytest.mark.parametrize("dtype_name, canonical, ir_dtype", [("float4e2m1x2", "fp4e2m1x2", "!tt.fp4e2m1x2"),
                                                             ("float8e8m0fnu", "fp8e8m0fnu", "f8E8M0FNU")])
def test_low_precision_copy_preserves_native_ir_type(dtype_name, canonical, ir_dtype):

    @triton.jit
    def identity(value, DTYPE: tl.constexpr):
        tl.static_assert(value.dtype == DTYPE)
        return value

    @triton.jit
    def copy_kernel(source, destination, DTYPE: tl.constexpr):
        offsets = tl.arange(0, 128)
        value = tl.load(source + offsets)
        tl.static_assert(value.dtype == DTYPE)
        tl.static_assert(value.dtype != tl.uint8)
        tl.static_assert(value.dtype.primitive_bitwidth == 8)
        value = identity(value, DTYPE)
        raw = value.to(tl.uint8, bitcast=True)
        tl.static_assert(raw.dtype == tl.uint8)
        value = raw.to(DTYPE, bitcast=True)
        tl.store(destination + offsets, value)

    source = ASTSource(copy_kernel, signature={"source": f"*{canonical}", "destination": f"*{canonical}"},
                       constexprs={"DTYPE": getattr(tl, dtype_name)})
    module, backend, options = _make_low_precision_test_ir(source)
    frontend_ir = str(module)
    assert f"!tt.ptr<{ir_dtype}>" in frontend_ir
    assert f"tensor<128x{ir_dtype}>" in frontend_ir
    assert "tt.bitcast" in frontend_ir
    metadata = {"target": backend.target, **vars(options)}
    stages = {}
    backend.add_stages(stages, options, source.language)
    for kind, stage in stages.items():
        module = stage(module, metadata)
        if kind == "ptx":
            assert "ld.global" in module
            assert "st.global" in module
            break
    else:
        pytest.fail("PTX stage not found")


@pytest.mark.parametrize("canonical", ["fp4e2m1x2", "fp8e8m0fnu"])
@pytest.mark.parametrize(
    "operation",
    ["add", "multiply", "compare", "unary_minus", "invert", "cast_uint8", "cast_float32", "implicit_store"])
def test_low_precision_numeric_operations_are_rejected(canonical, operation):

    @triton.jit
    def kernel(source, destination, OP: tl.constexpr):
        offsets = tl.arange(0, 128)
        value = tl.load(source + offsets)
        if OP == "add":
            value = value + value
        elif OP == "multiply":
            value = value * value
        elif OP == "compare":
            value = value == tl.load(source + offsets)
        elif OP == "unary_minus":
            value = -value
        elif OP == "invert":
            value = ~value
        elif OP == "cast_uint8":
            value = value.to(tl.uint8)
        elif OP == "cast_float32":
            value = value.to(tl.float32)
        elif OP == "implicit_store":
            value = tl.full((128, ), 1, tl.uint8)
        tl.store(destination + offsets, value)

    source = ASTSource(kernel, signature={"source": f"*{canonical}", "destination": f"*{canonical}"},
                       constexprs={"OP": operation})
    with pytest.raises(triton.CompilationError) as exc_info:
        _make_low_precision_test_ir(source)
    cause = exc_info.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert isinstance(cause, (TypeError, ValueError))
    assert canonical in str(cause)


@triton.jit
def _low_precision_scaled_dot(lhs_ptr, rhs_ptr, scale_ptr, output, LHS_FORMAT: tl.constexpr, RHS_FORMAT: tl.constexpr):
    rows = tl.arange(0, 32)[:, None]
    columns = tl.arange(0, 32)[None, :]
    lhs = tl.load(lhs_ptr + rows * 32 + columns)
    rhs = tl.load(rhs_ptr + rows * 32 + columns)
    scales = tl.load(scale_ptr + rows * 2 + tl.arange(0, 2)[None, :])
    result = tl.dot_scaled(lhs, scales, LHS_FORMAT, rhs, scales, RHS_FORMAT)
    tl.store(output + rows * 32 + columns, result)


def _make_low_precision_scaled_dot_ir(lhs_dtype, rhs_dtype, scale_dtype, lhs_format="e2m1", rhs_format="e2m1",
                                      lower=True):
    source = ASTSource(
        _low_precision_scaled_dot, signature={
            "lhs_ptr": f"*{lhs_dtype}", "rhs_ptr": f"*{rhs_dtype}", "scale_ptr": f"*{scale_dtype}", "output": "*fp32"
        }, constexprs={"LHS_FORMAT": lhs_format, "RHS_FORMAT": rhs_format})
    module, backend, options = _make_low_precision_test_ir(source)
    if not lower:
        return module
    metadata = {"target": backend.target, **vars(options)}
    stages = {}
    backend.add_stages(stages, options, source.language)
    return stages["ttir"](module, metadata)


@pytest.mark.parametrize("operand_dtype", ["u8", "fp4e2m1x2"])
@pytest.mark.parametrize("scale_dtype", ["u8", "fp8e8m0fnu"])
def test_low_precision_scaled_dot_preserves_native_ir_type(operand_dtype, scale_dtype):
    module = _make_low_precision_scaled_dot_ir(operand_dtype, operand_dtype, scale_dtype, lower=False)
    dot = next(line for line in str(module).splitlines() if "tt.dot_scaled" in line)
    operand_ir_dtype = "!tt.fp4e2m1x2" if operand_dtype == "fp4e2m1x2" else "i8"
    scale_ir_dtype = "f8E8M0FNU" if scale_dtype == "fp8e8m0fnu" else "i8"
    assert f"tensor<32x32x{operand_ir_dtype}>" in dot
    assert f"tensor<32x2x{scale_ir_dtype}>" in dot


@pytest.mark.parametrize("operand_dtype", ["u8", "fp4e2m1x2"])
@pytest.mark.parametrize("scale_dtype", ["u8", "i8", "fp8e8m0fnu"])
def test_low_precision_scaled_dot_preserves_legacy_lowering(operand_dtype, scale_dtype):
    native = _make_low_precision_scaled_dot_ir(operand_dtype, operand_dtype, scale_dtype)
    legacy = _make_low_precision_scaled_dot_ir("u8", "u8", "u8")
    assert "tt.dot_scaled" in str(native)
    assert str(native) == str(legacy)


@pytest.mark.parametrize("operand_dtype, scale_dtype, operand_format, rejected_dtype", [
    ("fp4e2m1x2", "fp8e8m0fnu", "e4m3", "fp4e2m1x2"),
    ("fp4e2m1x2", "fp8e8m0fnu", "e5m2", "fp4e2m1x2"),
    ("fp4e2m1x2", "fp8e8m0fnu", "bf16", "fp4e2m1x2"),
    ("fp4e2m1x2", "fp8e8m0fnu", "fp16", "fp4e2m1x2"),
    ("fp8e8m0fnu", "u8", "e2m1", "fp8e8m0fnu"),
    ("u8", "fp4e2m1x2", "e2m1", "fp4e2m1x2"),
])
@pytest.mark.parametrize("rhs", [False, True])
def test_low_precision_scaled_dot_rejects_incompatible_dtype(operand_dtype, scale_dtype, operand_format, rejected_dtype,
                                                             rhs):
    lhs_dtype, rhs_dtype = ("u8", operand_dtype) if rhs else (operand_dtype, "u8")
    lhs_format, rhs_format = ("e2m1", operand_format) if rhs else (operand_format, "e2m1")
    with pytest.raises(triton.CompilationError) as exc_info:
        _make_low_precision_scaled_dot_ir(lhs_dtype, rhs_dtype, scale_dtype, lhs_format, rhs_format)
    cause = exc_info.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert isinstance(cause, ValueError)
    assert rejected_dtype in str(cause)


@pytest.mark.parametrize("dtype_name, canonical", [("float4e2m1x2", "fp4e2m1x2"), ("float8e8m0fnu", "fp8e8m0fnu")])
@pytest.mark.parametrize("padding", ["zero", "nan"])
def test_low_precision_descriptor_padding(dtype_name, canonical, padding):

    @triton.jit
    def kernel(source, output, DTYPE: tl.constexpr, PADDING: tl.constexpr):
        descriptor = tl.make_tensor_descriptor(source, [16, 16], [16, 1], [16, 16], padding_option=PADDING)
        value = descriptor.load([0, 0])
        tl.static_assert(value.dtype == DTYPE)
        offsets = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
        tl.store(output + offsets, value)

    source = ASTSource(kernel, signature={"source": f"*{canonical}", "output": f"*{canonical}"},
                       constexprs={"DTYPE": getattr(tl, dtype_name), "PADDING": padding})
    if padding == "zero":
        module = _make_low_precision_test_ir(source)[0]
        assert "tt.make_tensor_descriptor" in str(module)
    else:
        with pytest.raises(triton.CompilationError) as exc_info:
            _make_low_precision_test_ir(source)
        cause = exc_info.value
        while cause.__cause__ is not None:
            cause = cause.__cause__
        assert isinstance(cause, ValueError)
        assert "Padding option `nan` is not supported for storage dtype" in str(cause)


@pytest.mark.parametrize("source_dtype, destination_dtype", [
    ("fp4e2m1x2", "u8"),
    ("u8", "fp4e2m1x2"),
    ("fp8e8m0fnu", "u8"),
    ("u8", "fp8e8m0fnu"),
    ("fp4e2m1x2", "fp8e8m0fnu"),
    ("fp8e8m0fnu", "fp4e2m1x2"),
])
def test_low_precision_descriptor_scatter_rejects_implicit_reinterpretation(source_dtype, destination_dtype):

    @triton.jit
    def kernel(source, output):
        rows = tl.arange(0, 8)
        offsets = rows[:, None] * 32 + tl.arange(0, 32)[None, :]
        value = tl.load(source + offsets)
        descriptor = tl.make_tensor_descriptor(output, [8, 32], [32, 1], [1, 32])
        descriptor.scatter(value, rows, 0)

    source = ASTSource(kernel, signature={"source": f"*{source_dtype}", "output": f"*{destination_dtype}"})
    with pytest.raises(triton.CompilationError) as exc_info:
        _make_low_precision_test_ir(source)
    cause = exc_info.value
    while cause.__cause__ is not None:
        cause = cause.__cause__
    assert isinstance(cause, TypeError)
    assert "must match" in str(cause)


@pytest.mark.parametrize("source_dtype, destination_dtype", [
    ("fp4e2m1x2", "u8"),
    ("u8", "fp4e2m1x2"),
    ("fp8e8m0fnu", "u8"),
    ("u8", "fp8e8m0fnu"),
    ("fp4e2m1x2", "fp4e2m1x2"),
    ("fp8e8m0fnu", "fp8e8m0fnu"),
])
@pytest.mark.parametrize("operation", ["reduce", "scan"])
def test_low_precision_combine_preserves_dtype(source_dtype, destination_dtype, operation):
    result_dtype = tl.str_to_ty(destination_dtype, None)

    @triton.jit
    def combine(lhs, rhs):
        return lhs.to(result_dtype, bitcast=True)

    @triton.jit
    def kernel(source, output, OP: tl.constexpr):
        offsets = tl.arange(0, 128)
        value = tl.load(source + offsets)
        if OP == "reduce":
            result = tl.reduce(value, 0, combine)
            tl.static_assert(result.dtype == source.dtype.element_ty)
            tl.store(output, result)
        else:
            result = tl.associative_scan(value, 0, combine)
            tl.static_assert(result.dtype == source.dtype.element_ty)
            tl.store(output + offsets, result)

    source = ASTSource(kernel, signature={"source": f"*{source_dtype}", "output": f"*{source_dtype}"},
                       constexprs={"OP": operation})
    if source_dtype == destination_dtype:
        _make_low_precision_test_ir(source)
    else:
        with pytest.raises(triton.CompilationError) as exc_info:
            _make_low_precision_test_ir(source)
        cause = exc_info.value
        while cause.__cause__ is not None:
            cause = cause.__cause__
        assert isinstance(cause, TypeError)
        assert "must match" in str(cause)


def test_compile_only_sm100() -> None:

    @triton.jit
    def kernel_add(a, b, c):
        idx = tl.arange(0, 32)
        tl.store(c + idx, tl.load(a + idx) + tl.load(b + idx))

    k = triton.compile(
        triton.compiler.ASTSource(fn=kernel_add, signature={"a": "*fp32", "b": "*fp32", "c": "*fp32"}, constexprs={}),
        target=GPUTarget("cuda", 100, 32))
    ptx = k.asm["ptx"]
    assert ".target sm_100a" in ptx
    assert ".address_size 64" in ptx
    assert k.asm["cubin"] != b""


@pytest.mark.parametrize("seed_type, dtype", [("u32", tl.uint32), ("u64", tl.uint32), ("u64", tl.uint64)])
def test_umulhi_truncated_input(seed_type, dtype):
    from triton._C.libtriton import ir
    from triton.compiler.compiler import make_backend

    @triton.jit
    def kernel(seed, out, DTYPE: tl.constexpr):
        x = seed.to(DTYPE)
        tl.store(out, tl.umulhi(x, x))

    target = GPUTarget("cuda", 100, 32)
    backend = make_backend(target)
    options = backend.parse_options({"num_warps": 1, "ptx_version": 94})
    bits = dtype.primitive_bitwidth
    source = ASTSource(kernel, signature={"seed": seed_type, "out": f"*u{bits}", "DTYPE": "constexpr"},
                       constexprs={"DTYPE": dtype})
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = source.make_ir(target, options, backend.get_codegen_implementation(options), backend.get_module_map(),
                            context)
    metadata = {"target": target, **vars(options)}
    stages = {}
    backend.add_stages(stages, options, source.language)
    # Stop at PTX so this regression test needs neither a GPU nor ptxas.
    for kind, stage in stages.items():
        module = stage(module, metadata)
        if kind == "llir":
            intrinsic = "ui" if bits == 32 else "ull"
            assert f"@llvm.nvvm.mulhi.{intrinsic}" in str(module)
        if kind == "ptx":
            assert f"mul.hi.u{bits}" in module
            assert "mul.lo." not in module
            break
    else:
        pytest.fail("PTX stage not found")


@pytest.mark.parametrize("instrumentation_mode", ["", "fpsan"])
def test_compile_only_nvptx_fabs_preserves_nan_payload(instrumentation_mode):
    from triton._C.libtriton import llvm
    from triton.backends.nvidia.compiler import CUDABackend

    llvm.init_targets()
    backend = CUDABackend(GPUTarget("cuda", 90, 32))
    options = backend.parse_options({"ptx_version": 93, "instrumentation_mode": instrumentation_mode})
    source = """
target triple = "nvptx64-nvidia-cuda"
declare <4 x float> @llvm.fabs.v4f32(<4 x float>)
define ptx_kernel void @abs_kernel(ptr addrspace(1) %input, ptr addrspace(1) %output) {
  %value = load <4 x float>, ptr addrspace(1) %input, align 16
  %absolute = call <4 x float> @llvm.fabs.v4f32(<4 x float> %value)
  store <4 x float> %absolute, ptr addrspace(1) %output, align 16
  ret void
}
"""
    ptx = backend.make_ptx(source, {}, options, 90)
    if instrumentation_mode == "fpsan":
        assert "abs.f32" not in ptx
        masks = re.findall(r"\band\.b32\s+[^;]+,\s*(0x[0-9a-fA-F]+|\d+)\s*;", ptx)
        assert len(masks) == 4
        assert all(int(mask, 0) == 0x7FFFFFFF for mask in masks)
    else:
        assert len(re.findall(r"\babs\.f32\b", ptx)) == 4
        assert "and.b32" not in ptx


@pytest.mark.parametrize("element_type", ["f32", "f16", "bf16"])
def test_compile_only_packed_arith_chains(element_type, tmp_path) -> None:
    packed_type = f"{element_type}x2"
    tensor_type = f"tensor<256x{element_type}, #blocked>"
    pointer_type = f"!tt.ptr<{element_type}>"
    operations = [
        ("add", "%av, %bv", 2),
        ("sub", "%add, %bv", 2),
        ("mul", "%sub, %bv", 2),
        ("fma", "%mul, %bv, %cv", 3),
    ]
    if element_type != "f32":
        operations.extend([("min", "%fma, %bv", 2), ("max", "%min, %cv", 2)])

    instructions = []
    for name, operands, operand_count in operations:
        tensor_types = ", ".join([tensor_type] * operand_count)
        instructions.append(f"    %{name} = ttng.packed_arith {name} "
                            f"{operands} : ({tensor_types}) -> {tensor_type}")
    packed_operations = "\n".join(instructions)
    result = operations[-1][0]
    src = f"""
#blocked = #ttg.blocked<{{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32}} {{
  tt.func public @packed_chain_{element_type}(%a: {pointer_type} {{tt.divisibility = 16 : i32}}, %b: {pointer_type} {{tt.divisibility = 16 : i32}}, %c: {pointer_type} {{tt.divisibility = 16 : i32}}, %out: {pointer_type} {{tt.divisibility = 16 : i32}}) attributes {{noinline = false}} {{
    %offsets = tt.make_range {{start = 0 : i32, end = 256 : i32}} : tensor<256xi32, #blocked>
    %ap = tt.splat %a : {pointer_type} -> tensor<256x{pointer_type}, #blocked>
    %bp = tt.splat %b : {pointer_type} -> tensor<256x{pointer_type}, #blocked>
    %cp = tt.splat %c : {pointer_type} -> tensor<256x{pointer_type}, #blocked>
    %op = tt.splat %out : {pointer_type} -> tensor<256x{pointer_type}, #blocked>
    %aa = tt.addptr %ap, %offsets : tensor<256x{pointer_type}, #blocked>, tensor<256xi32, #blocked>
    %ba = tt.addptr %bp, %offsets : tensor<256x{pointer_type}, #blocked>, tensor<256xi32, #blocked>
    %ca = tt.addptr %cp, %offsets : tensor<256x{pointer_type}, #blocked>, tensor<256xi32, #blocked>
    %oa = tt.addptr %op, %offsets : tensor<256x{pointer_type}, #blocked>, tensor<256xi32, #blocked>
    %av = tt.load %aa : tensor<256x{pointer_type}, #blocked>
    %bv = tt.load %ba : tensor<256x{pointer_type}, #blocked>
    %cv = tt.load %ca : tensor<256x{pointer_type}, #blocked>
{packed_operations}
    tt.store %oa, %{result} : tensor<256x{pointer_type}, #blocked>
    tt.return
  }}
}}
"""
    source_file = tmp_path / f"packed_chain_{element_type}.ttgir"
    source_file.write_text(src)
    ptx = triton.compile(str(source_file), target=GPUTarget("cuda", 100, 32)).asm["ptx"]
    pattern = re.compile(
        rf"^\s*(?P<operation>add|sub|mul|fma|min|max)\.(?:rn\.)?{packed_type}\s+"
        r"(?P<output>%\w+),\s*(?P<input>%\w+)", re.MULTILINE)
    emitted_operations = list(pattern.finditer(ptx))
    assert [match.group("operation") for match in emitted_operations] == [name for name, _, _ in operations]
    for previous, current in zip(emitted_operations, emitted_operations[1:]):
        assert current.group("input") == previous.group("output")
    assert "prmt.b32" not in ptx


def test_compile_only_ws_cluster_barrier_shared_memory(tmp_path) -> None:
    src = """
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ws_cluster_barrier() {
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<5x1xi8, #shared, #ttg.shared_memory, mutable>
    ttg.warp_specialize()
    default {
      ttng.cluster_barrier
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
"""
    temp_file = tmp_path / "ws_cluster_barrier.ttgir"
    temp_file.write_text(src)
    k = triton.compile(str(temp_file), target=GPUTarget("cuda", 90, 32))
    ptx = k.asm["ptx"]
    assert "mbarrier.arrive.release.cluster.shared::cluster.b64" in ptx
    assert "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64" in ptx
    assert "mapa" not in ptx
    assert k.metadata.shared == 40
    assert k.asm["cubin"] != b""


@pytest.mark.parametrize("instrumentation_mode", ["", "consan", "gsan", "iisan", "fpsan", "gsan,consan"])
def test_maxnreg_instrumentation_mode(instrumentation_mode, monkeypatch, fresh_triton_cache):

    class UnavailableDriver:

        def __getattr__(self, name):
            raise AssertionError(f"Compilation accessed the driver: {name}")

    monkeypatch.setattr(driver, "_active", UnavailableDriver())

    @triton.jit
    def kernel(out):
        tl.store(out, 1)

    src = ASTSource(fn=kernel, signature={"out": "*i32"})
    compiled = triton.compile(src, target=GPUTarget("cuda", 90, 32),
                              options={"maxnreg": 42, "max_occupancy": 4, "instrumentation_mode": instrumentation_mode})
    assert compiled.module is None
    assert compiled.metadata.max_occupancy == (1 if "gsan" in instrumentation_mode else 4)
    if instrumentation_mode:
        assert compiled.metadata.maxnreg is None
        assert ".maxnreg" not in compiled.asm["ptx"]
    else:
        assert compiled.metadata.maxnreg == 42
        assert ".maxnreg 42" in compiled.asm["ptx"]


@pytest.mark.parametrize("max_occupancy", [None, 1, 4])
def test_max_occupancy_compile_only(max_occupancy, monkeypatch, fresh_triton_cache):

    class UnavailableDriver:

        def __getattr__(self, name):
            raise AssertionError(f"Compilation accessed the driver: {name}")

    monkeypatch.setattr(driver, "_active", UnavailableDriver())

    @triton.jit
    def kernel(out):
        tl.store(out, 1)

    src = ASTSource(fn=kernel, signature={"out": "*i32"})
    compiled = triton.compile(src, target=GPUTarget("cuda", 90, 32), options={"max_occupancy": max_occupancy})
    assert compiled.module is None
    assert compiled.metadata.max_occupancy == max_occupancy
    assert compiled.metadata.shared == 0


@pytest.mark.parametrize("max_occupancy", [0, -1, 1.5, True, "2"])
def test_max_occupancy_invalid(max_occupancy):
    from triton.backends.nvidia.compiler import CUDABackend

    backend = CUDABackend(GPUTarget("cuda", 90, 32))
    with pytest.raises(ValueError, match="max_occupancy must be a positive integer or None"):
        backend.parse_options({"max_occupancy": max_occupancy})


def test_compile_only_expect_zero() -> None:

    @triton.jit
    def expect_zero_kernel(x_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)
        y = tl.expect_zero(x, offsets < 8)
        tl.store(out_ptr + offsets, y)

    src = triton.compiler.ASTSource(
        fn=expect_zero_kernel,
        signature={"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK_SIZE": "constexpr"},
        constexprs={"BLOCK_SIZE": 16},
    )
    target = GPUTarget("cuda", 100, 32)

    regular = triton.compile(src, target=target)
    assert "arith.select" not in regular.asm["ttir"]
    assert "tt.assert" not in regular.asm["ttir"]

    debug = triton.compile(src, target=target, options={"debug": True})
    assert "arith.select" not in debug.asm["ttir"]
    assert "tt.assert" in debug.asm["ttir"]

    fpsan = triton.compile(src, target=target, options={"instrumentation_mode": "fpsan"})
    assert "arith.select" in fpsan.asm["ttir"]
    assert "tt.assert" not in fpsan.asm["ttir"]


def test_compile_only_dot() -> None:

    @triton.jit
    def simple_dot(a_base, b_base, out):
        SIZE: tl.constexpr = 64
        a_ptr = a_base + tl.arange(0, SIZE)[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
        b_ptr = b_base + tl.arange(0, SIZE)[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        c = tl.dot(a, b)
        out_ptr = out + tl.arange(0, SIZE)[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
        tl.store(out_ptr, c)

    k = triton.compile(
        triton.compiler.ASTSource(fn=simple_dot, signature={"a_base": "*fp16", "b_base": "*fp16", "out": "*fp16"},
                                  constexprs={}), target=GPUTarget("cuda", 100, 32))
    ttgir = k.asm["ttgir"]
    pattern = (r"%(?P<A>\w+) = tt\.load"
               r"(.|\n)*?"
               r"%(?P<A_SHMEM>\w+) = ttg\.local_alloc %(?P=A)"
               r"(.|\n)*?"
               r"%(?P<B>\w+) = tt\.load"
               r"(.|\n)*?"
               r"%(?P<B_SHMEM>\w+) = ttg\.local_alloc %(?P=B)"
               r"(.|\n)*?"
               r"%(?P<TMEM_BASE>\w+) = ttng\.tmem_alloc"
               r"(.|\n)*?"
               r"ttng\.tc_gen5_mma %(?P=A_SHMEM), %(?P=B_SHMEM), %(?P=TMEM_BASE)"
               r"(.|\n)*?"
               r"ttng\.tmem_load %(?P=TMEM_BASE)")

    assert re.search(pattern, str(ttgir)), "The TTGIR does not match the expected pattern."

    ptx = k.asm["ptx"]
    pattern = (r"mov\.b32 	%r(?P<G>\d+), global_smem;"
               r"(.|\n)*"
               r"tcgen05\.alloc\.cta_group::1\.sync\.aligned\.shared::cta\.b32 \[%r(?P=G)], 64"
               r"(.|\n)*"
               r"tcgen05\.relinquish_alloc_permit\.cta_group::1\.sync\.aligned"
               r"(.|\n)*"
               r"tcgen05\.st\.sync\.aligned\.16x32bx2.x32.b32"
               r"(.|\n)*"
               r"tcgen05\.mma\.cta_group::1.kind::f16"
               r"(.|\n)*"
               r"tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64"
               r"(.|\n)*"
               r"mbarrier.try_wait.parity.shared::cta.b64"
               r"(.|\n)*"
               r"tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
               r"(.|\n)*"
               r"tcgen05.wait::ld.sync.aligned")
    assert re.search(pattern, str(ptx)), "The PTX does not match the expected pattern."
    assert k.asm["cubin"] != b""


def test_compile_only_k_loop() -> None:

    @triton.jit
    def k_loop(a_base, b_base, out, k_tiles):
        SIZE: tl.constexpr = 128
        offs_k = tl.arange(0, SIZE)
        c = tl.zeros((SIZE, SIZE), dtype=tl.float32)
        for k in range(k_tiles):
            a_ptr = a_base + tl.arange(0, SIZE)[:, None] * SIZE + offs_k[None, :]
            b_ptr = b_base + offs_k[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
            offs_k = offs_k + SIZE
            a = tl.load(a_ptr)
            b = tl.load(b_ptr)
            c += tl.dot(a, b)
        out_ptr = out + tl.arange(0, SIZE)[:, None] * SIZE + tl.arange(0, SIZE)[None, :]
        tl.store(out_ptr, c)

    k = triton.compile(
        triton.compiler.ASTSource(fn=k_loop,
                                  signature={"a_base": "*fp16", "b_base": "*fp16", "out": "*fp16", "k_tiles":
                                             "i32"}, constexprs={}), target=GPUTarget("cuda", 100, 32))
    ttgir = k.asm["ttgir"]

    pattern = (r"%(?P<TMEM_BASE>\w+) = arith.constant dense<0.000000e\+00>"
               r"(.|\n)*?"
               r"%(?P<TMEM>\w+) = ttng\.tmem_alloc (%(?P=TMEM_BASE))?"
               r"(.|\n)*?"
               r"scf\.for"
               r"(.|\n)*?"
               r"%(?P<A>\w+) = tt\.load"
               r"(.|\n)*?"
               r"%(?P<A_SHMEM>\w+) = ttg\.local_alloc %(?P=A)"
               r"(.|\n)*?"
               r"%(?P<B>\w+) = tt\.load"
               r"(.|\n)*?"
               r"%(?P<B_SHMEM>\w+) = ttg\.local_alloc %(?P=B)"
               r"(.|\n)*?"
               r"ttng\.tc_gen5_mma %(?P=A_SHMEM), %(?P=B_SHMEM), %(?P=TMEM)"
               r"(.|\n)*?"
               r"scf\.yield")

    assert re.search(pattern, str(ttgir)), "The TTGIR does not match the expected pattern."
    assert k.asm["cubin"] != b""


def test_compile_only_dot_mxfp() -> None:

    @triton.jit
    def simple_dot_mxfp(a_base, b_base, a_scale, b_scale, out, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                        BLOCK_K: tl.constexpr):
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_K
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_K
        a_ptr = a_base + tl.arange(0, BLOCK_M)[:, None] * PACKED_BLOCK_K_A + tl.arange(0, PACKED_BLOCK_K_A)[None, :]
        b_ptr = b_base + tl.arange(0, PACKED_BLOCK_K_B)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

        SCALE_BLOCK_K: tl.constexpr = BLOCK_K // 32
        scale_a_ptr = a_scale + tl.arange(0, BLOCK_M)[:, None] * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)[None, :]
        scale_b_ptr = b_scale + tl.arange(0, BLOCK_N)[:, None] * SCALE_BLOCK_K + tl.arange(0, SCALE_BLOCK_K)[None, :]

        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        a_scale = tl.load(scale_a_ptr)
        b_scale = tl.load(scale_b_ptr)
        c = tl.dot_scaled(a, a_scale, "e4m3", b, b_scale, "e4m3")
        out_ptr = out + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        tl.store(out_ptr, c)

    k = triton.compile(
        triton.compiler.ASTSource(
            fn=simple_dot_mxfp, signature={
                "a_base": "*u8", "b_base": "*u8", "a_scale": "*u8", "b_scale": "*u8", "out": "*fp32", "BLOCK_M":
                "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr"
            }, constexprs={"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}), target=GPUTarget("cuda", 100, 32))
    ttgir = k.asm["ttgir"]
    pattern = (r"ttng.tc_gen5_mma_scaled (.*) lhs = e4m3 rhs = e4m3")
    assert re.search(pattern, str(ttgir)), "The TTGIR does not match the expected pattern."

    ptx = k.asm["ptx"]
    pattern = (r"tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32")
    assert re.search(pattern, str(ptx)), "The PTX does not match the expected pattern."
    assert k.asm["cubin"] != b""


def test_signature_ordering():
    """
    Checks that ASTSource always uses the argument order from
    fn.arg_names and not the signature.
    """

    @triton.jit
    def kernel(a, o, N: tl.constexpr):
        tl.store(o + N, tl.load(a + N))

    # Add the arguments so the order always differs
    # from the order in fn.arg_names.
    signature = {}
    signature["N"] = "constexpr"
    signature["a"] = "*fp32"
    signature["o"] = "*fp32"
    src = ASTSource(
        fn=kernel,
        constexprs={"N": 32},
        signature=signature,
    )
    target = triton.runtime.driver.active.get_current_target()
    triton.compile(src=src, target=target)


def test_fp8_compiles_for_multiple_architectures_hip():
    """
    Validate FP8 compilation succeeds for architectures with different
    hardware support.

    gfx950 has native FP8 instructions; gfx942 does not and requires software
    conversion. Compiling for both in sequence must succeed for each target.
    """

    @triton.jit
    def fp8_convert(src, dst):
        idx = tl.arange(0, 64)
        tl.store(dst + idx, tl.load(src + idx).to(tl.float8e5))

    src = ASTSource(fn=fp8_convert, signature={"src": "*fp32", "dst": "*fp8e5"}, constexprs={})
    triton.compile(src, target=GPUTarget("hip", "gfx950", 64))
    triton.compile(src, target=GPUTarget("hip", "gfx942", 64))


def test_fp8_compiles_for_multiple_architectures_cuda():
    """
    Validate FP8 compilation succeeds for architectures with different
    hardware support.

    SM90 has native FP8 instructions; SM80 does not and requires software
    conversion. Compiling for both in sequence must succeed for each target.
    """

    @triton.jit
    def fp8_convert(src, dst):
        idx = tl.arange(0, 64)
        tl.store(dst + idx, tl.load(src + idx).to(tl.float8e5))

    src = ASTSource(fn=fp8_convert, signature={"src": "*fp32", "dst": "*fp8e5"}, constexprs={})
    triton.compile(src, target=GPUTarget("cuda", 90, 32))
    triton.compile(src, target=GPUTarget("cuda", 80, 32))
