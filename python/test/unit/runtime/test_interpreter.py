import numpy as np
import pytest
import itertools

from triton._C.libtriton import interpreter as _interpreter
from triton._C.libtriton import ir
import triton.language as tl
from triton.runtime import interpreter


def _element_ptrs(array: np.ndarray) -> np.ndarray:
    base = np.uint64(array.ctypes.data)
    offsets = np.arange(array.size, dtype=np.uint64) * np.uint64(array.itemsize)
    return (base + offsets).reshape(array.shape)


@pytest.mark.parametrize("dtype, prefix", [
    (tl.float16, "fcmpO"),
    (tl.float32, "fcmpO"),
    (tl.float64, "fcmpO"),
    (tl.float32, "fcmpU"),
    (tl.int32, "icmpS"),
    (tl.uint32, "icmpU"),
    (tl.int1, "icmpU"),
])
@pytest.mark.parametrize("op, expected", [
    ("LT", [True, False, False, False]),
    ("LE", [True, True, False, True]),
    ("GT", [False, False, True, False]),
    ("GE", [False, True, True, True]),
    ("EQ", [False, True, False, True]),
    ("NE", [True, False, True, False]),
])
def test_compare_op_return_type(dtype, prefix, op, expected):
    builder = interpreter.InterpreterBuilder()
    np_dtype = interpreter._get_np_dtype(dtype)
    lhs = interpreter.TensorHandle(np.array([0, 1, 2, 3], dtype=np_dtype), dtype)
    rhs = interpreter.TensorHandle(np.array([1, 1, 0, 3], dtype=np_dtype), dtype)
    if dtype.is_int() and op in ("EQ", "NE"):
        prefix = "icmp"
    result = getattr(builder, f"create_{prefix}{op}")(lhs, rhs)
    assert result.dtype == tl.int1
    assert result.data.dtype == np.bool_
    np.testing.assert_array_equal(result.data, expected)
    converted = builder.create_ui_to_fp(result, tl.float16)
    np.testing.assert_array_equal(converted.data, np.array(expected, dtype=np.float16))


def test_bf16_to_fp16_rounding_and_overflow():
    # Signed zeros, finite values, overflow, subnormal rounding, and infinities.
    bits = np.array([0x0000, 0x8000, 0x3F80, 0xBF80, 0x4781, 0xC781, 0x3300, 0x3301, 0x33C0, 0xB3C0, 0x7F80, 0xFF80],
                    dtype=np.uint16)
    expected = np.array(
        [0x0000, 0x8000, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x0000, 0x0001, 0x0002, 0x8002, 0x7C00, 0xFC00],
        dtype=np.uint16)
    src = interpreter.TensorHandle(bits, tl.bfloat16)
    with np.errstate(over="ignore"):
        result = interpreter.InterpreterBuilder().create_fp_to_fp(src, tl.float16, ir.ROUNDING_MODE.RTNE)
    np.testing.assert_array_equal(result.data.view(np.uint16), expected)


def test_fp16_to_bf16_exact_values():
    bits = np.array([0x0000, 0x8000, 0x3C00, 0xBC00, 0x3C08, 0x0001, 0x7C00, 0xFC00, 0x7E00], dtype=np.uint16)
    expected = np.array([0x0000, 0x8000, 0x3F80, 0xBF80, 0x3F81, 0x3380, 0x7F80, 0xFF80, 0x7FC0], dtype=np.uint16)
    src = interpreter.TensorHandle(bits.view(np.float16), tl.float16)
    result = interpreter.InterpreterBuilder().create_fp_to_fp(src, tl.bfloat16, ir.ROUNDING_MODE.RTNE)
    np.testing.assert_array_equal(result.data, expected)


@pytest.mark.parametrize("src_dtype, dst_dtype, bits, expected", [
    (tl.float32, tl.bfloat16, [0x00008000, 0x00018000, 0x007F8000, 0x80008000], [0x0000, 0x0002, 0x0080, 0x8000]),
    (tl.float16, tl.bfloat16, [0x0001, 0x8001], [0x3380, 0xB380]),
    (tl.float8e4nv, tl.float8e5, [0x01, 0x02, 0x07], [0x18, 0x1C, 0x23]),
])
def test_float_conversion_subnormal_rounding(src_dtype, dst_dtype, bits, expected):
    uint_dtype = getattr(np, f"uint{src_dtype.primitive_bitwidth}")
    data = np.array(bits, dtype=uint_dtype).view(interpreter._get_np_dtype(src_dtype))
    result = interpreter._convert_float(data, src_dtype, dst_dtype, ir.ROUNDING_MODE.RTNE)
    np.testing.assert_array_equal(result, expected)


def test_atomic_poll_tensor_shares_timeout(monkeypatch) -> None:
    builder = interpreter.InterpreterBuilder()
    data = np.array([0, 0, 1], dtype=np.int32)
    addresses = _element_ptrs(data)
    ptr = interpreter.TensorHandle(addresses, tl.pointer_type(tl.int32))
    expected = interpreter.TensorHandle(np.ones(3, dtype=np.int32), tl.int32)
    timeout = interpreter.TensorHandle(np.array(2, dtype=np.uint64), tl.uint64)
    clock = itertools.count()
    monkeypatch.setattr(interpreter.time, "perf_counter_ns", lambda: next(clock))
    loads = []
    original_load = builder.create_load

    def load(element_ptr, *args):
        loads.append(element_ptr.data.item())
        return original_load(element_ptr, *args)

    monkeypatch.setattr(builder, "create_load", load)
    result = builder.create_atomic_poll(ptr, expected, timeout, None, None)

    np.testing.assert_array_equal(result.data, [False, False, True])
    # The first element exhausts the budget; later elements still get one load.
    assert loads == [addresses[0], addresses[0], addresses[1], addresses[2]]


def test_load_accepts_non_contiguous_ndarray_views() -> None:
    data = np.arange(12, dtype=np.int32).reshape(3, 4)
    ptrs = _element_ptrs(data)[:, ::2]
    mask = np.array([[True, False, True, False], [False, True, False, True], [True, True, False, False]])[:, ::2]
    other = (np.arange(12, dtype=np.int32).reshape(3, 4) + 100)[:, ::2]

    loaded = _interpreter.load(ptrs, mask, other, np.int32)

    np.testing.assert_array_equal(loaded, np.where(mask, data[:, ::2], other))


def test_store_accepts_non_contiguous_ndarray_views() -> None:
    dst = np.zeros((3, 4), dtype=np.int32)
    ptrs = _element_ptrs(dst)[:, 1::2]
    values = (np.arange(12, dtype=np.int32).reshape(3, 4) + 10)[:, 1::2]
    mask = np.array([[True, False, False, True], [False, True, True, False], [True, False, True, False]])[:, 1::2]

    _interpreter.store(ptrs, values, mask)

    expected = np.zeros((3, 4), dtype=np.int32)
    expected[:, 1::2] = np.where(mask, values, expected[:, 1::2])
    np.testing.assert_array_equal(dst, expected)


def test_atomic_rmw_accepts_non_contiguous_ndarray_views() -> None:
    dst = np.arange(12, dtype=np.int32).reshape(3, 4)
    ptrs = _element_ptrs(dst)[:, ::2]
    values = (np.arange(12, dtype=np.int32).reshape(3, 4) + 1)[:, ::2]
    mask = np.ones((3, 4), dtype=bool)[:, ::2]

    old = _interpreter.atomic_rmw(_interpreter.RMW_OP.ADD, ptrs, values, mask, _interpreter.MEM_SEMANTIC.RELAXED)

    original = np.arange(12, dtype=np.int32).reshape(3, 4)
    np.testing.assert_array_equal(old, original[:, ::2])
    original[:, ::2] += values
    np.testing.assert_array_equal(dst, original)


def test_atomic_cas_accepts_non_contiguous_ndarray_views() -> None:
    dst = np.arange(12, dtype=np.int32).reshape(3, 4)
    ptrs = _element_ptrs(dst)[:, ::2]
    expected = dst.copy()[:, ::2]
    desired = (np.arange(12, dtype=np.int32).reshape(3, 4) + 200)[:, ::2]

    old = _interpreter.atomic_cas(ptrs, expected, desired, _interpreter.MEM_SEMANTIC.RELAXED)

    original = np.arange(12, dtype=np.int32).reshape(3, 4)
    np.testing.assert_array_equal(old, original[:, ::2])
    original[:, ::2] = desired
    np.testing.assert_array_equal(dst, original)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_fma_broadcast_and_strides(dtype):
    x = np.arange(12, dtype=dtype).reshape(3, 4)[:, ::2]
    y = np.array(2, dtype=dtype)
    z = np.arange(2, dtype=dtype)
    handles = [interpreter.TensorHandle(value, getattr(tl, dtype)) for value in (x, y, z)]
    result = interpreter.InterpreterBuilder().create_fma(*handles)
    assert result.data.shape == x.shape
    assert result.data.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result.data, x * y + z)


@pytest.mark.parametrize("compute_type", [tl.bfloat16, tl.float16])
@pytest.mark.parametrize("native_operands, native_scales, signed_scale", [
    (False, False, False),
    (False, False, True),
    (True, False, False),
    (True, False, True),
    (False, True, False),
    (True, True, False),
])
@pytest.mark.parametrize("rhs_scale", [False, True])
def test_dot_scaled_e8m0_boundaries(compute_type, native_operands, native_scales, signed_scale, rhs_scale):
    scales = np.repeat(np.array([0, 127, 128, 255], dtype=np.uint8), 8)[:, None]
    if signed_scale:
        scales = scales.view(np.int8)
    scale_dtype = tl.float8e8m0fnu if native_scales else (tl.int8 if signed_scale else tl.uint8)
    scales = interpreter.TensorHandle(scales, scale_dtype)
    # Each packed byte contains two FP4 values of one.
    values = np.full((32, 16), 0x22, dtype=np.uint8)
    value_dtype = tl.float4e2m1x2 if native_operands else tl.uint8
    values = interpreter.TensorHandle(values.T if rhs_scale else values, value_dtype)
    normal_bits = 0x3F80 if compute_type == tl.bfloat16 else 0x3C00
    normal = interpreter.TensorHandle(np.full((32, 32), normal_bits, dtype=np.uint16), compute_type)
    normal_format = ir.ScaleDotElemTypeTY.BF16 if compute_type == tl.bfloat16 else ir.ScaleDotElemTypeTY.FP16
    scaled_operand = (values, scales, ir.ScaleDotElemTypeTY.E2M1)
    normal_operand = (normal, None, normal_format)
    lhs, rhs = (normal_operand, scaled_operand) if rhs_scale else (scaled_operand, normal_operand)
    acc = interpreter.TensorHandle(np.zeros((32, 32), dtype=np.float32), tl.float32)
    result = interpreter.InterpreterBuilder().create_dot_scaled(*lhs, *rhs, False, True, True, acc)

    minimum = 2.0**-122 if compute_type == tl.bfloat16 else 0.0
    expected = np.repeat(np.array([minimum, 32.0, 64.0, np.nan], dtype=np.float32), 8)[:, None]
    expected = np.broadcast_to(expected, (32, 32))
    np.testing.assert_array_equal(result.data, expected.T if rhs_scale else expected)


@pytest.mark.parametrize("dtype_name", ["float4e2m1x2", "float8e8m0fnu"])
@pytest.mark.parametrize("transpose", [False, True])
def test_low_precision_copy_preserves_packed_bytes(dtype_name, transpose):
    dtype = getattr(tl, dtype_name)
    builder = interpreter.InterpreterBuilder()
    storage = np.arange(256, dtype=np.uint8).reshape(16, 16)
    addresses = _element_ptrs(storage)
    if transpose:
        addresses = addresses.T
    source = interpreter.TensorHandle(addresses, tl.pointer_type(dtype))
    loaded = builder.create_load(source, None, False)
    assert loaded.dtype == dtype
    assert loaded.data.dtype == np.uint8
    assert loaded.data.shape == storage.shape

    raw = builder.create_bitcast(loaded, tl.uint8)
    restored = builder.create_bitcast(raw, dtype)
    assert raw.dtype == tl.uint8
    assert restored.dtype == dtype
    result = np.zeros_like(storage)
    target = interpreter.TensorHandle(_element_ptrs(result), tl.pointer_type(dtype))
    builder.create_store(target, restored, None)
    np.testing.assert_array_equal(result, storage.T if transpose else storage)


@pytest.mark.parametrize("dtype_name", ["float4e2m1x2", "float8e8m0fnu"])
@pytest.mark.parametrize("numeric_dtype", [tl.uint8, tl.float32])
@pytest.mark.parametrize("to_native", [False, True])
def test_low_precision_numeric_cast_requires_explicit_conversion(dtype_name, numeric_dtype, to_native):
    native_dtype = getattr(tl, dtype_name)
    src_dtype, dst_dtype = (numeric_dtype, native_dtype) if to_native else (native_dtype, numeric_dtype)
    data = np.array([0, 1, 127, 128, 254, 255], dtype=interpreter._get_np_dtype(src_dtype))
    value = interpreter.TensorHandle(data, src_dtype)
    with pytest.raises((TypeError, ValueError), match="(?i)(cast|convert|conversion|storage)"):
        interpreter.InterpreterBuilder().cast_impl(value, dst_dtype)


@pytest.mark.parametrize("dtype", [tl.float4e2m1x2, tl.float8e8m0fnu])
@pytest.mark.parametrize("operation", ["device_assert", "assume"])
@pytest.mark.parametrize("debug", [False, True])
def test_low_precision_assert_conditions_reject_raw_bits(dtype, operation, debug):
    from triton.language.semantic import TritonSemantic

    builder = interpreter.InterpreterBuilder()
    builder.options = interpreter.InterpreterOptions(debug=debug)
    semantic = TritonSemantic(builder)
    condition = tl.tensor(interpreter.TensorHandle(np.array([127], dtype=np.uint8), dtype), tl.block_type(dtype, [1]))

    def evaluate():
        if operation == "device_assert":
            return semantic.device_assert(condition, "native condition", None)
        return semantic.assume(condition)

    if operation == "device_assert" and not debug:
        assert evaluate() is None
    else:
        with pytest.raises(TypeError, match=str(dtype)):
            evaluate()


@pytest.mark.parametrize("torch_name, triton_name", [("float4_e2m1fn_x2", "float4e2m1x2"),
                                                     ("float8_e8m0fnu", "float8e8m0fnu")])
@pytest.mark.parametrize("transpose", [False, True])
def test_low_precision_torch_copy_preserves_type_and_strides(torch_name, triton_name, transpose):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, torch_name):
        pytest.skip(f"PyTorch does not provide {torch_name}")

    def kernel(source, destination, stride_m: tl.constexpr, stride_n: tl.constexpr, DTYPE: tl.constexpr):
        rows = tl.arange(0, 16)[:, None]
        columns = tl.arange(0, 16)[None, :]
        value = tl.load(source + rows * stride_m + columns * stride_n)
        tl.static_assert(value.dtype == DTYPE)
        raw = value.to(tl.uint8, bitcast=True)
        restored = raw.to(DTYPE, bitcast=True)
        tl.store(destination + rows * 16 + columns, restored)

    storage = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    source = storage.view(getattr(torch, torch_name))
    if transpose:
        source = source.T
    destination = torch.empty((16, 16), dtype=source.dtype)
    interpreter.InterpretedFunction(kernel)[(1, )](source, destination, *source.stride(), getattr(tl, triton_name))
    torch.testing.assert_close(destination.view(torch.uint8), storage.T if transpose else storage)


@pytest.mark.parametrize("torch_name", ["float4_e2m1fn_x2", "float8_e8m0fnu"])
@pytest.mark.parametrize("kind", ["tensor", "tuple", "descriptor"])
@pytest.mark.parametrize("annotation", ["*u8", "fp32", "i32"])
def test_low_precision_interpreter_rejects_wrong_annotation(torch_name, kind, annotation):
    from triton.tools.tensor_descriptor import TensorDescriptor

    torch = pytest.importorskip("torch")
    if not hasattr(torch, torch_name):
        pytest.skip(f"PyTorch does not provide {torch_name}")

    def kernel(source: "*u8"):  # noqa: F722 - Triton pointer annotation syntax.
        tl.load(source)

    kernel.__annotations__["source"] = annotation
    argument = torch.empty((16, 16), dtype=getattr(torch, torch_name))
    if kind == "tuple":
        argument = (1, (argument, ))
    elif kind == "descriptor":
        argument = TensorDescriptor.from_tensor(argument, [16, 16])
    with pytest.raises(TypeError, match="does not match argument dtype"):
        interpreter.InterpretedFunction(kernel)[(1, )](argument)


@pytest.mark.parametrize("annotation", ["i32", "*u8", "fp32", "u1"])
@pytest.mark.parametrize("argument", [np.float32(1), object()])
def test_low_precision_annotations_preserve_legacy_interpreter_arguments(annotation, argument):
    observed = []

    def kernel(value):
        observed.append(value)

    kernel.__annotations__ = {"value": annotation}
    interpreter.GridExecutor(kernel, ["value"], (1, ))(argument)
    assert observed == [argument]


def test_low_precision_annotations_preserve_legacy_interpreter_constexpr():

    def quoted(value: "tl.constexpr"):
        pass

    def unquoted(value: tl.constexpr):
        pass

    quoted_fn = interpreter.InterpretedFunction(quoted).rewrite()
    unquoted_fn = interpreter.InterpretedFunction(unquoted).rewrite()
    assert interpreter.GridExecutor(quoted_fn, ["value"], (1, )).constexprs == []
    assert interpreter.GridExecutor(unquoted_fn, ["value"], (1, )).constexprs == ["value"]


@pytest.mark.parametrize("dtype", [tl.float4e2m1x2, tl.float8e8m0fnu])
@pytest.mark.parametrize("operation, combine_name", [
    ("reduce", "_elementwise_max_ignore_nan"),
    ("reduce", "_elementwise_min_ignore_nan"),
    ("reduce", "_elementwise_max_propagate_nan"),
    ("reduce", "_elementwise_min_propagate_nan"),
    ("reduce", "_sum_combine"),
    ("scan", "_sum_combine"),
    ("scan", "_prod_combine"),
])
def test_low_precision_numeric_reduce_scan_rejected(dtype, operation, combine_name):
    value = tl.tensor(interpreter.TensorHandle(np.array([1, 2, 3, 4], dtype=np.uint8), dtype),
                      tl.block_type(dtype, [4]))
    combine = getattr(tl.standard, combine_name)
    if operation == "reduce":
        op = interpreter.ReduceOps(0, combine, keep_dims=False)
    else:
        op = interpreter.ScanOps(0, combine, reverse=False)
    with pytest.raises(TypeError, match="storage dtype"):
        op.apply(value)


@pytest.mark.parametrize("dtype", [tl.float4e2m1x2, tl.float8e8m0fnu])
@pytest.mark.parametrize("operation", ["reduce", "scan"])
@pytest.mark.parametrize("result", ["same", "typed_constant", "uint8", "other_storage", "constant"])
def test_low_precision_reduce_scan_combine_type_checked(dtype, operation, result):
    value = tl.tensor(interpreter.TensorHandle(np.array([255, 1, 127, 0], dtype=np.uint8), dtype),
                      tl.block_type(dtype, [4]))

    def combine(lhs, rhs):
        if result == "same":
            return lhs
        if result == "constant":
            return 0
        if result == "typed_constant":
            return tl.tensor(interpreter.TensorHandle(np.array([0], dtype=np.uint8), dtype), dtype)
        result_dtype = tl.uint8 if result == "uint8" else (
            tl.float8e8m0fnu if dtype == tl.float4e2m1x2 else tl.float4e2m1x2)
        return tl.tensor(interpreter.TensorHandle(lhs.handle.data, result_dtype), result_dtype)

    combine = interpreter.InterpretedFunction(combine)
    if operation == "reduce":
        op = interpreter.ReduceOps(0, combine, keep_dims=False)
    else:
        op = interpreter.ScanOps(0, combine, reverse=False)
    if result in ("same", "typed_constant"):
        output = op.apply(value)
        assert output.dtype == dtype
        assert output.handle.dtype == dtype
        result_byte = 255 if result == "same" else 0
        expected = [result_byte] if operation == "reduce" else [255] + [result_byte] * 3
        np.testing.assert_array_equal(output.handle.data, expected)
    else:
        with pytest.raises(TypeError, match="(must match|storage dtype)"):
            op.apply(value)


@pytest.mark.parametrize("operation", ["reduce", "scan"])
def test_reduce_scan_numeric_constant_combine_legacy(operation):
    value = tl.tensor(interpreter.TensorHandle(np.array([1, 2, 3, 4], dtype=np.uint8), tl.uint8),
                      tl.block_type(tl.uint8, [4]))

    def combine(lhs, rhs):
        return 0

    combine = interpreter.InterpretedFunction(combine)
    if operation == "reduce":
        op = interpreter.ReduceOps(0, combine, keep_dims=False)
    else:
        op = interpreter.ScanOps(0, combine, reverse=False)
    output = op.apply(value)
    assert output.dtype == tl.uint8
    np.testing.assert_array_equal(output.handle.data, [0] if operation == "reduce" else [1, 0, 0, 0])


@pytest.mark.parametrize("torch_name, triton_dtype", [("uint8", tl.uint8), ("float4_e2m1fn_x2", tl.float4e2m1x2),
                                                      ("float8_e8m0fnu", tl.float8e8m0fnu)])
@pytest.mark.parametrize("fill_bits", [0, 0xA5])
@pytest.mark.parametrize("mode", ["scalar", "block_scalar_other", "block_other"])
def test_low_precision_masked_load_preserves_typed_other(torch_name, triton_dtype, fill_bits, mode):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, torch_name):
        pytest.skip(f"PyTorch does not provide {torch_name}")

    def kernel(source, destination, DTYPE: tl.constexpr, FILL_BITS: tl.constexpr, MODE: tl.constexpr):
        if MODE == "scalar":
            other = tl.full((), FILL_BITS, tl.uint8).to(DTYPE, bitcast=True)
            active = tl.load(source, mask=True, other=other)
            inactive = tl.load(source + 1, mask=False, other=other)
            tl.static_assert(active.dtype == DTYPE)
            tl.static_assert(inactive.dtype == DTYPE)
            tl.store(destination, active)
            tl.store(destination + 1, inactive)
        else:
            offsets = tl.arange(0, 4)
            if MODE == "block_scalar_other":
                other = tl.full((), FILL_BITS, tl.uint8).to(DTYPE, bitcast=True)
            else:
                other = tl.full([4], FILL_BITS, tl.uint8).to(DTYPE, bitcast=True)
            value = tl.load(source + offsets, mask=offsets % 2 == 0, other=other)
            tl.static_assert(value.dtype == DTYPE)
            tl.store(destination + offsets, value)

    source = torch.tensor([17, 34, 51, 68], dtype=torch.uint8).view(getattr(torch, torch_name))
    output_size = 2 if mode == "scalar" else 4
    destination = torch.empty((output_size, ), dtype=source.dtype)
    interpreter.InterpretedFunction(kernel)[(1, )](source, destination, triton_dtype, fill_bits, mode)
    expected = [17, fill_bits] if mode == "scalar" else [17, fill_bits, 51, fill_bits]
    torch.testing.assert_close(destination.view(torch.uint8), torch.tensor(expected, dtype=torch.uint8))


@pytest.mark.parametrize("torch_name", ["float4_e2m1fn_x2", "float8_e8m0fnu"])
@pytest.mark.parametrize("scalar", [False, True])
def test_low_precision_masked_load_rejects_native_mask(torch_name, scalar):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, torch_name):
        pytest.skip(f"PyTorch does not provide {torch_name}")

    def kernel(source, masks, destination, SCALAR: tl.constexpr):
        if SCALAR:
            mask = tl.load(masks)
            value = tl.load(source, mask=mask, other=0)
            tl.store(destination, value)
        else:
            offsets = tl.arange(0, 4)
            mask = tl.load(masks + offsets)
            value = tl.load(source + offsets, mask=mask, other=0)
            tl.store(destination + offsets, value)

    source = torch.tensor([17, 34, 51, 68], dtype=torch.uint8)
    masks = torch.tensor([0, 127, 255, 1], dtype=torch.uint8).view(getattr(torch, torch_name))
    destination = torch.empty_like(source)
    with pytest.raises(interpreter.InterpreterError, match="storage dtype"):
        interpreter.InterpretedFunction(kernel)[(1, )](source, masks, destination, scalar)


@pytest.mark.parametrize("torch_name", ["float4_e2m1fn_x2", "float8_e8m0fnu"])
@pytest.mark.parametrize("bits", [0, 127])
def test_low_precision_masked_load_keeps_native_conditions_invalid(torch_name, bits):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, torch_name):
        pytest.skip(f"PyTorch does not provide {torch_name}")

    def kernel(source, destination):
        if tl.load(source):
            tl.store(destination, 1)

    source = torch.tensor([bits], dtype=torch.uint8).view(getattr(torch, torch_name))
    destination = torch.empty((1, ), dtype=torch.uint8)
    with pytest.raises(interpreter.InterpreterError, match="cannot be used as conditions"):
        interpreter.InterpretedFunction(kernel)[(1, )](source, destination)


@pytest.mark.parametrize("lhs_dtype", [tl.float4e2m1x2, tl.float8e8m0fnu, tl.uint8])
@pytest.mark.parametrize("rhs_dtype", [tl.float4e2m1x2, tl.float8e8m0fnu, tl.uint8])
def test_low_precision_pointer_join_type_checked(lhs_dtype, rhs_dtype):
    from triton.language.semantic import TritonSemantic

    builder = interpreter.InterpreterBuilder()
    semantic = TritonSemantic(builder)
    storage = np.arange(256, dtype=np.uint8).reshape(128, 2)
    addresses = _element_ptrs(storage)
    lhs_type = tl.pointer_type(lhs_dtype)
    rhs_type = tl.pointer_type(rhs_dtype)
    lhs = tl.tensor(interpreter.TensorHandle(addresses[:, 0], lhs_type), tl.block_type(lhs_type, [128]))
    rhs = tl.tensor(interpreter.TensorHandle(addresses[:, 1], rhs_type), tl.block_type(rhs_type, [128]))

    if lhs_dtype != rhs_dtype:
        with pytest.raises(TypeError, match="must match"):
            semantic.join(lhs, rhs)
    else:
        joined = semantic.join(lhs, rhs)
        assert joined.dtype == lhs_type
        assert joined.handle.dtype == lhs_type
        loaded = builder.create_load(joined.handle, None, False)
        assert loaded.dtype == lhs_dtype
        np.testing.assert_array_equal(loaded.data, storage)


@pytest.mark.parametrize("dtype", [tl.float4e2m1x2, tl.float8e8m0fnu])
@pytest.mark.parametrize("operation", ["reduce", "scan"])
@pytest.mark.parametrize("result", ["same", "uint8", "other_storage", "constant"])
def test_low_precision_pointer_reduce_scan_combine_type_checked(dtype, operation, result):
    builder = interpreter.InterpreterBuilder()
    storage = np.array([255, 1, 127, 0], dtype=np.uint8)
    pointer_type = tl.pointer_type(dtype)
    value = tl.tensor(interpreter.TensorHandle(_element_ptrs(storage), pointer_type), tl.block_type(pointer_type, [4]))

    def combine(lhs, rhs):
        if result == "same":
            return lhs
        if result == "constant":
            return 0
        result_dtype = tl.uint8 if result == "uint8" else (
            tl.float8e8m0fnu if dtype == tl.float4e2m1x2 else tl.float4e2m1x2)
        result_type = tl.pointer_type(result_dtype)
        return tl.tensor(interpreter.TensorHandle(lhs.handle.data, result_type), result_type)

    combine = interpreter.InterpretedFunction(combine)
    if operation == "reduce":
        op = interpreter.ReduceOps(0, combine, keep_dims=False)
    else:
        op = interpreter.ScanOps(0, combine, reverse=False)
    if result != "same":
        with pytest.raises(TypeError, match="must match"):
            op.apply(value)
    else:
        output = op.apply(value)
        assert output.dtype == pointer_type
        assert output.handle.dtype == pointer_type
        loaded = builder.create_load(output.handle, None, False)
        assert loaded.dtype == dtype
        np.testing.assert_array_equal(loaded.data, [255] if operation == "reduce" else [255, 255, 255, 255])


@pytest.mark.parametrize("operation", ["reduce", "scan"])
def test_pointer_reduce_scan_numeric_constant_combine_legacy(operation):
    storage = np.array([1, 2, 3, 4], dtype=np.uint8)
    pointer_type = tl.pointer_type(tl.uint8)
    addresses = _element_ptrs(storage)
    value = tl.tensor(interpreter.TensorHandle(addresses, pointer_type), tl.block_type(pointer_type, [4]))

    def combine(lhs, rhs):
        return 0

    combine = interpreter.InterpretedFunction(combine)
    if operation == "reduce":
        op = interpreter.ReduceOps(0, combine, keep_dims=False)
    else:
        op = interpreter.ScanOps(0, combine, reverse=False)
    output = op.apply(value)
    assert output.dtype == pointer_type
    np.testing.assert_array_equal(output.handle.data, [0] if operation == "reduce" else [addresses[0], 0, 0, 0])
