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
@pytest.mark.parametrize("signed_scale", [False, True])
@pytest.mark.parametrize("rhs_scale", [False, True])
def test_dot_scaled_e8m0_boundaries(compute_type, signed_scale, rhs_scale):
    scales = np.repeat(np.array([0, 127, 128, 255], dtype=np.uint8), 8)[:, None]
    if signed_scale:
        scales = scales.view(np.int8)
    scales = interpreter.TensorHandle(scales, tl.int8 if signed_scale else tl.uint8)
    # Each packed byte contains two FP4 values of one.
    values = np.full((32, 16), 0x22, dtype=np.uint8)
    values = interpreter.TensorHandle(values.T if rhs_scale else values, tl.uint8)
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
