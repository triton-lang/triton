import numpy
import pytest
import subprocess
import sys
import textwrap
import torch
from collections import namedtuple
from triton._C.libtriton import native_specialize_impl
from triton.runtime.jit import MockTensor, JITCallable
from triton._utils import canonicalize_dtype
from triton.backends.nvidia.compiler import CUDABackend
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import NATIVE_TENSOR_SPEC_ALIGN, NATIVE_TENSOR_SPEC_RANGE
from triton.language import constexpr
from triton.language import target_info
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonTensorDescriptor
from triton.experimental.gluon.language._layouts import NVMMASharedLayout


def mock_tensor_from_tensor(tensor):
    return MockTensor(tensor.dtype, tensor.shape)


class MockJITCallable(JITCallable):

    def __init__(self):
        pass

    def cache_key(self):
        return "mock_jit_callable"


class MockFloat(float):

    def __new__(cls, value):
        return super().__new__(cls, value)


class MockInt(int):

    def __new__(cls, value):
        return super().__new__(cls, value)


def reference_specialize_impl(backend, arg, is_const, specialize_value, align):
    if arg is None:
        return ("constexpr", None)
    elif isinstance(arg, bool):
        return ("u1", None)
    elif isinstance(arg, int):
        key = backend.get_int_specialization(arg, align=align) if specialize_value else None
        if arg == 1 and specialize_value:
            return ("constexpr", 1)
        elif -(2**31) <= arg and arg <= 2**31 - 1:
            return ("i32", key)
        elif 2**63 <= arg and arg <= 2**64 - 1:
            return ("u64", key)
        else:
            return ("i64", key)
    elif isinstance(arg, float):
        return ("fp32", None)
    elif hasattr(arg, "data_ptr"):
        dsk = (arg.dtype, is_const)
        res = ("*k" if dsk[1] else "*") + canonicalize_dtype(dsk[0])
        key = backend.get_tensor_specialization(arg, align=align) if specialize_value else None
        return (res, key)
    elif isinstance(arg, JITCallable):
        return ("constexpr", arg.cache_key)
    elif isinstance(arg, constexpr):
        return ("constexpr", arg)
    elif isinstance(arg, tuple):
        spec = [reference_specialize_impl(backend, x, False, True, True) for x in arg]
        make_tuple = lambda vals: type(arg)(*vals) if hasattr(arg, "_fields") else tuple(vals)
        tys = make_tuple([x[0] for x in spec])
        keys = make_tuple([x[1] for x in spec])
        return (tys, keys)
    elif isinstance(arg, TensorDescriptor):
        assert hasattr(arg.base, "data_ptr")
        inner = canonicalize_dtype(arg.base.dtype)
        return (f"tensordesc<{inner}{list(arg.block_shape)}>", None)
    elif isinstance(arg, GluonTensorDescriptor):
        assert hasattr(arg.base, "data_ptr")
        inner = canonicalize_dtype(arg.base.dtype)
        is_im2col = arg.__class__.__name__ == "TensorDescriptorIm2Col"
        type_name = "tensordesc_im2col" if is_im2col else "tensordesc"
        # For im2col mode, include the original tensor rank in the signature
        rank_suffix = f",input_rank={len(arg.shape)}" if is_im2col else ""
        return (f"{type_name}<{inner}{list(arg.block_shape)}{rank_suffix},{arg.layout!r}>", None)
    else:
        raise TypeError("Unsupported type: %s" % type(arg))


def native_inputs_to_specialize():
    return [
        1.0,
        None,
        False,
        True,
        1,
        0,
        -1,
        16,
        17,
        2**31 - 1,
        2**31,
        -2 * 31 - 1,
        2**63 - 1,
        2**63,
        2**63 + 1,
        2**64 - 1,
    ]


def derived_inputs_to_specialize():
    return [
        constexpr(1),
        constexpr(False),
        constexpr(1.0),
        numpy.float64(1.0),
        MockFloat(1.0),
        MockInt(1),
        MockJITCallable(),
    ]


def tuples_to_specialize():
    return [
        (1, 1),
        (False, True),
        namedtuple('strides', ['x', 'y'])(1, 1),
        namedtuple('flags', ['x', 'y'])(False, True),
    ]


def tensors_to_specialize():
    return [
        torch.empty(shape, dtype=dtype, device="cpu")
        for shape in [(1, ), (1, 1), (16, ), (16, 16), (128, ), (128, 128)]
        for dtype in [torch.float64, torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]
    ]


def tensordescriptors_to_specialize():
    return [
        TensorDescriptor.from_tensor(tensor, block_shape=tensor.shape)
        for tensor in tensors_to_specialize()
        if tensor.shape[-1] % 16 == 0
    ]


def gluon_tensordescriptors_to_specialize():
    return [
        GluonTensorDescriptor.from_tensor(
            tensor,
            block_shape=tensor.shape,
            layout=NVMMASharedLayout(0, tensor.dtype.itemsize * 8, len(tensor.shape)),
        ) for tensor in tensors_to_specialize() if tensor.shape[-1] % 16 == 0 and tensor.dtype.itemsize <= 4
    ]


def test_specialize_gluon_fp4_descriptor_without_active_target(monkeypatch):
    monkeypatch.setattr(target_info, "current_target", lambda: None)
    tensor = torch.empty((128, 128), dtype=torch.uint8)
    layout = NVMMASharedLayout(128, 8, fp4_padded=True)

    descriptor = GluonTensorDescriptor.from_tensor(tensor, [128, 64], layout)

    assert native_specialize_impl(CUDABackend, descriptor, False, True,
                                  True) == reference_specialize_impl(CUDABackend, descriptor, False, True, True)


def mock_tensors_to_specialize():
    return [mock_tensor_from_tensor(tensor) for tensor in tensors_to_specialize()]


@pytest.mark.parametrize("input_generator", [
    native_inputs_to_specialize,
    tuples_to_specialize,
    tensors_to_specialize,
    tensordescriptors_to_specialize,
    gluon_tensordescriptors_to_specialize,
    mock_tensors_to_specialize,
])
@pytest.mark.parametrize("backend", [CUDABackend, HIPBackend])
@pytest.mark.parametrize("is_const", [True, False])
@pytest.mark.parametrize("specialize_value", [True, False])
@pytest.mark.parametrize("align", [True, False])
def test_specialize_impl(input_generator, backend, is_const, specialize_value, align):
    for arg in input_generator():
        result = native_specialize_impl(backend, arg, is_const, specialize_value, align)
        expected = reference_specialize_impl(backend, arg, is_const, specialize_value, align)
        assert result == expected


# ---------------------------------------------------------------------------
# AMD "S" (32-bit pointer range) specialization bit.
#
# HIP sets supports_native_tensor_specialization = NATIVE_TENSOR_SPEC_RANGE, so the
# native specializer appends "S" in C++ when the argument's *storage* fits in 2GB. The resulting key is
# consumed by HIPBackend.parse_attr() and must be exactly one of "", "D", "S",
# "DS", with "D" always before "S".
#
# The key is not gated on AMDGCN_USE_BUFFER_OPS -- it states what is true of the
# argument. HIPBackend.parse_attr() applies the knob when turning "S" into
# tt.pointer_range = 32, which is where the knob has always had its effect.
#
# None of this needs a ROCm device or 2GB of real memory: meta-device tensors
# carry a real untyped_storage().size() without any backing allocation.
# ---------------------------------------------------------------------------

MAX_INT_32 = 2**31 - 1


def small_meta_tensor():
    """A tensor whose entire storage comfortably fits in 2GB."""
    return torch.empty(1024, dtype=torch.float32, device="meta")


def over_2gb_meta_tensor():
    """A tensor whose storage is just past 2GB (2147483712 > 2**31 - 1 bytes)."""
    return torch.empty(2**29 + 16, dtype=torch.float32, device="meta")


def make_hip_backend():
    from triton.backends.compiler import GPUTarget
    return HIPBackend(GPUTarget("hip", "gfx942", 64))


def spec_key(arg, align=True, backend=HIPBackend):
    """The specialization key for a tensor argument.

    Asserts that the native specializer and the backend's own
    get_tensor_specialization() agree: the latter is the reference the former has
    to reproduce, and it is what runs when the native path declines the argument.
    """
    key = native_specialize_impl(backend, arg, False, True, align)[1]
    assert key == backend.get_tensor_specialization(arg, align=align)
    return key


def hip_key(arg, align=True, backend=HIPBackend):
    return spec_key(arg, align=align, backend=backend)


class RangedTensor(MockTensor):
    """A non-torch tensor-like that declares its own pointer range."""

    def __init__(self, data_ptr, ptr_range):
        super().__init__(torch.float32, (16, ))
        self._data_ptr = data_ptr
        self._ptr_range = ptr_range

    def data_ptr(self):
        return self._data_ptr

    def ptr_range(self):
        return self._ptr_range


def make_arg(storage_size, offset, view_size, ptr_range):
    """An argument with the requested storage layout, or pointer range.

    `storage_size is None` selects a non-torch object that answers the range
    question itself through `ptr_range()`; otherwise the argument is a view into
    an int8 meta tensor, whose element count is its size in bytes and which
    allocates nothing.
    """
    if storage_size is None:
        return RangedTensor(offset, ptr_range)
    base = torch.empty(storage_size, dtype=torch.int8, device="meta")
    view = base[offset:offset + view_size]
    assert view.untyped_storage().size() == storage_size
    assert view.data_ptr() == offset
    return view


@pytest.mark.parametrize(
    "storage_size, offset, view_size, ptr_range, within",
    [
        # The range is the whole allocation's, and the threshold is inclusive:
        # MAX_INT_32 bytes still fits in a 32-bit offset, one more does not.
        (MAX_INT_32 - 1, 0, MAX_INT_32 - 1, None, True),
        (MAX_INT_32, 0, MAX_INT_32, None, True),
        (MAX_INT_32 + 1, 0, MAX_INT_32 + 1, None, False),
        # It is the *storage* that counts, not the view. A small view of an
        # oversized allocation must not be promoted; using numel * itemsize here
        # would report 16 bytes and silently enable 32-bit addressing.
        (MAX_INT_32 + 1, 0, 16, None, False),
        # ... and a view of a small allocation keeps the range whatever its own
        # size or alignment.
        (4000, 40, 40, None, True),
        (4000, 64, 64, None, True),
        # An object that declares a pointer range answers for itself.
        (None, 0, None, 0, True),
        (None, 0, None, MAX_INT_32, True),
        (None, 0, None, MAX_INT_32 + 1, False),
        (None, 8, None, MAX_INT_32, True),
    ],
)
@pytest.mark.parametrize("align", [True, False])
@pytest.mark.parametrize("use_buffer_ops", [True, False])
def test_hip_tensor_specialization(storage_size, offset, view_size, ptr_range, within, align, use_buffer_ops,
                                   fresh_knobs):
    # The key states what is true of the argument, so `use_buffer_ops` does not
    # enter into it -- acting on the range is a codegen decision taken in
    # make_ttgir. The two bits are independent, so align=False drops "D" and
    # keeps "S".
    fresh_knobs.amd.use_buffer_ops = use_buffer_ops
    arg = make_arg(storage_size, offset, view_size, ptr_range)
    expected = ("D" if align and offset % 16 == 0 else "") + ("S" if within else "")
    assert hip_key(arg, align=align) == expected
    assert HIPBackend.is_within_2gb(arg) is within


def subclass_with_data_ptr(value):
    """A torch.Tensor subclass whose data_ptr() is overridden in Python."""

    class Overridden(torch.Tensor):

        def data_ptr(self):
            return value

    return small_meta_tensor().as_subclass(Overridden)


def plain_with_ptr_range(value):
    """A non-torch object carrying ptr_range on the instance, not on the type."""

    class Plain:
        dtype = torch.float32

        def data_ptr(self):
            return 0

    arg = Plain()
    arg.ptr_range = lambda: value
    return arg


@pytest.mark.parametrize("make_arg, expected", [
    (lambda: subclass_with_data_ptr(8), "S"),
    (lambda: subclass_with_data_ptr(0), "DS"),
    (lambda: plain_with_ptr_range(MAX_INT_32), "DS"),
    (lambda: plain_with_ptr_range(MAX_INT_32 + 1), "D"),
])
def test_hip_specialization_honors_python_overrides(make_arg, expected):
    # Both bits can be answered in Python, and the native path has to let them be:
    # the torch stable API reads the TensorImpl directly, so it is used only for an
    # exact torch.Tensor and a subclass overriding data_ptr() still decides "D";
    # ptr_range is a duck-typed escape hatch, so it is looked up on the instance
    # rather than the type and still decides "S".
    arg = make_arg()
    assert hip_key(arg) == expected
    # CUDA asks only for the alignment bit, decided by the same override.
    assert spec_key(arg, backend=CUDABackend) == ("D" if "D" in expected else "")


def test_native_tensor_spec_modes():
    # Mode 1 produces only "D"; mode 2 adds "S".
    def backend_with(mode):
        return type("Modal", (), {"supports_native_tensor_specialization": mode})

    small, big = small_meta_tensor(), over_2gb_meta_tensor()
    assert native_specialize_impl(backend_with(NATIVE_TENSOR_SPEC_ALIGN), small, False, True, True)[1] == "D"
    assert native_specialize_impl(backend_with(NATIVE_TENSOR_SPEC_ALIGN), small, False, True, False)[1] == ""
    assert native_specialize_impl(backend_with(NATIVE_TENSOR_SPEC_RANGE), small, False, True, True)[1] == "DS"
    assert native_specialize_impl(backend_with(NATIVE_TENSOR_SPEC_RANGE), small, False, True, False)[1] == "S"
    assert native_specialize_impl(backend_with(NATIVE_TENSOR_SPEC_RANGE), big, False, True, True)[1] == "D"


def test_cuda_backend_has_no_extra_specialization_bit():
    # The range bit is opt-in; CUDA asks only for the alignment bit.
    assert spec_key(small_meta_tensor(), backend=CUDABackend) == "D"
    assert spec_key(small_meta_tensor(), align=False, backend=CUDABackend) == ""


def test_hip_tensor_specialization_after_late_torch_import():
    script = textwrap.dedent("""
        import sys

        from triton._C.libtriton import native_specialize_impl
        from triton.backends.amd.compiler import HIPBackend

        assert "torch" not in sys.modules
        native_specialize_impl(HIPBackend, 0, False, True, True)
        assert "torch" not in sys.modules

        import torch

        tensor = torch.empty(1024, dtype=torch.float32, device="meta")
        native = native_specialize_impl(HIPBackend, tensor, False, True, True)[1]
        reference = HIPBackend.get_tensor_specialization(tensor, align=True)
        assert native == reference == "DS", (native, reference)
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
