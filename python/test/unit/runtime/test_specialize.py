import inspect
import numpy
import pytest
import torch
import triton
import triton.language as tl
from collections import namedtuple
from triton._C.libtriton import native_specialize_impl
from triton.runtime.jit import MockTensor, JITCallable, KernelParam, _normalize_ty, create_function_from_signature
from triton._utils import canonicalize_dtype, get_primitive_bitwidth
from triton.backends.nvidia.compiler import CUDABackend
from triton.backends.amd.compiler import HIPBackend
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


@pytest.mark.parametrize("torch_name,triton_name,canonical", [
    ("float4_e2m1fn_x2", "float4e2m1x2", "fp4e2m1x2"),
    ("float8_e8m0fnu", "float8e8m0fnu", "fp8e8m0fnu"),
])
def test_low_precision_dtype_mapping(torch_name, triton_name, canonical):
    torch_dtype = getattr(torch, torch_name, None)
    if torch_dtype is None:
        pytest.skip(f"Torch does not expose {torch_name}")
    triton_dtype = getattr(tl, triton_name)
    assert canonicalize_dtype(torch_dtype) == canonical
    assert canonicalize_dtype(triton_dtype) == canonical
    assert get_primitive_bitwidth(canonical) == 8
    assert triton_dtype.itemsize == torch_dtype.itemsize == 1
    for alias in (canonical, torch_name, triton_name, "tl." + triton_name, triton_dtype):
        assert _normalize_ty(alias) == canonical
    assert _normalize_ty("const " + triton_name + "*") == "*k" + canonical


@pytest.mark.parametrize("backend", [CUDABackend, HIPBackend])
def test_low_precision_specialization_identity(backend):
    names = ("uint8", "int8", "float4_e2m1fn_x2", "float8_e8m0fnu")
    if not all(hasattr(torch, name) for name in names):
        pytest.skip("Torch does not expose the native low precision types")
    storage = torch.arange(32, dtype=torch.uint8)
    tensors = [storage.view(getattr(torch, name)) for name in names]
    expected = ("u8", "i8", "fp4e2m1x2", "fp8e8m0fnu")
    # Repeatedly interleave equal-width views to exercise the dtype cache.
    for indices in (range(4), reversed(range(4))):
        for i in indices:
            for is_const in (False, True):
                signature, _ = native_specialize_impl(backend, tensors[i], is_const, True, True)
                assert signature == ("*k" if is_const else "*") + expected[i]


def _bind_annotated_argument(annotation, arg):
    param = inspect.Parameter("arg", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation)
    binder = create_function_from_signature(inspect.Signature([param]), [KernelParam(0, param, False, False)],
                                            CUDABackend)
    return binder(arg)[1][0][0]


@pytest.mark.parametrize("name", ["fp4e2m1x2", "fp8e8m0fnu"])
def test_low_precision_annotation_type_safety(name):
    dtype = tl.dtype(name)
    storage = torch.empty(32, dtype=torch.uint8)
    native = triton.reinterpret(storage, dtype)
    pointer = "*" + name
    assert _bind_annotated_argument(pointer, native) == pointer
    assert _bind_annotated_argument("const " + name + "*", native) == "*k" + name
    assert _bind_annotated_argument("*u8", triton.reinterpret(native, tl.uint8)) == "*u8"
    for annotation in ("*u8", "*fp4e2m1x2" if name != "fp4e2m1x2" else "*fp8e8m0fnu", "fp32", "u1", "i32"):
        for value in (native, (native, ), TensorDescriptor.from_tensor(native, [32])):
            with pytest.raises(TypeError, match="does not match argument dtype"):
                _bind_annotated_argument(annotation, value)
    for value in (storage, 0, 1, constexpr(1)):
        with pytest.raises(TypeError, match="does not match argument dtype"):
            _bind_annotated_argument(pointer, value)


def test_low_precision_annotations_preserve_legacy_behavior():
    legacy = torch.empty(32, dtype=torch.int8)
    assert _bind_annotated_argument("*u8", legacy) == "*u8"
    for annotation in ("fp32", "u1"):
        for value in (numpy.float32(1), object(), legacy):
            assert _bind_annotated_argument(annotation, value) == annotation
    native = triton.reinterpret(legacy, tl.float4e2m1x2)
    for annotation in ("my_types.float4Tensor", "float8e8_config", "fp4_not_a_dtype"):
        assert _bind_annotated_argument(annotation, legacy) == "*i8"
        assert _bind_annotated_argument(annotation, native) == "*fp4e2m1x2"
