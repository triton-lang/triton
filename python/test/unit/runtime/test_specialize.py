import inspect
from collections import namedtuple

import numpy
import pytest
import torch
import triton.language as tl
from triton._C.libtriton import native_specialize_impl
from triton._utils import canonicalize_dtype, get_primitive_bitwidth
from triton.backends.amd.compiler import HIPBackend
from triton.backends.nvidia.compiler import CUDABackend
from triton.experimental.gluon.amd.cdna5 import TensorDescriptor as AMDTensorDescriptor
from triton.experimental.gluon.language._layouts import NVMMASharedLayout, SwizzledSharedLayout
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor as GluonTensorDescriptor, TensorDescriptorIm2Col
from triton.language import constexpr, target_info
from triton.runtime.jit import (
    JITCallable,
    JITFunction,
    KernelParam,
    MockTensor,
    _normalize_ty,
    create_function_from_signature,
    reinterpret,
)
from triton.tools.tensor_descriptor import TensorDescriptor


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


@pytest.mark.parametrize("torch_name, triton_name, canonical", [
    ("float4_e2m1fn_x2", "float4e2m1x2", "fp4e2m1x2"),
    ("float8_e8m0fnu", "float8e8m0fnu", "fp8e8m0fnu"),
    ("uint8", "uint8", "u8"),
    ("int8", "int8", "i8"),
    ("float8_e4m3fn", "float8e4nv", "fp8e4nv"),
    ("float8_e5m2", "float8e5", "fp8e5"),
])
def test_low_precision_dtype_translation(torch_name, triton_name, canonical):
    torch_dtype = getattr(torch, torch_name, None)
    if torch_dtype is None:
        pytest.skip(f"PyTorch does not provide {torch_name}")
    dtype = getattr(tl, triton_name)
    assert canonicalize_dtype(torch_dtype) == canonical
    assert canonicalize_dtype(dtype) == canonical
    assert get_primitive_bitwidth(canonical) == dtype.primitive_bitwidth == 8
    assert dtype.itemsize == torch_dtype.itemsize == 1
    assert tl.str_to_ty(canonical, None) == dtype
    assert tl.str_to_ty(f"*{canonical}", None).element_ty == dtype
    for annotation in (triton_name, torch_name, canonical, f"tl.{triton_name}", dtype):
        assert _normalize_ty(annotation) == canonical
    for annotation in (f"torch.{torch_name}", torch_dtype):
        expected = canonical if dtype.is_low_precision_storage() else f"torch.{torch_name}"
        assert _normalize_ty(annotation) == expected
    for annotation in (f"{triton_name}*", f"*{triton_name}", tl.pointer_type(dtype)):
        assert _normalize_ty(annotation) == f"*{canonical}"
    assert _normalize_ty(f"const {triton_name}*") == f"*k{canonical}"


@pytest.mark.parametrize("annotation, expected", [
    (torch.float16, "torch.float16"),
    ("torch.float16", "torch.float16"),
    ("'tl.float16'", "'tl.float16'"),
    (repr(repr("tl.float16")), repr(repr("tl.float16"))),
    ("'*tl.float16'", "'*tl.float16'"),
    ("*kfloat16", "*kfloat16"),
    ("*kfp16", "*kfp16"),
    (tl.pointer_type(tl.float32, const=True), "*fp32"),
    ("uint8_t", "u8"),
    ("const uint8*", "*ku8"),
])
def test_low_precision_annotations_preserve_legacy_normalization(annotation, expected):
    assert _normalize_ty(annotation) == expected


@pytest.mark.parametrize("annotation, expected", [
    (torch.float16, "*fp16"),
    ("torch.float16", "*fp16"),
    ("'tl.float16'", "*fp16"),
    ("*kfloat16", "*kfp16"),
])
def test_low_precision_annotations_preserve_legacy_tensor_inference(annotation, expected):
    argument = torch.empty(16, dtype=torch.float16)
    assert _bind_typed_argument(annotation, argument) == expected


@pytest.mark.parametrize("annotation", ["fp32", "u1"])
@pytest.mark.parametrize("dtype", [None, "custom_dtype", torch.float16])
def test_low_precision_annotations_preserve_legacy_scalar_bypass(annotation, dtype):

    class LegacyArgument:

        def data_ptr(self):
            return 0

    argument = LegacyArgument()
    if dtype is not None:
        argument.dtype = dtype
    assert _bind_typed_argument(annotation, argument) == annotation


@pytest.mark.parametrize("backend", [CUDABackend, HIPBackend])
@pytest.mark.parametrize("is_const", [False, True])
@pytest.mark.parametrize("kind", ["tensor", "descriptor", "gluon_descriptor"])
def test_low_precision_specialization_preserves_dtype(backend, is_const, kind):
    torch_names = ["uint8", "float4_e2m1fn_x2", "float8_e8m0fnu"]
    if any(not hasattr(torch, name) for name in torch_names):
        pytest.skip("PyTorch does not provide the native packed FP4 and E8M0 types")
    canonical_names = ["u8", "fp4e2m1x2", "fp8e8m0fnu"]
    storage = torch.arange(256, dtype=torch.uint8).reshape(16, 16)
    arguments = []
    for name in torch_names:
        value = storage.view(getattr(torch, name))
        assert value.shape == storage.shape
        assert value.stride() == storage.stride()
        if kind == "descriptor":
            value = TensorDescriptor.from_tensor(value, [16, 16])
        elif kind == "gluon_descriptor":
            value = GluonTensorDescriptor.from_tensor(value, [16, 16], NVMMASharedLayout(0, 8, 2))
        arguments.append(value)

    # Revisit the same storage in reverse order to exercise cached dtype lookups.
    expected_types = []
    for values, names in ((arguments, canonical_names), (arguments[::-1], canonical_names[::-1])):
        for value, name in zip(values, names):
            actual, key = native_specialize_impl(backend, value, is_const, False, False)
            if kind == "tensor":
                expected = ("*k" if is_const else "*") + name
            elif kind == "descriptor":
                expected = f"tensordesc<{name}[16, 16]>"
            else:
                expected = f"tensordesc<{name}[16, 16],{value.layout!r}>"
            assert actual == expected
            assert key is None
            expected_types.append(actual)
    assert len(set(expected_types)) == 3
    dtypes = [tl.uint8, tl.float4e2m1x2, tl.float8e8m0fnu]
    assert len(set(dtypes)) == 3
    assert len({dtype.mangle() for dtype in dtypes}) == 3


def _bind_typed_argument(annotation, value, specialize=True):
    parameter = inspect.Parameter("value", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation)
    signature = inspect.Signature([parameter])
    parameters = [KernelParam(0, parameter, do_not_specialize=not specialize, do_not_specialize_on_alignment=False)]
    binder = create_function_from_signature(signature, parameters, CUDABackend)
    return binder(value)[1][0][0]


@pytest.mark.parametrize("annotation", ["*u8", "*fp4e2m1x2", "*fp8e8m0fnu"])
@pytest.mark.parametrize("torch_name, canonical", [("uint8", "u8"), ("float4_e2m1fn_x2", "fp4e2m1x2"),
                                                   ("float8_e8m0fnu", "fp8e8m0fnu")])
@pytest.mark.parametrize("specialize", [False, True])
def test_low_precision_pointer_annotations_reject_implicit_reinterpretation(annotation, torch_name, canonical,
                                                                            specialize):
    if not hasattr(torch, torch_name):
        pytest.skip(f"PyTorch does not provide {torch_name}")
    argument = torch.empty(16, dtype=getattr(torch, torch_name))
    if annotation == f"*{canonical}":
        assert _bind_typed_argument(annotation, argument, specialize) == annotation
    else:
        with pytest.raises(TypeError, match="does not match argument dtype"):
            _bind_typed_argument(annotation, argument, specialize)
        explicit = reinterpret(argument, tl.str_to_ty(annotation[1:], None))
        assert _bind_typed_argument(annotation, explicit, specialize) == annotation


@pytest.mark.parametrize("annotation", ["fp4e2m1x2", "fp8e8m0fnu", "*fp4e2m1x2", "*fp8e8m0fnu"])
@pytest.mark.parametrize("value", [0, 1.0])
def test_low_precision_scalar_annotations_reject_numeric_values(annotation, value):
    with pytest.raises(TypeError, match="does not match argument dtype"):
        _bind_typed_argument(annotation, value)


@pytest.mark.parametrize("torch_name", ["float4_e2m1fn_x2", "float8_e8m0fnu"])
@pytest.mark.parametrize("annotation", ["i32", "fp32", "u1"])
@pytest.mark.parametrize("specialize", [False, True])
def test_low_precision_tensor_rejects_scalar_annotation(torch_name, annotation, specialize):
    if not hasattr(torch, torch_name):
        pytest.skip(f"PyTorch does not provide {torch_name}")
    argument = torch.empty(16, dtype=getattr(torch, torch_name))
    with pytest.raises(TypeError, match="does not match argument dtype"):
        _bind_typed_argument(annotation, argument, specialize)


@pytest.mark.parametrize("torch_name", ["uint8", "float4_e2m1fn_x2", "float8_e8m0fnu"])
@pytest.mark.parametrize("annotation", ["*u8", "i32", "fp32", "u1", "*fp4e2m1x2", "*fp8e8m0fnu"])
@pytest.mark.parametrize("kind",
                         ["tuple", "nested_tuple", "descriptor", "gluon_descriptor", "amd_descriptor", "im2col"])
def test_low_precision_container_annotations_cannot_override_native_dtype(torch_name, annotation, kind):
    if not hasattr(torch, torch_name):
        pytest.skip(f"PyTorch does not provide {torch_name}")
    tensor = torch.empty((16, 16), dtype=getattr(torch, torch_name))
    if kind == "tuple":
        argument = (tensor, )
    elif kind == "nested_tuple":
        argument = (1, (tensor, ))
    elif kind == "descriptor":
        argument = TensorDescriptor.from_tensor(tensor, [16, 16])
    elif kind == "gluon_descriptor":
        argument = GluonTensorDescriptor.from_tensor(tensor, [16, 16], NVMMASharedLayout(0, 8, 2))
    elif kind == "amd_descriptor":
        argument = AMDTensorDescriptor.from_tensor(tensor, [16, 16], SwizzledSharedLayout(1, 1, 1, [1, 0]))
    else:
        tensor = torch.empty((16, 16, 16), dtype=tensor.dtype)
        argument = TensorDescriptorIm2Col.from_tensor(tensor, [16, 16], NVMMASharedLayout(0, 8, 2),
                                                      pixel_box_lower_corner=[0], pixel_box_upper_corner=[0])
    if torch_name != "uint8" or annotation in ("*fp4e2m1x2", "*fp8e8m0fnu"):
        with pytest.raises(TypeError, match="does not match argument dtype"):
            _bind_typed_argument(annotation, argument)
    else:
        assert _bind_typed_argument(annotation, argument) == annotation


@pytest.mark.parametrize("annotation", ["fp4e2m1x2_t", "fp8e8m0fnu_t", "f_tloat4e2m1x2"])
@pytest.mark.parametrize("prefix", ["", "tl.", "torch.", "gl.", "ttgl.", "triton.language."])
def test_low_precision_annotations_reject_accidental_suffix_aliases(annotation, prefix):
    with pytest.raises(ValueError, match="Unknown low-precision storage dtype"):
        _normalize_ty(prefix + annotation)


@pytest.mark.parametrize("annotation", [
    "float4e2m1",
    "float8_e8m0",
    "mytypes.float4Tensor",
    "gl.float4Buffer",
    "fp8e8_config",
    "float8e8_config",
    "float8_e8_config",
    repr("mytypes.float4Tensor"),
    type("fp4_buffer", (), {}),
    type("float4Tensor", (), {}),
])
@pytest.mark.parametrize("torch_name, canonical", [("uint8", "u8"), ("float4_e2m1fn_x2", "fp4e2m1x2"),
                                                   ("float8_e8m0fnu", "fp8e8m0fnu")])
@pytest.mark.parametrize("specialize", [False, True])
def test_low_precision_unknown_annotations_preserve_jit_inference(annotation, torch_name, canonical, specialize):
    if not hasattr(torch, torch_name):
        pytest.skip(f"PyTorch does not provide {torch_name}")

    def kernel(source: "mytypes.float4Tensor", destination):  # noqa: F821 - A legacy forward annotation.
        tl.store(destination, tl.load(source))

    # Before native storage support, unrecognized hints were ignored and tensor
    # dtypes were inferred at the call site, including ordinary uint8 buffers.
    kernel.__annotations__["source"] = annotation
    compiled = JITFunction(kernel, do_not_specialize=[] if specialize else ["source"])
    assert compiled.params[0].annotation_type == ""
    expected_hint = annotation.__name__ if isinstance(annotation, type) else annotation
    assert compiled.params[0].annotation == expected_hint
    binder = create_function_from_signature(compiled.signature, compiled.params, CUDABackend)
    source = torch.empty(16, dtype=getattr(torch, torch_name))
    destination = torch.empty_like(source)
    _, signature, _ = binder(source, destination)
    assert [entry[0] for entry in signature] == [f"*{canonical}", f"*{canonical}"]


def test_low_precision_annotations_preserve_legacy_pointer_override():
    argument = torch.empty(16, dtype=torch.int8)
    assert _bind_typed_argument("*u8", argument) == "*u8"


def test_low_precision_preserves_fp8_type_predicate():
    for dtype in (tl.float8e4nv, tl.float8e5, tl.float8e4b8, tl.float8e5b16, tl.float8e4b15):
        assert dtype.is_fp8()
        assert tl.block_type(dtype, [16]).is_fp8()
        assert tl.pointer_type(dtype).is_fp8()
    for dtype in (tl.float4e2m1x2, tl.float8e8m0fnu):
        assert not dtype.is_fp8()
        assert not tl.block_type(dtype, [16]).is_fp8()
        assert not tl.pointer_type(dtype).is_fp8()


@pytest.mark.parametrize("triton_name, canonical", [("float4e2m1x2", "fp4e2m1x2"), ("float8e8m0fnu", "fp8e8m0fnu")])
@pytest.mark.parametrize("prefix", ["", "tl.", "gl.", "ttgl.", "triton.language."])
@pytest.mark.parametrize("quote_depth", [1, 3])
def test_low_precision_quoted_annotations_preserve_type(triton_name, canonical, prefix, quote_depth):
    annotation = f"*{prefix}{triton_name}"
    for _ in range(quote_depth):
        annotation = repr(annotation)
    assert _normalize_ty(annotation) == f"*{canonical}"
    assert _normalize_ty(repr(f"*k{prefix}{triton_name}")) == f"*k{canonical}"
    with pytest.raises(TypeError, match="does not match argument dtype"):
        _bind_typed_argument(annotation, torch.empty(16, dtype=torch.uint8))


@pytest.mark.parametrize("torch_name", ["float4_e2m1fn_x2", "float8_e8m0fnu"])
def test_low_precision_descriptor_rejects_nan_padding(torch_name):
    if not hasattr(torch, torch_name):
        pytest.skip(f"PyTorch does not provide {torch_name}")
    argument = torch.empty((16, 16), dtype=getattr(torch, torch_name))
    with pytest.raises(ValueError, match="Padding option `nan` is not supported for storage dtype"):
        TensorDescriptor.from_tensor(argument, [16, 16], padding="nan")
