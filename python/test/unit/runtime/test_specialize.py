import inspect
import numpy
import pytest
import torch
from collections import namedtuple
from triton._C.libtriton import native_specialize_impl, native_specialize_impl_batched
from triton.runtime.jit import MockTensor, JITCallable, KernelParam, create_function_from_signature
from triton._utils import canonicalize_dtype
from triton.backends.nvidia.compiler import CUDABackend
from triton.backends.amd.compiler import HIPBackend
from triton.language import constexpr
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


def reference_specialize_impl_batched(backend, args, constexpr_flags, is_consts, specialize_values, aligns,
                                      override_types, return_keys):
    result = []
    for arg, is_constexpr, is_const, specialize_value, align, override_type, return_key in zip(
            args, constexpr_flags, is_consts, specialize_values, aligns, override_types, return_keys):
        if is_constexpr:
            result.append(("constexpr", arg))
            continue
        ty, key = reference_specialize_impl(backend, arg, is_const, specialize_value, align)
        if override_type is not None:
            result.append((override_type, key if return_key else None))
        else:
            result.append((ty, key))
    return result


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


@pytest.mark.parametrize("backend", [CUDABackend, HIPBackend])
def test_specialize_impl_batched(backend):
    args = (
        17,
        16,
        torch.empty((16, ), dtype=torch.float32, device="cpu"),
        torch.empty((15, ), dtype=torch.float32, device="cpu"),
        False,
        (16, False),
    )
    constexpr_flags = (True, False, False, False, False, False)
    is_consts = (False, False, False, False, False, False)
    specialize_values = (True, True, True, True, True, True)
    aligns = (True, False, True, True, True, True)
    override_types = (None, "i32", "*fp32", "fp16", "u1", None)
    return_keys = (True, True, True, False, False, True)

    result = native_specialize_impl_batched(backend, args, constexpr_flags, is_consts, specialize_values, aligns,
                                            override_types, return_keys)
    expected = reference_specialize_impl_batched(backend, args, constexpr_flags, is_consts, specialize_values, aligns,
                                                 override_types, return_keys)
    assert result == expected


@pytest.mark.parametrize("backend", [CUDABackend, HIPBackend])
def test_create_function_from_signature_uses_batched_specialization(backend):
    fp16 = "fp16"
    i32 = "i32"
    u1 = "u1"

    def kernel(x, y: fp16, z: constexpr, w: i32 = 16, flag: u1 = False, align_only=32):
        pass

    sig = inspect.signature(kernel)
    params = list(sig.parameters.values())
    kparams = [
        KernelParam(0, params[0], False, False),
        KernelParam(1, params[1], False, False),
        KernelParam(2, params[2], False, False),
        KernelParam(3, params[3], True, False),
        KernelParam(4, params[4], False, False),
        KernelParam(5, params[5], False, True),
    ]

    binder = create_function_from_signature(sig, kparams, backend)
    tensor = torch.empty((16, ), dtype=torch.float32, device="cpu")
    bound_args, specialization, options = binder(tensor, 1.0, 7, num_warps=4)

    expected_args = (tensor, 1.0, 7, 16, False, 32)
    expected = reference_specialize_impl_batched(
        backend,
        expected_args,
        tuple(kp.is_constexpr for kp in kparams),
        tuple(kp.is_const for kp in kparams),
        tuple(not kp.do_not_specialize for kp in kparams),
        tuple(not kp.do_not_specialize_on_alignment for kp in kparams),
        tuple(kp.annotation_type or None for kp in kparams),
        tuple(not (kp.annotation_type == "u1" or kp.annotation_type[:2] in ["fp", "bf"]) if kp.annotation_type else True
              for kp in kparams),
    )

    assert bound_args == {
        "x": tensor,
        "y": 1.0,
        "z": 7,
        "w": 16,
        "flag": False,
        "align_only": 32,
    }
    assert specialization == expected
    assert options == {"num_warps": 4}
