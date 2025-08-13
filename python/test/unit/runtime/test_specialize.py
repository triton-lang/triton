import pytest
import torch

from triton.runtime.driver import driver
from triton.tools.tensor_descriptor import TensorDescriptor
from triton.compiler import make_backend
from triton.runtime.jit import create_specialize_fallback
from triton._C.libtriton import native_specialize_impl

@pytest.mark.parametrize("arg", [
    -1,
    0,
    1,
    16,
    -16,
    32,
    -32,
    33,
    -33,
    1024,
    -1024,
    2**63,
    -2**63,
    2**63 + 1,
    2**63 - 1,
    2**64 - 1,
    False,
    True,
    None,
    (False, True),
    (None, 1),
    (1, False, None),
    (1, 2.0, False, None),
    0.0,
    1.0,
    2.0,
])
@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("is_const", [False, True])
@pytest.mark.parametrize("specialize_value", [False, True])
def test_specialize_base_arg(arg, align, is_const, specialize_value):
    target = driver.active.get_current_target()
    backend = make_backend(target)

    fallback_result = create_specialize_fallback()(backend, arg, is_const, specialize_value, align)
    native_result = native_specialize_impl(backend, arg, is_const, specialize_value, align)

    assert str(native_result) == str(fallback_result)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.float32, torch.float16])
@pytest.mark.parametrize("size", [(1, ), (32, ), (32, 32), (32, 32, 32)])
@pytest.mark.parametrize("make_desc", [False, True])
@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("is_const", [False, True])
@pytest.mark.parametrize("specialize_value", [False, True])
def test_specialize_tensor(dtype, size, make_desc, align, is_const, specialize_value):
    device = driver.active.get_active_torch_device()
    backend = make_backend(driver.active.get_current_target())

    arg = torch.empty(size, dtype=dtype, device=device)
    if make_desc:
        arg = TensorDescriptor.from_tensor(arg, block_shape=[1] * len(size))

    fallback_result = create_specialize_fallback()(backend, arg, is_const, specialize_value, align)
    native_result = native_specialize_impl(backend, arg, is_const, specialize_value, align)

    assert str(native_result) == str(fallback_result)
