from unittest.mock import Mock, call

import pytest
from triton_kernels import target_info
from triton_kernels.roofline import get_memset_tbps, get_blas_tflops
from triton_kernels.target_info import cuda_capability_geq, is_cuda


def test_num_sms_cached_per_driver_and_device(monkeypatch):
    monkeypatch.delenv("CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING", raising=False)
    driver = Mock()
    driver.get_current_device.return_value = 0
    driver.utils.get_device_properties.side_effect = lambda device: {"multiprocessor_count": (80, 132)[device]}
    driver_config = Mock(active=driver)
    monkeypatch.setattr(target_info.triton.runtime, "driver", driver_config)

    assert target_info.num_sms() == 80
    assert target_info.num_sms() == 80
    driver.utils.get_device_properties.assert_called_once_with(0)
    driver.get_current_device.return_value = 1
    assert target_info.num_sms() == 132
    assert target_info.num_sms() == 132
    driver.get_current_device.return_value = 0
    assert target_info.num_sms() == 80
    assert driver.utils.get_device_properties.call_args_list == [call(0), call(1)]

    other_driver = Mock()
    other_driver.get_current_device.return_value = 0
    other_driver.utils.get_device_properties.return_value = {"multiprocessor_count": 64}
    driver_config.active = other_driver
    assert target_info.num_sms() == 64
    assert target_info.num_sms() == 64
    other_driver.utils.get_device_properties.assert_called_once_with(0)


def test_num_sms_retries_failed_query(monkeypatch):
    monkeypatch.delenv("CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING", raising=False)
    driver = Mock()
    driver.get_current_device.return_value = 0
    driver.utils.get_device_properties.side_effect = [RuntimeError("query failed"), {"multiprocessor_count": 80}]
    monkeypatch.setattr(target_info.triton.runtime, "driver", Mock(active=driver))

    with pytest.raises(RuntimeError, match="query failed"):
        target_info.num_sms()
    assert target_info.num_sms() == 80
    assert target_info.num_sms() == 80
    assert driver.utils.get_device_properties.call_args_list == [call(0), call(0)]


def test_num_sms_does_not_cache_per_context_mps(monkeypatch):
    monkeypatch.setenv("CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING", "1")
    driver = Mock()
    driver.get_current_device.return_value = 0
    driver.utils.get_device_properties.side_effect = [{"multiprocessor_count": 80}, {"multiprocessor_count": 40}]
    monkeypatch.setattr(target_info.triton.runtime, "driver", Mock(active=driver))

    assert target_info.num_sms() == 80
    assert target_info.num_sms() == 40
    assert driver.utils.get_device_properties.call_args_list == [call(0), call(0)]


def test_get_memset_tbps():
    tbps = get_memset_tbps()
    assert tbps > 0


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp8"])
def test_get_blas_tflops(dtype):
    if dtype in ["fp8"] and is_cuda() and not cuda_capability_geq(9, 0):
        pytest.skip("FP8 not supported on this GPU")
    tflops = get_blas_tflops(dtype)
    assert tflops > 0
