from typing import Any
from triton._C.libproton import proton as libproton
import triton.runtime.driver as driver
import triton.language as tl
import triton
from triton import MockTensor
from .state import exit_state, enter_state, COMPUTE_METADATA_SCOPE_NAME


@triton.jit
def tensor_metric_kernel(device_ptr, device_offset_ptr, size: tl.uint64, metric_id: tl.uint64, metric_value_ptr,
                         metric_value_size: tl.uint64):
    BLOCK_SIZE: tl.constexpr = 256
    device_offset = tl.load(device_offset_ptr)
    tl.store(device_ptr + device_offset, metric_id)
    device_offset = (device_offset + 1) % size
    num_iters = tl.cdiv(metric_value_size, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_iters):
        cur_offsets = offsets + i * BLOCK_SIZE
        mask = cur_offsets < metric_value_size
        metric_value = tl.load(metric_value_ptr + cur_offsets, mask=mask)
        tl.store(device_ptr + (device_offset + cur_offsets) % size, metric_value, mask=mask)
    tl.debug_barrier()
    device_offset = (device_offset + metric_value_size) % size
    tl.store(device_offset_ptr, device_offset)


@triton.jit
def scalar_metric_kernel(device_ptr, device_offset_ptr, size: tl.uint64, metric_id: tl.uint64, metric_value: tl.uint64):
    device_offset = tl.load(device_offset_ptr)
    tl.store(device_ptr + device_offset, metric_id)
    device_offset = (device_offset + 1) % size
    tl.store(device_ptr + device_offset, metric_value)
    device_offset = (device_offset + 1) % size
    tl.debug_barrier()
    tl.store(device_offset_ptr, device_offset)


def _get_kernel(kernel_fn, *args):
    kernel = kernel_fn.warmup(*args, grid=(1, ), num_warps=1)
    kernel._init_handles()
    return kernel.function


def set_metric_kernels():
    mock_ptr = MockTensor(tl.uint64)
    mock_metric_id = 0
    mock_size = 1
    mock_metric_value_size = 1
    tensor_metric_kernel_fn = _get_kernel(
        tensor_metric_kernel,
        mock_ptr,
        mock_ptr,
        mock_size,
        mock_metric_id,
        mock_ptr,
        mock_metric_value_size,
    )
    scalar_metric_kernel_fn = _get_kernel(
        scalar_metric_kernel,
        mock_ptr,
        mock_ptr,
        mock_size,
        mock_metric_id,
        mock_metric_id,
    )
    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)
    libproton.set_metric_kernels(tensor_metric_kernel_fn, scalar_metric_kernel_fn, stream)


class _TensorMetric(libproton.TensorMetric):

    def __init__(self, value, metric_type_index):
        super().__init__(value.data_ptr(), metric_type_index, value.numel())
        # Hold a reference to the backing tensor so its device memory stays alive.
        self._value = value


def transform_tensor_metrics(metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, libproton.TensorMetric]]:
    tensor_metrics = {}
    scalar_metrics: dict[str, Any] = {}
    for key, value in metrics.items():
        if hasattr(value, "data_ptr"):  # tensor
            if value.device.type == "cpu":
                scalar_metrics[key] = value
            else:  # device tensor
                enter_state(COMPUTE_METADATA_SCOPE_NAME)
                # implicit casting to double or int64 tensors
                if value.is_floating_point():
                    value = value.double()
                    if value.numel() > 1:
                        metric_index = libproton.metric_type_vector_double_index
                    else:
                        metric_index = libproton.metric_type_double_index
                else:
                    value = value.long()
                    if value.numel() > 1:
                        metric_index = libproton.metric_type_vector_int64_index
                    else:
                        metric_index = libproton.metric_type_int64_index
                exit_state()
                tensor_metrics[key] = _TensorMetric(value, metric_index)
        else:
            scalar_metrics[key] = value
    return scalar_metrics, tensor_metrics
