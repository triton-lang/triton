from typing import Any
from triton._C.libproton import proton as libproton
import triton.runtime.driver as driver
import triton.language as tl
import triton
from triton import MockTensor


@triton.jit
def tensor_metric_kernel(device_ptr, device_offset_ptr, size: tl.uint64, seq_id: tl.uint64, metric_value_ptr,
                         metric_value_size: tl.uint64):
    # Record layout is {seq_id, <metric_values>}.
    BLOCK_SIZE: tl.constexpr = 128
    record_size = metric_value_size + 1
    # Reserve the full record atomically so replayed graph streams can append
    # concurrently.
    device_offset = tl.atomic_add(device_offset_ptr, record_size, sem="relaxed") % size
    tl.store(device_ptr + device_offset, seq_id)
    device_offset = (device_offset + 1) % size
    num_iters = tl.cdiv(metric_value_size, BLOCK_SIZE)
    offsets = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_iters):
        cur_offsets = offsets + i * BLOCK_SIZE
        mask = cur_offsets < metric_value_size
        metric_value = tl.load(metric_value_ptr + cur_offsets, mask=mask)
        tl.store(device_ptr + (device_offset + cur_offsets) % size, metric_value, mask=mask)


@triton.jit
def scalar_metric_kernel(device_ptr, device_offset_ptr, size: tl.uint64, seq_id: tl.uint64, metric_value: tl.uint64):
    # Record layout is {seq_id, metric_value}.
    device_offset = tl.atomic_add(device_offset_ptr, 2, sem="relaxed") % size
    tl.store(device_ptr + device_offset, seq_id)
    device_offset = (device_offset + 1) % size
    tl.store(device_ptr + device_offset, metric_value)


def _get_kernel(kernel_fn, *args):
    kernel = kernel_fn.warmup(*args, grid=(1, ), num_warps=1)
    kernel._init_handles()
    target = getattr(kernel.metadata, "target", None)
    warp_size = getattr(target, "warp_size", None)
    if warp_size is None:
        warp_size = driver.active.get_current_target().warp_size
    num_threads = kernel.metadata.num_warps * warp_size
    return kernel.function, num_threads, kernel.metadata.shared


def set_metric_kernels():
    mock_ptr = MockTensor(tl.uint64)
    mock_seq_id = 0
    mock_size = 1
    mock_metric_value_size = 1
    tensor_metric_kernel_fn, tensor_metric_kernel_num_threads, tensor_metric_kernel_shared = _get_kernel(
        tensor_metric_kernel,
        mock_ptr,
        mock_ptr,
        mock_size,
        mock_seq_id,
        mock_ptr,
        mock_metric_value_size,
    )
    scalar_metric_kernel_fn, scalar_metric_kernel_num_threads, scalar_metric_kernel_shared = _get_kernel(
        scalar_metric_kernel,
        mock_ptr,
        mock_ptr,
        mock_size,
        mock_seq_id,
        mock_metric_value_size,
    )
    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)
    libproton.set_metric_kernels(
        tensor_metric_kernel_fn,
        scalar_metric_kernel_fn,
        stream,
        tensor_metric_kernel_num_threads,
        tensor_metric_kernel_shared,
        scalar_metric_kernel_num_threads,
        scalar_metric_kernel_shared,
    )


class _TensorMetric(libproton.TensorMetric):

    def __init__(self, value, metric_type_index):
        super().__init__(value.data_ptr(), metric_type_index, value.numel())
        # Hold a reference to the backing tensor so its device memory stays alive.
        self._value = value


FLOPS_WIDTHS = (8, 16, 32, 64)
FLOPS_DTYPES = {
    **{"flops": (float, libproton.metric_type_double_index, libproton.metric_type_vector_double_index)},
    **{
        f"flops{width}": (float, libproton.metric_type_double_index, libproton.metric_type_vector_double_index)
        for width in FLOPS_WIDTHS
    },
}
ROOFLINE_DTYPES = {
    **FLOPS_DTYPES,
    "bytes": (int, libproton.metric_type_int64_index, libproton.metric_type_vector_int64_index),
}


def _scalar_or_vector_value(value: Any, convert: type) -> Any:
    if hasattr(value, "data_ptr"):
        value = value.tolist() if value.numel() > 1 else value.item()
    if isinstance(value, (list, tuple)):
        return [convert(v) for v in value]
    return convert(value)


def _scalar_metric_value(key: str, value: Any) -> Any:
    # Proton's built-in roofline metrics should have stable dtypes regardless of
    # whether launch_metadata produced a Python scalar or a device tensor.
    if key in ROOFLINE_DTYPES:
        convert, _, _ = ROOFLINE_DTYPES[key]
        return _scalar_or_vector_value(value, convert)
    return value


def _tensor_metric_value_and_index(key: str, value: Any) -> tuple[Any, int]:
    if key in ROOFLINE_DTYPES:
        _, scalar_index, vector_index = ROOFLINE_DTYPES[key]
        value = value.long() if key == "bytes" else value.double()
        return value, vector_index if value.numel() > 1 else scalar_index

    # implicit casting to double or int64 tensors
    if value.is_floating_point():
        value = value.double()
        if value.numel() > 1:
            return value, libproton.metric_type_vector_double_index
        return value, libproton.metric_type_double_index

    value = value.long()
    if value.numel() > 1:
        return value, libproton.metric_type_vector_int64_index
    return value, libproton.metric_type_int64_index


def transform_tensor_metrics(metrics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, libproton.TensorMetric]]:
    tensor_metrics = {}
    scalar_metrics: dict[str, Any] = {}
    for key, value in metrics.items():
        if hasattr(value, "data_ptr"):  # tensor
            if value.device.type == "cpu":
                scalar_metrics[key] = _scalar_metric_value(key, value)
            else:  # device tensor
                value, metric_index = _tensor_metric_value_and_index(key, value)
                tensor_metrics[key] = _TensorMetric(value, metric_index)
        else:
            scalar_metrics[key] = _scalar_metric_value(key, value)
    return scalar_metrics, tensor_metrics
