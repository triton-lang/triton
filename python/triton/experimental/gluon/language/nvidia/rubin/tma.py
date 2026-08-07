from ..blackwell.tma import (
    async_gather,
    async_scatter,
    async_atomic_add,
    async_atomic_and,
    async_atomic_max,
    async_atomic_min,
    async_atomic_or,
    async_atomic_xor,
    async_load_im2col,
    async_store,
    make_tensor_descriptor,
    store_wait,
    tensor_descriptor,
    tensor_descriptor_type,
)
from ..hopper.tma import (
    _emit_alignment_check,
    tensor_descriptor_im2col,
    tensor_descriptor_im2col_type,
)
from ..._core import _unwrap_if_constexpr, builtin

__all__ = [
    "async_atomic_add",
    "async_atomic_and",
    "async_atomic_max",
    "async_atomic_min",
    "async_atomic_or",
    "async_atomic_xor",
    "async_gather",
    "async_load",
    "async_load_im2col",
    "async_scatter",
    "async_store",
    "store_wait",
    "tensor_descriptor",
    "tensor_descriptor_im2col",
    "tensor_descriptor_type",
    "tensor_descriptor_im2col_type",
    "make_tensor_descriptor",
]


@builtin
def async_load(tensor_desc, coord, barrier, result, pred=True, multicast=False, report_validity="none", _semantic=None):
    """
    Load data from global memory to shared memory using TMA.

    Args:
        tensor_desc: Tensor descriptor (tiled).
        coord: Coordinates in the source tensor.
        barrier: Barrier for synchronization.
        result: Destination memory descriptor.
        pred: Predicate for conditional execution.
        multicast: Enable multicast.
        report_validity: Optional payload validity mode carried on the TMA
            completion barrier. Validity reporting does not support two-CTA
            TMA mode. Supported values are:

            - ``"none"``: disable payload inspection.
            - ``"per_16B_fp32"``: sample one FP32 element in each aligned
              16-byte chunk and match the ``-0`` bit pattern ``0x80000000``.
            - ``"per_16B_fp16"``: sample one FP16 element in each aligned
              16-byte chunk and match the ``-0`` bit pattern ``0x8000``.
            - ``"per_16B_fp8"``: sample one FP8 element in each aligned
              16-byte chunk and match ``0x80``.
            - ``"per_16B_fp4"``: sample one FP4 element in each aligned
              16-byte chunk and match ``0x8``.
            - ``"per_elem_1B"``: match each byte against ``0xff``.

            Given the lack of a dedicated fp4 type in Triton, we cannot infer
            the right sentinal value for a given buffer data type. So for now,
            the report_validity kind needs to be passed explicitly as a string.

            A sentinel match prevents the mbarrier conditional phase from
            completing. Use
            ``rubin.mbarrier.wait(..., phase_type="conditional")`` to
            consume the load and ``rubin.mbarrier.test_wait_validity(...)`` to
            decide whether to retry it.
    """
    if _semantic.builder.options.enable_iisan:
        _emit_alignment_check(tensor_desc, coord, "async_load", "innermost coordinate", _semantic=_semantic)

    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    pred = _semantic.to_tensor(pred)
    multicast = _unwrap_if_constexpr(multicast)
    report_validity = _semantic._str_to_report_validity(report_validity)

    _semantic.builder.create_async_tma_copy_global_to_local(
        tensor_desc.handle,
        coord,
        barrier.handle,
        result.handle,
        pred.handle,
        multicast,
        None,
        report_validity,
    )
