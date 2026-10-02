"""Host adapter for TW's paired-buffer communication runtime."""

from collections import namedtuple
from dataclasses import dataclass

from triton._utils import canonicalize_dtype
from triton.language import constexpr, int64, int8, str_to_ty

__all__ = ["GluonSynchronizedBuffer"]

_State = namedtuple(
    "_State", "recv_lock ack_lock sends_completed aborted wait_count ack_count send_count "
    "periscope_recv_lock periscope_ack_lock periscope_sends_completed")
_Queue = namedtuple("_Queue", "head tail cached_head buffer")


@dataclass(frozen=True)
class _Pointer:
    address: int
    dtype: object

    def data_ptr(self):
        return self.address


@dataclass(frozen=True)
class _Protocol:
    request_layout: tuple
    send_type: int
    recv_ack_type: int
    bypass_ar_false: int
    ready_value: int
    aborted_value: int


class _SynchronizedBufferArgs(namedtuple("_SynchronizedBufferArgsBase",
                                         "ptr state queue handle capacity protocol base")):

    @staticmethod
    def __triton_type__(types):
        from ..language.nvidia.fabric import _synchronized_buffer_type
        return _synchronized_buffer_type(types)


def GluonSynchronizedBuffer(group, handle, *, dtype=None):
    """Bind an existing TW buffer and request queue without allocating state.

    ``dtype`` interprets the registered bytes; it never copies or converts data.
    Keep this adapter alive until GPU work and communication have completed, and
    follow the group's normal channel acquisition and release protocol.
    """
    from synchronized_buffer import SynchronizedBuffer

    tensor = handle.tensor
    if tensor.device.type != "cuda":
        raise ValueError("GluonSynchronizedBuffer requires a GPU buffer")
    dtype = str_to_ty(canonicalize_dtype(tensor.dtype if dtype is None else dtype), None)
    if dtype.primitive_bitwidth < 8:
        raise ValueError("the buffer element type must occupy whole bytes")
    native = group.synchronized_buffer_store.get(handle).gpu_synchronized_buffer()
    fields = native.gpu_fields()
    itemsize = dtype.primitive_bitwidth // 8
    if fields["local_addr"] % itemsize or fields["size"] % itemsize:
        raise ValueError("the registered buffer must be aligned and sized for dtype")
    queue = group.gpu_cpu_queue.queue
    queue_fields, layout = queue.gpu_fields(), queue.request_layout()
    layout_fields = ("stride", "ready_offset", "type_offset", "handle_offset", "src_offset", "dst_offset",
                     "length_offset", "bypass_ar_offset")
    protocol = _Protocol(tuple(layout[name] for name in layout_fields), layout["send_type"], layout["recv_ack_type"],
                         layout["bypass_ar_false"], layout["ready_value"], SynchronizedBuffer.ABORTED)
    ptr = _Pointer(fields["local_addr"], dtype)
    result = _SynchronizedBufferArgs(
        ptr, _State(*(_Pointer(fields[name], int64) for name in _State._fields)),
        _Queue(*(_Pointer(queue_fields[name], int8 if name == "buffer" else int64) for name in _Queue._fields)),
        fields["handle"], constexpr(queue_fields["capacity"]), constexpr(protocol), ptr)
    result._owners = (group, handle, native, queue)
    return result
