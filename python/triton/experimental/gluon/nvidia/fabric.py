"""Fabric argument construction and TW's paired-buffer host adapter."""

from collections import namedtuple
from dataclasses import dataclass, fields

import triton.language as tl
from triton._utils import canonicalize_dtype

__all__ = ["SynchronizedBuffer", "RequestLayout", "Protocol"]

_State = namedtuple(
    "_State", "recv_lock ack_lock sends_completed aborted wait_count ack_count send_count "
    "periscope_recv_lock periscope_ack_lock periscope_sends_completed")
_Queue = namedtuple("_Queue", "head tail cached_head buffer")


@dataclass(frozen=True)
class _Pointer:
    address: int
    dtype: tl.dtype

    def __post_init__(self):
        if not self.dtype.itemsize:
            raise ValueError("the buffer element type must occupy whole bytes")
        if self.address % self.dtype.itemsize:
            raise ValueError(f"the address must be aligned for dtype {self.dtype}")

    def data_ptr(self):
        return self.address


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestLayout:
    """Request stride and field offsets, in bytes, in the proxy queue."""

    stride: int
    ready_offset: int
    type_offset: int
    handle_offset: int
    src_offset: int
    dst_offset: int
    length_offset: int
    bypass_ar_offset: int

    def __post_init__(self):
        if type(self.stride) is not int or self.stride <= 0 or self.stride % 8:
            raise ValueError("request stride must be a positive multiple of 8")
        fields = [(self.ready_offset, 8), (self.type_offset, 4), (self.handle_offset, 4), (self.src_offset, 8),
                  (self.dst_offset, 8), (self.length_offset, 8), (self.bypass_ar_offset, 4)]
        occupied = set()
        for offset, width in fields:
            if type(offset) is not int or offset < 0 or offset % width or offset + width > self.stride:
                raise ValueError("request field must be aligned and fit within stride")
            field = set(range(offset, offset + width))
            if field & occupied:
                raise ValueError("request fields must not overlap")
            occupied.update(field)


@dataclass(frozen=True, slots=True)
class Protocol:
    """Compile-time request layout and constants exported by the transport."""

    request_layout: RequestLayout
    send_type: int
    recv_ack_type: int
    bypass_ar_false: int
    bypass_ar_true: int
    ready_value: int
    aborted_value: int


class SynchronizedBuffer(namedtuple("_SynchronizedBufferArgs", "ptr state queue handle capacity protocol")):
    """A host argument constructed from device addresses and transport metadata.

    Call ``fabric.bind(argument)`` inside a Gluon kernel to construct its native
    aggregate descriptor.

    ``address`` points to the payload, interpreted as ``dtype``. ``state`` and
    ``queue`` are mappings of names to device addresses, in bytes. State requires
    ``recv_lock``, ``ack_lock``, ``sends_completed``, ``aborted``, ``wait_count``,
    ``ack_count``, ``send_count``, ``periscope_recv_lock``, ``periscope_ack_lock``
    and ``periscope_sends_completed``. Queue requires ``head``, ``tail``,
    ``cached_head`` and ``buffer``. State and queue addresses must be 8-byte
    aligned. Additional mapping entries are ignored.

    The handle is a runtime int32. Capacity and protocol are constexprs. Pointer
    alignment is specialized. Offline compilation can use placeholder addresses
    with alignment guaranteed by the runtime allocations.

    Construction requires no TW runtime or device allocation. The caller must
    retain the payload, state, queue and transport resources until GPU work and
    communication have completed.
    """

    __triton_do_not_specialize__ = ("handle", )

    def __new__(cls, *fields, **kwargs):
        # Triton reconstructs namedtuples with type strings and specialization keys.
        if fields:
            return super().__new__(cls, *fields, **kwargs)
        return cls._from_addresses(**kwargs)

    @classmethod
    def _from_addresses(cls, *, address, dtype, state, queue, handle, capacity, protocol: Protocol):
        dtype = tl.str_to_ty(canonicalize_dtype(dtype), None)
        if type(handle) is not int or not 0 <= handle < 2**31:
            raise ValueError("the fabric handle must be a nonnegative int32")
        if type(capacity) is not int or not 2 <= capacity < 2**63:
            raise ValueError("capacity must satisfy 2 <= capacity < 2**63")
        # Each byte-addressed queue slot contains a 64-bit ready word.
        if queue["buffer"] % tl.int64.itemsize:
            raise ValueError("the queue buffer must be 8-byte aligned")
        return super().__new__(
            cls, _Pointer(address, dtype), _State(*(_Pointer(state[name], tl.int64) for name in _State._fields)),
            _Queue(*(_Pointer(queue[name], tl.int8 if name == "buffer" else tl.int64) for name in _Queue._fields)),
            handle, tl.constexpr(capacity), tl.constexpr(protocol))

    @classmethod
    def from_tw(cls, group, handle, *, dtype=None):
        """Bind an existing TW buffer and request queue without allocating state.

        ``dtype`` interprets the registered bytes; it never copies or converts data.
        Keep this adapter alive until GPU work and communication have completed, and
        follow the group's normal channel acquisition and release protocol.
        """
        from synchronized_buffer import BypassAr, SynchronizedBuffer as TWSynchronizedBuffer

        tensor = handle.tensor
        if tensor.device.type != "cuda":
            raise ValueError("SynchronizedBuffer requires a GPU buffer")
        native = group.synchronized_buffer_store.get(handle).gpu_synchronized_buffer()
        buffer_fields = native.gpu_fields()
        queue = group.gpu_cpu_queue.queue
        queue_fields, layout = queue.gpu_fields(), queue.request_layout()
        request_layout = RequestLayout(**{field.name: layout[field.name] for field in fields(RequestLayout)})
        protocol = Protocol(request_layout, layout["send_type"], layout["recv_ack_type"], layout["bypass_ar_false"],
                            int(BypassAr.TRUE), layout["ready_value"], TWSynchronizedBuffer.ABORTED)
        result = cls(address=buffer_fields["local_addr"], dtype=tensor.dtype if dtype is None else dtype,
                     state=buffer_fields, queue=queue_fields, handle=buffer_fields["handle"],
                     capacity=queue_fields["capacity"], protocol=protocol)
        if buffer_fields["size"] % result.ptr.dtype.itemsize:
            raise ValueError("the registered buffer must be sized for dtype")
        result._owners = (group, handle, native, queue)
        return result
