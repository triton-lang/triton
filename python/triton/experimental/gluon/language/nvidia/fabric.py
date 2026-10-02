"""Communication through TW's existing synchronized buffers and CPU proxy.

Each operation is collective over the current Gluon execution region. Buffer
views share notification cursors; consuming waits require exclusive ownership.
"""

from dataclasses import dataclass

from triton.language.core import base_type, base_value
from triton.experimental.gluon.nvidia.fabric import GluonSynchronizedBuffer
from .. import _core as gl
from .._core import builtin, _unwrap_if_constexpr

__all__ = ["GluonSynchronizedBuffer", "barrier", "synchronized_buffer"]


@dataclass
class _synchronized_buffer_type(base_type):
    types: list

    def __getitem__(self, index):
        return self.types[index]

    def _flatten_ir_types(self, builder, out):
        for ty in self.types:
            ty._flatten_ir_types(builder, out)

    def _unflatten_ir(self, handles, cursor):
        values = []
        for ty in self.types:
            value, cursor = ty._unflatten_ir(handles, cursor)
            values.append(value)
        return synchronized_buffer(*values), cursor

    def mangle(self):
        return "FAB<" + ",".join(ty.mangle() for ty in self.types) + ">"


@dataclass
class _view_type(base_type):
    buffer_type: _synchronized_buffer_type
    kind: str

    def _flatten_ir_types(self, builder, out):
        self.buffer_type._flatten_ir_types(builder, out)

    def _unflatten_ir(self, handles, cursor):
        buffer, cursor = self.buffer_type._unflatten_ir(handles, cursor)
        return _view(buffer, self.kind), cursor

    def mangle(self):
        return self.kind + self.buffer_type.mangle()


class _view(base_value):

    def __init__(self, buffer, kind):
        self.buffer = buffer
        self.kind = kind
        self.type = _view_type(buffer.type, kind)

    def _flatten_ir(self, handles):
        self.buffer._flatten_ir(handles)

    def _set_name(self, builder, name):
        self.buffer._set_name(builder, name)

    @property
    def ack_barrier(self):
        if self.kind != "peer":
            raise AttributeError("only a peer exposes an acknowledgement arrival target")
        return _view(self.buffer, "peer_ack")


def _integer(value, name, semantic, *, nonnegative=False):
    constant = _unwrap_if_constexpr(value)
    if isinstance(constant, int):
        if nonnegative and constant < 0:
            raise ValueError(f"{name} must be nonnegative")
        return semantic.scalar_constant(constant, gl.int64)
    value = semantic.to_tensor(value)
    if value.type.is_block() or not value.dtype.is_int():
        raise TypeError(f"{name} must be a scalar integer")
    return semantic.cast(value, gl.int64)


class synchronized_buffer(base_value):
    """A typed view of an existing TW registration, passed by the host adapter.

    Offsets and transfer lengths use elements of ``ptr.dtype.element_ty``.
    Source and destination ranges must remain within the paired registration.
    The caller must retain the host adapter and reserve its TW channel until
    all kernels and outstanding communication have completed.
    """

    __triton_operator_priority__ = 2

    def __init__(self, ptr, state, queue, handle, capacity, protocol, base):
        self.ptr, self.state, self.queue = ptr, state, queue
        self.handle, self.capacity, self.protocol, self.base = handle, capacity, protocol, base
        self.values = (ptr, state, queue, handle, capacity, protocol, base)
        self.type = _synchronized_buffer_type([value.type for value in self.values])

    def _flatten_ir(self, handles):
        for value in self.values:
            value._flatten_ir(handles)

    def _set_name(self, builder, name):
        for index, value in enumerate(self.values):
            value._set_name(builder, f"{name}.{index}")

    @property
    def recv_barrier(self):
        return _view(self, "recv")

    @property
    def ack_barrier(self):
        return _view(self, "ack")

    @property
    def peer(self):
        return _view(self, "peer")

    @builtin
    def __add__(self, offset, _semantic=None):
        offset = _integer(offset, "offset", _semantic)
        return synchronized_buffer(_semantic.add(self.ptr, offset, False), self.state, self.queue, self.handle,
                                   self.capacity, self.protocol, self.base)

    @builtin
    def async_store(self, src, numel, _semantic=None):
        """Submit one notifying send, including when ``numel`` is zero.

        ``src`` must point into this registration's local source allocation.
        A true result means the request was accepted, not completed.
        """
        src = _semantic.to_tensor(src)
        if src.type.is_block() or not src.dtype.is_ptr() or src.dtype != self.ptr.dtype:
            raise TypeError("src must be a scalar pointer with the buffer's element type")
        numel = _integer(numel, "numel", _semantic, nonnegative=True)
        itemsize = self.ptr.dtype.element_ty.primitive_bitwidth // 8
        base = _semantic.cast(self.base, gl.int64)
        dst_offset = _semantic.sub(_semantic.cast(self.ptr, gl.int64), base, False)
        src_offset = _semantic.sub(_semantic.cast(src, gl.int64), base, False)
        nbytes = _semantic.mul(numel, _semantic.to_tensor(itemsize), False)
        return self._submit(src_offset, dst_offset, nbytes, True, _semantic)

    def _submit(self, src_offset, dst_offset, nbytes, is_send, semantic):
        protocol = _unwrap_if_constexpr(self.protocol)
        queue = self.queue
        result = semantic.builder.create_comm_submit(
            queue.head.handle, queue.tail.handle, queue.cached_head.handle, queue.buffer.handle,
            _integer(self.capacity, "capacity", semantic).handle,
            semantic.cast(semantic.to_tensor(self.handle), gl.int32).handle, src_offset.handle, dst_offset.handle,
            nbytes.handle, self.state.send_count.handle, self.state.aborted.handle, is_send, protocol.request_layout,
            protocol.send_type if is_send else protocol.recv_ack_type, protocol.bypass_ar_false, protocol.ready_value,
            protocol.aborted_value)
        return gl.tensor(result, gl.int1)

    @builtin
    def store_wait(self, _semantic=None):
        """Wait for all preceding normal sends; true means the source is reusable."""
        return _wait(self, "send", 0, False, _semantic)

    @builtin
    def is_aborted(self, _semantic=None):
        protocol = _unwrap_if_constexpr(self.protocol)
        result = _semantic.builder.create_comm_is_aborted(self.state.aborted.handle, protocol.aborted_value)
        return gl.tensor(result, gl.int1)


def _wait(buffer, kind, count, consume, semantic):
    count = _integer(count, "count", semantic, nonnegative=True)
    consume = _unwrap_if_constexpr(consume)
    if not isinstance(consume, bool):
        raise TypeError("consume must be a constexpr bool")
    state = buffer.state
    counter, cursor, monitor = {
        "recv": (state.recv_lock, state.wait_count, state.periscope_recv_lock),
        "ack": (state.ack_lock, state.ack_count, state.periscope_ack_lock),
        "send": (state.sends_completed, state.send_count, state.periscope_sends_completed),
    }[kind]
    protocol = _unwrap_if_constexpr(buffer.protocol)
    result = semantic.builder.create_comm_wait(counter.handle, cursor.handle, state.aborted.handle, monitor.handle,
                                               count.handle, kind, consume, kind == "recv", protocol.aborted_value)
    return gl.tensor(result, gl.int1)


class barrier:
    """Views of TW receive notifications and explicit acknowledgements."""

    @staticmethod
    @builtin
    def wait(local_barrier, *, count=1, consume=True, _semantic=None):
        """Wait for relative notifications; abort returns false without consuming.

        ``count`` must be nonnegative. Zero checks abort without polling.
        Coordinated observers can pass ``consume=False`` to retain notifications.
        """
        if not isinstance(local_barrier, _view) or local_barrier.kind not in ("recv", "ack"):
            raise TypeError("wait requires a local receive or acknowledgement barrier")
        return _wait(local_barrier.buffer, local_barrier.kind, count, consume, _semantic)

    @staticmethod
    @builtin
    def arrive(peer_ack_barrier, _semantic=None):
        """Submit one peer acknowledgement; true means the request was accepted."""
        if not isinstance(peer_ack_barrier, _view) or peer_ack_barrier.kind != "peer_ack":
            raise TypeError("arrive requires a peer acknowledgement barrier")
        zero = _semantic.scalar_constant(0, gl.int64)
        return peer_ack_barrier.buffer._submit(zero, zero, zero, False, _semantic)
