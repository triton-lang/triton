"""Communication through TW's existing synchronized buffers and CPU proxy.

Each operation is collective over the current Gluon execution region. Buffer
views share notification cursors and send-completion state. Consuming
notification waits require exclusive ownership.

Kernel operations return scalar ``gl.int1`` values. Sends and arrivals report
request acceptance, waits report readiness, and ``is_aborted`` reports abort
status. A false nonblocking wait can mean either pending or aborted.

Counters persist for the lifetime of the TW registration. Notification, send,
and completion counters and wait targets must remain below 2**63 because TW
also uses signed int64 bookkeeping. Completion starts at -1; other counters
are nonnegative. Counter rollover is unsupported.
"""

from dataclasses import dataclass

from triton import aggregate
from triton.language.core import base_type, base_value
from triton.experimental.gluon.nvidia.fabric import SynchronizedBuffer
from .. import _core as gl
from .._core import builtin, _unwrap_if_constexpr

__all__ = ["SynchronizedBuffer", "barrier"]


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
        return _synchronized_buffer(*values), cursor

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
        self._buffer = buffer
        self.type = _view_type(buffer.type, kind)

    def _flatten_ir(self, handles):
        self._buffer._flatten_ir(handles)

    def _set_name(self, builder, name):
        self._buffer._set_name(builder, name)

    @property
    def ack_barrier(self):
        if self.type.kind != "peer":
            raise AttributeError("only a peer exposes an acknowledgement arrival target")
        return _view(self._buffer, "peer_ack")


@aggregate
class _local_barrier:
    _counter: gl.tensor
    _cursor: gl.tensor
    _aborted: gl.tensor
    _monitor_ptr: gl.tensor
    _kind: gl.constexpr
    _aborted_value: gl.constexpr


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


class _synchronized_buffer(base_value):
    """A typed view of an existing TW registration, passed by the host adapter.

    ``ptr`` exposes the local payload pointer.
    Offsets and transfer lengths use elements of ``ptr.dtype.element_ty``.
    Source and destination ranges must remain within the paired registration.
    The caller must retain the host adapter and reserve its TW channel until
    all kernels and outstanding communication have completed.
    """

    __triton_operator_priority__ = 2

    def __init__(self, ptr, state, queue, handle, capacity, protocol, base=None):
        # [Perf] Pass the registration pointer once, retaining it in offset views.
        base = ptr if base is None else base
        self.ptr, self._state, self._queue = ptr, state, queue
        self._handle, self._capacity, self._protocol, self._base = handle, capacity, protocol, base
        self._values = (ptr, state, queue, handle, capacity, protocol, base)
        self.type = _synchronized_buffer_type([value.type for value in self._values])

    def _flatten_ir(self, handles):
        for value in self._values:
            value._flatten_ir(handles)

    def _set_name(self, builder, name):
        for index, value in enumerate(self._values):
            value._set_name(builder, f"{name}.{index}")

    @property
    def recv_barrier(self):
        return self._local_barrier("recv")

    @property
    def ack_barrier(self):
        return self._local_barrier("ack")

    @property
    def peer(self):
        return _view(self, "peer")

    def _local_barrier(self, kind):
        state = self._state
        counter, cursor, monitor_ptr = {
            "recv": (state.recv_lock, state.wait_count, state.periscope_recv_lock),
            "ack": (state.ack_lock, state.ack_count, state.periscope_ack_lock),
            "send": (state.sends_completed, state.send_count, state.periscope_sends_completed),
        }[kind]
        protocol = _unwrap_if_constexpr(self._protocol)
        return _local_barrier(counter, cursor, state.aborted, monitor_ptr, kind, protocol.aborted_value)

    @builtin
    def __add__(self, offset, _semantic=None):
        offset = _integer(offset, "offset", _semantic)
        return _synchronized_buffer(_semantic.add(self.ptr, offset, False), self._state, self._queue, self._handle,
                                    self._capacity, self._protocol, self._base)

    @builtin
    def async_store(self, src, numel, *, bypass_ar=False, _semantic=None):
        """Submit one notifying send, including when ``numel`` is zero.

        ``src`` must point into this registration's local source allocation.
        A true result means the request was accepted, not completed.

        ``bypass_ar`` is a constexpr bool requesting that TW bypass adaptive
        routing. The transport may override this choice. Completion and buffer
        lifetime requirements are the same for both values.
        """
        bypass_ar = _unwrap_if_constexpr(bypass_ar)
        if not isinstance(bypass_ar, bool):
            raise TypeError("bypass_ar must be a constexpr bool")
        src = _semantic.to_tensor(src)
        if src.type.is_block() or not src.dtype.is_ptr() or src.dtype != self.ptr.dtype:
            raise TypeError("src must be a scalar pointer with the buffer's element type")
        numel = _integer(numel, "numel", _semantic, nonnegative=True)
        itemsize = self.ptr.dtype.element_ty.primitive_bitwidth // 8
        base = _semantic.cast(self._base, gl.int64)
        dst_offset = _semantic.sub(_semantic.cast(self.ptr, gl.int64), base, False)
        src_offset = _semantic.sub(_semantic.cast(src, gl.int64), base, False)
        nbytes = _semantic.mul(numel, _semantic.to_tensor(itemsize), False)
        return self._submit(src_offset, dst_offset, nbytes, True, _semantic, bypass_ar)

    def _submit(self, src_offset, dst_offset, nbytes, is_send, semantic, bypass_ar=False):
        capacity = _unwrap_if_constexpr(self._capacity)
        if type(capacity) is not int:
            raise TypeError("capacity must be a constexpr integer")
        if not 2 <= capacity < 2**63:
            raise ValueError("capacity must satisfy 2 <= capacity < 2**63")
        protocol = _unwrap_if_constexpr(self._protocol)
        queue = self._queue
        bypass_value = protocol.bypass_ar_true if bypass_ar else protocol.bypass_ar_false
        result = semantic.builder.create_comm_submit(queue.head.handle, queue.tail.handle, queue.cached_head.handle,
                                                     queue.buffer.handle, capacity, self._handle.handle,
                                                     src_offset.handle, dst_offset.handle, nbytes.handle,
                                                     self._state.send_count.handle, self._state.aborted.handle, is_send,
                                                     protocol.request_layout,
                                                     protocol.send_type if is_send else protocol.recv_ack_type,
                                                     bypass_value, protocol.ready_value, protocol.aborted_value)
        return gl.tensor(result, gl.int1)

    @builtin
    def store_wait(self, *, blocking: bool = True, monitor: bool = False, _semantic=None):
        """Wait for all preceding sends; true means the source is reusable.

        With ``blocking=False``, check completion once and return false if
        pending or aborted. ``blocking`` must be a constexpr bool.
        Use ``is_aborted()`` to distinguish an abort from pending sends.

        ``monitor=True`` publishes the completion target to the progress monitor
        before checking abort or readiness. It must be a constexpr bool. The
        default retains delayed publication for blocking waits.

        All submitting regions must complete their ``async_store`` calls and
        synchronize before this wait. Do not start new sends on any view of
        this buffer until the wait and source updates finish.
        """
        return _wait(self._local_barrier("send"), 0, False, blocking, monitor, _semantic)

    @builtin
    def is_aborted(self, _semantic=None):
        """Return true if this buffer has been aborted."""
        protocol = _unwrap_if_constexpr(self._protocol)
        result = _semantic.builder.create_comm_is_aborted(self._state.aborted.handle, protocol.aborted_value)
        return gl.tensor(result, gl.int1)


def _wait(target, count, consume, blocking, monitor, semantic):
    count = _integer(count, "count", semantic, nonnegative=True)
    consume = _unwrap_if_constexpr(consume)
    if not isinstance(consume, bool):
        raise TypeError("consume must be a constexpr bool")
    blocking = _unwrap_if_constexpr(blocking)
    if not isinstance(blocking, bool):
        raise TypeError("blocking must be a constexpr bool")
    monitor = _unwrap_if_constexpr(monitor)
    if not isinstance(monitor, bool):
        raise TypeError("monitor must be a constexpr bool")
    result = semantic.builder.create_comm_wait(target._counter.handle, target._cursor.handle, target._aborted.handle,
                                               target._monitor_ptr.handle, count.handle,
                                               _unwrap_if_constexpr(target._kind), consume,
                                               _unwrap_if_constexpr(target._aborted_value), blocking, monitor)
    return gl.tensor(result, gl.int1)


class barrier:
    """Views of TW receive notifications and explicit acknowledgements."""

    @staticmethod
    @builtin
    def wait(local_barrier, *, blocking: bool = True, count=1, consume=True, monitor: bool = False, _semantic=None):
        """Wait for relative notifications; abort returns false without consuming.

        With ``blocking=False``, check readiness once and return false if
        pending or aborted. A ready wait still honors ``consume``.
        ``blocking`` must be a constexpr bool. Use the buffer's ``is_aborted()``
        to distinguish an abort from pending notifications.

        ``monitor=True`` publishes the target to the progress monitor before
        checking abort or readiness. It must be a constexpr bool. The default
        retains delayed publication for blocking waits. Concurrent observers
        must coordinate publication so a smaller target cannot hide a larger one.

        ``count`` must be nonnegative, and adding it to the notification cursor
        must produce a target below 2**63. Zero checks abort without polling.
        Coordinated observers can pass ``consume=False`` to retain notifications.
        """
        if not isinstance(local_barrier, _local_barrier) or _unwrap_if_constexpr(
                local_barrier._kind) not in ("recv", "ack"):
            raise TypeError("wait requires a local receive or acknowledgement barrier")
        return _wait(local_barrier, count, consume, blocking, monitor, _semantic)

    @staticmethod
    @builtin
    def arrive(peer_ack_barrier, _semantic=None):
        """Submit one peer acknowledgement; true means the request was accepted."""
        if not isinstance(peer_ack_barrier, _view) or peer_ack_barrier.type.kind != "peer_ack":
            raise TypeError("arrive requires a peer acknowledgement barrier")
        zero = _semantic.scalar_constant(0, gl.int64)
        return peer_ack_barrier._buffer._submit(zero, zero, zero, False, _semantic)
