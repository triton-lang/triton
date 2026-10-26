"""TW communication expressed with native Gluon aggregates and JIT functions.

Use ``bind(host_argument)`` inside a kernel. Operations are collective over the
current Gluon execution region. Views share notification cursors and completion
state. Consuming waits require exclusive ownership. Retain the host adapter and
its registration until all kernels and outstanding transfers have completed.

Counters and notification targets must remain below 2**63. Send completion starts
at -1. Queue positions are monotonic unsigned counters, with no rollover.
"""

import builtins

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.fabric import SynchronizedBuffer

__all__ = ["SynchronizedBuffer", "bind", "barrier"]


@gluon.aggregate
class _Registration:
    base: gl.tensor
    state: gl.tuple
    queue: gl.tuple
    handle: gl.tensor
    capacity: gl.constexpr
    protocol: gl.constexpr


@gluon.aggregate
class _LocalBarrier:
    _counter: gl.tensor
    _cursor: gl.tensor
    _aborted: gl.tensor
    _monitor_ptr: gl.tensor
    _kind: gl.constexpr
    _aborted_value: gl.constexpr


@gluon.aggregate
class _PeerAck:
    registration: _Registration


@gluon.aggregate
class _Peer:
    ack_barrier: _PeerAck


@gluon.aggregate
class _Buffer:
    ptr: gl.tensor
    _registration: _Registration
    recv_barrier: _LocalBarrier
    ack_barrier: _LocalBarrier
    peer: _Peer

    @gluon.jit
    def __add__(self, offset):
        offset = _integer(offset, "offset")
        return gl.aggregate_replace(self, ptr=self.ptr + offset)

    @gluon.jit
    def async_store(self, src, numel, *, bypass_ar: gl.constexpr = False):
        """Submit one notifying send, including a zero-length send.

        ``numel`` must be nonnegative. Both element ranges must lie in the
        registered allocation. A true result means the proxy request was queued.
        """
        gl.static_assert(isinstance(bypass_ar, builtins.bool), "bypass_ar must be a constexpr bool")
        src = gl.to_tensor(src)
        gl.static_assert(src.shape == [] and src.dtype == self.ptr.dtype,
                         "src must be a scalar pointer with the buffer's element type")
        numel = _integer(numel, "numel", True)
        base = self._registration.base.to(gl.int64)
        src_offset = src.to(gl.int64) - base
        dst_offset = self.ptr.to(gl.int64) - base
        nbytes = numel * self.ptr.dtype.element_ty.itemsize
        return _submit(self._registration, src_offset, dst_offset, nbytes, True, bypass_ar)

    @gluon.jit
    def store_wait(self, *, blocking: gl.constexpr = True, monitor: gl.constexpr = False):
        """Wait for preceding sends before reusing their sources.

        Finish and synchronize all submissions first, and exclude new sends
        until this wait and subsequent source updates finish.
        """
        state = self._registration.state
        target = _LocalBarrier(state.sends_completed, state.send_count, state.aborted, state.periscope_sends_completed,
                               "send", self._registration.protocol.aborted_value)
        return _wait(target, 0, False, blocking, monitor)

    @gluon.jit
    def is_aborted(self):
        return _is_aborted(self._registration.state.aborted, self._registration.protocol.aborted_value)


@gluon.jit
def _integer(value, name: gl.constexpr, nonnegative: gl.constexpr = False):
    if not isinstance(value, gl.tensor):
        gl.static_assert(isinstance(value, int), name + " must be a scalar integer")
        if nonnegative:
            gl.static_assert(value >= 0, name + " must be nonnegative")
    value = gl.to_tensor(value)
    gl.static_assert(value.shape == [] and value.dtype.is_int(), name + " must be a scalar integer")
    return value.to(gl.int64)


@gluon.jit
def bind(argument):
    """Construct an aggregate view of a host ``SynchronizedBuffer`` argument."""
    gl.static_assert(gl.num_ctas() == 1, "fabric communication requires one CTA per program")
    gl.static_assert(isinstance(argument[4], int), "capacity must be a constexpr integer")
    gl.static_assert(2 <= argument[4] < 2**63, "capacity must satisfy 2 <= capacity < 2**63")
    ptr, state, queue, handle = argument[:4]
    protocol: gl.constexpr = argument[5]
    registration = _Registration(ptr, state, queue, gl.to_tensor(handle).to(gl.int32), argument[4], protocol)
    recv = _LocalBarrier(state.recv_lock, state.wait_count, state.aborted, state.periscope_recv_lock, "recv",
                         protocol.aborted_value)
    ack = _LocalBarrier(state.ack_lock, state.ack_count, state.aborted, state.periscope_ack_lock, "ack",
                        protocol.aborted_value)
    return _Buffer(ptr, registration, recv, ack, _Peer(_PeerAck(registration)))


@gluon.jit
def _is_aborted(ptr, aborted_value: gl.constexpr):
    return gl.atomic_load(ptr, sem="acquire", scope="sys") == aborted_value


@gluon.jit
def _target(bar, count):
    if bar._kind == "send":
        target = gl.atomic_load(bar._cursor, sem="relaxed", scope="gpu") - 1
    else:
        target = gl.atomic_load(bar._cursor, sem="acquire", scope="gpu") + count
    return target


@gluon.jit
def _wait(bar, count, consume: gl.constexpr, blocking: gl.constexpr, monitor: gl.constexpr):
    gl.static_assert(isinstance(consume, builtins.bool), "consume must be a constexpr bool")
    gl.static_assert(isinstance(blocking, builtins.bool), "blocking must be a constexpr bool")
    gl.static_assert(isinstance(monitor, builtins.bool), "monitor must be a constexpr bool")
    count = _integer(count, "count", True)
    if monitor:
        target = _target(bar, count)
        gl.atomic_store(bar._monitor_ptr, target, sem="release", scope="sys")
    aborted = _is_aborted(bar._aborted, bar._aborted_value)
    empty = bar._kind != "send" and count == 0
    ready = not aborted and empty
    if not aborted and not empty:
        if not monitor:
            target = _target(bar, count)
        matched = gl.atomic_load(bar._counter, sem="relaxed", scope="sys") >= target
        if blocking:
            spins = 1
            while not matched and not aborted:
                matched = gl.atomic_load(bar._counter, sem="relaxed", scope="sys") >= target
                spins += 1
                if not matched:
                    if not monitor and spins == 32:
                        gl.atomic_store(bar._monitor_ptr, target, sem="release", scope="sys")
                    if (spins & 2047) == 0:
                        aborted = _is_aborted(bar._aborted, bar._aborted_value)
        if matched:
            ready = not _is_aborted(bar._aborted, bar._aborted_value)
            if ready:
                if bar._kind == "recv":
                    gl.atomic_load(bar._counter, sem="acquire", scope="sys")
                if consume:
                    gl.atomic_store(bar._cursor, target, sem="release", scope="gpu")
    return ready


@gluon.jit
def _submit(registration, src_offset, dst_offset, nbytes, is_send: gl.constexpr, bypass_ar: gl.constexpr = False):
    queue = registration.queue
    state = registration.state
    protocol: gl.constexpr = registration.protocol
    layout: gl.constexpr = protocol.request_layout
    tail_ptr = queue.tail.to(gl.pointer_type(gl.uint64))
    cached_head_ptr = queue.cached_head.to(gl.pointer_type(gl.uint64))
    claimed = False
    aborted = False
    tail = gl.to_tensor(0).to(gl.uint64)
    while not claimed and not aborted:
        aborted = _is_aborted(state.aborted, protocol.aborted_value)
        if not aborted:
            cached = gl.atomic_load(cached_head_ptr, sem="acquire", scope="gpu")
            tail = gl.atomic_load(tail_ptr, sem="relaxed", scope="gpu")
            if tail - cached < registration.capacity - 1:
                previous = gl.atomic_cas(tail_ptr, tail, tail + 1, sem="relaxed", scope="gpu")
                claimed = previous == tail
            else:
                head = gl.atomic_load(queue.head, sem="acquire", scope="sys").to(gl.uint64)
                if head > cached:
                    # Relay the CPU's reads and ready-word clears to producers
                    # that observe this cached head before reusing ring slots.
                    gl.atomic_max(cached_head_ptr, head, sem="release", scope="gpu")
    if claimed:
        # Once reserved, publish the slot even if an abort races with the CAS.
        slot = queue.buffer + (tail % registration.capacity) * layout.stride
        gl.store((slot + layout.type_offset).to(gl.pointer_type(gl.int32)),
                 protocol.send_type if is_send else protocol.recv_ack_type)
        gl.store((slot + layout.handle_offset).to(gl.pointer_type(gl.int32)), registration.handle)
        if is_send:
            gl.store((slot + layout.src_offset).to(gl.pointer_type(gl.int64)), src_offset)
            gl.store((slot + layout.dst_offset).to(gl.pointer_type(gl.int64)), dst_offset)
            gl.store((slot + layout.length_offset).to(gl.pointer_type(gl.int64)), nbytes)
            gl.store((slot + layout.bypass_ar_offset).to(gl.pointer_type(gl.int32)),
                     protocol.bypass_ar_true if bypass_ar else protocol.bypass_ar_false)
        gl.atomic_store((slot + layout.ready_offset).to(gl.pointer_type(gl.uint64)), protocol.ready_value,
                        sem="release", scope="sys")
        if is_send:
            gl.atomic_add(state.send_count, 1, sem="relaxed", scope="gpu")
    return claimed


class barrier:
    """Receive notifications and explicit peer acknowledgements."""

    @staticmethod
    @gluon.jit
    def wait(local_barrier, *, blocking: gl.constexpr = True, count=1, consume: gl.constexpr = True,
             monitor: gl.constexpr = False):
        """Wait for ``count`` notifications. Only a successful wait consumes them.

        The count must be nonnegative and the cursor plus count must fit in
        signed int64. Zero checks abort. Nonblocking waits check once.
        Monitoring publishes the target before checking abort or readiness.
        Other blocking waits publish on their 32nd miss. Coordinate concurrent observers.
        """
        gl.static_assert(isinstance(local_barrier, _LocalBarrier),
                         "wait requires a local receive or acknowledgement barrier")
        return _wait(local_barrier, count, consume, blocking, monitor)

    @staticmethod
    @gluon.jit
    def arrive(peer_ack_barrier):
        """Submit one peer acknowledgement and return whether it was accepted."""
        gl.static_assert(isinstance(peer_ack_barrier, _PeerAck), "arrive requires a peer acknowledgement barrier")
        return _submit(peer_ack_barrier.registration, 0, 0, 0, False)
