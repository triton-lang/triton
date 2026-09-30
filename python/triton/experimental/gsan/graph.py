"""Explicit CUDA graph ordering for external launchers.

The caller owns CUDA graph construction and must install the returned boundary
kernels with full entry/exit edges. CUDA events remain the caller's responsibility:
a Completion describes a real dependency, it does not establish one by itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import weakref

import triton
import triton.language as tl
from triton._C.libtriton.gsan_testing import PER_DEVICE_STATE_STRIDE_BYTES

from . import _stream_sync as sync
from ._allocator import get_device_rank, get_global_state_pointer

_live_plans = weakref.WeakSet()
_live_completions = weakref.WeakSet()
_pending = []


@dataclass(frozen=True)
class GraphNode:
    """A topologically ordered node; dependency indices refer to earlier nodes."""
    dependencies: tuple[int, ...] = ()
    programmatic_dependencies: tuple[int, ...] = ()
    is_kernel: bool = True


def dependency_frontiers(nodes):
    """Return completion frontiers acquired at entry and at gdc_wait.

    Following a PDL edge inherits only the producer's *entry* frontier. It
    cannot inherit anything the producer acquires or publishes while running.
    """
    entries, waits = [], []
    for index, node in enumerate(nodes):
        full, deferred = set(node.dependencies), set(node.programmatic_dependencies)
        if any(not isinstance(dep, int) or dep < 0 or dep >= index for dep in full | deferred):
            raise ValueError("Graph dependencies must refer to earlier nodes")
        if deferred and not node.is_kernel:
            raise ValueError("Programmatic dependencies require kernel nodes")
        if any(not nodes[dep].is_kernel for dep in deferred):
            raise ValueError("A programmatic predecessor must be a kernel")
        entry = set()
        for dep in full:
            entry.update((dep, ) if nodes[dep].is_kernel else entries[dep])
        for dep in deferred:
            entry.update(entries[dep])
        entries.append(tuple(sorted(entry)))
        waits.append(tuple(sorted(deferred)))
    return tuple(entries), tuple(waits)


@triton.jit(do_not_specialize=["num_inputs"])
def _graph_begin(Outputs, Baseline, Inputs, num_inputs, N: tl.constexpr, NODES: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Outputs + i, 0, i < N * NODES)
    clock = tl.full((BLOCK, ), 0, tl.uint32)
    for j in range(num_inputs):
        ptr = tl.load(Inputs + j).to(tl.pointer_type(tl.uint32))
        clock = tl.maximum(clock, tl.load(ptr + i, i < N, other=0))
    tl.store(Baseline + i, clock, i < N)


@triton.jit
def _graph_end(Outputs, Baseline, Last, Stream, Completion, N: tl.constexpr, NODES: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    clock = tl.load(Baseline + i, i < N, other=0).to(tl.uint32)
    for j in range(NODES):
        clock = tl.maximum(clock, tl.load(Outputs + j * N + i, i < N, other=0).to(tl.uint32))
    tl.store(Last + i, clock, i < N)
    tl.store(Completion + i, clock, i < N)
    # A reused eager stream slot already contains an older, ordered clock.
    tl.atomic_max(Stream + i, clock.to(tl.int32), i < N, sem="relaxed")


@triton.jit(do_not_specialize=["num_inputs"])
def _merge_clocks(Inputs, Output, num_inputs, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    clock = tl.full((BLOCK, ), 0, tl.uint32)
    for j in range(num_inputs):
        ptr = tl.load(Inputs + j).to(tl.pointer_type(tl.uint32))
        clock = tl.maximum(clock, tl.load(ptr + i, i < N, other=0))
    tl.store(Output + i, clock, i < N)


@dataclass(frozen=True)
class KernelLaunch:
    """Native CUDA kernel parameters. Every argument is an eight-byte value."""
    function: int
    grid: int
    block: int
    shared: int
    arguments: tuple[int, ...]


def _kernel_launch(compiled, grid, arguments):
    compiled._init_handles()
    if compiled.metadata.global_scratch_size or compiled.metadata.profile_scratch_size:
        raise RuntimeError("GSan graph helpers must not require scratch allocations")
    return KernelLaunch(compiled.function, grid, compiled.metadata.num_warps * 32, compiled.metadata.shared,
                        (*arguments, 0, 0))


def _collect_pending():
    _pending[:] = [(event, owner) for event, owner in _pending if not event.query()]


def _retain_until_stream_completes(device, stream, owner):
    import torch

    _collect_pending()
    with sync._clock_storage(device, stream):
        event = torch.cuda.Event()
        event.record()
    _pending.append((event, owner))


class Completion:
    """An immutable device clock snapshot, retained by every dependent launch."""

    def __init__(self, device, clock):
        self.device = device
        self.clock = clock
        _live_completions.add(self)


class GraphLaunch:

    def __init__(self, plan, stream, inputs, dependencies, completion, begin, end, before, after):
        self.plan = plan
        self.stream = stream
        self.inputs = inputs
        self.dependencies = dependencies
        self.completion = completion
        self.begin = begin
        self.end = end
        self.before = before
        self.after = after
        self._finished = False

    def finish(self, submitted: bool = True):
        """Call after enqueueing the graph, including partial-submission failures."""
        if not self._finished:
            if not submitted:
                with sync._clock_storage(self.plan.device, self.stream):
                    self.after.copy_(self.before)
            _retain_until_stream_completes(self.plan.device, self.stream, self)
            self._finished = True


class GraphPlan:
    """Clock storage owned by one CUDA executable graph.

    Install ``boundary_kernels`` before instantiation. Add a full edge from the
    entry helper to every original root and from every original node to the exit
    helper. Before each launch patch both helpers from ``prepare_launch``; call
    ``GraphLaunch.finish`` immediately after submission. Node launch arguments
    remain unchanged across replays and kernel-argument rebinding.
    """

    def __init__(self, device: int, nodes, stream: int = 0):
        import torch

        self.device = device
        self.nodes = tuple(nodes)
        self.entries, self.waits = dependency_frontiers(self.nodes)
        self.closed = False
        self.num_threads = sync._runtime_state_layout(get_device_rank(device), device).num_threads
        self.global_state = get_global_state_pointer() + get_device_rank(device) * PER_DEVICE_STATE_STRIDE_BYTES
        with sync._clock_storage(device, stream), triton.knobs.compilation.scope():
            triton.knobs.compilation.instrumentation_mode = ""
            self.outputs = torch.zeros((len(self.nodes), self.num_threads), device=device, dtype=torch.int32)
            self.baseline = torch.zeros(self.num_threads, device=device, dtype=torch.int32)
            self.last = torch.zeros_like(self.baseline)
            self._dummy_inputs = torch.tensor([self.last.data_ptr()], device=device, dtype=torch.int64)
            entries = [[self.baseline.data_ptr(), *(self.outputs[j].data_ptr() for j in deps)] for deps in self.entries]
            waits = [[self.outputs[j].data_ptr() for j in deps] for deps in self.waits]
            self.table, self.pointers = sync._make_launch_table(entries, waits,
                                                                [row.data_ptr() for row in self.outputs], device)
            self._begin_grid = triton.cdiv(max(1, len(self.nodes)) * self.num_threads, 128)
            self._end_grid = triton.cdiv(self.num_threads, 128)
            # Force num_inputs to i64, matching KernelLaunch's native ABI.
            self._begin = _graph_begin.warmup(self.outputs, self.baseline, self._dummy_inputs, 1 << 32,
                                              N=self.num_threads, NODES=len(self.nodes), BLOCK=128, num_warps=4,
                                              grid=(self._begin_grid, ))
            self._end = _graph_end.warmup(self.outputs, self.baseline, self.last, self.last, self.last,
                                          N=self.num_threads, NODES=len(self.nodes), BLOCK=128, num_warps=4,
                                          grid=(self._end_grid, ))
            self._ready = torch.cuda.Event()
            self._ready.record()
        _live_plans.add(self)

    def _boundary(self, inputs, count, stream_clock, completion):
        begin = _kernel_launch(self._begin, self._begin_grid,
                               (self.outputs.data_ptr(), self.baseline.data_ptr(), inputs.data_ptr(), count))
        end = _kernel_launch(self._end, self._end_grid,
                             (self.outputs.data_ptr(), self.baseline.data_ptr(), self.last.data_ptr(),
                              stream_clock.data_ptr(), completion.data_ptr()))
        return begin, end

    @property
    def boundary_kernels(self):
        return self._boundary(self._dummy_inputs, 1, self.last, self.last)

    def node_launch_args(self, node: int):
        if self.closed:
            raise RuntimeError("GSan graph plan is closed")
        if not 0 <= node < len(self.nodes) or not self.nodes[node].is_kernel:
            raise ValueError("Expected an instrumented graph kernel node")
        return self.global_state, self.table.data_ptr(), node

    def prepare_launch(self, stream: int, dependencies=()):
        import torch

        if self.closed:
            raise RuntimeError("GSan graph plan is closed")
        dependencies = tuple(dependencies)
        if any(dep.device != self.device for dep in dependencies):
            raise ValueError("GSan graph completion belongs to another device")
        _collect_pending()
        before, after = sync._next_stream_clocks(self.device, stream)
        with sync._clock_storage(self.device, stream):
            torch.cuda.current_stream().wait_event(self._ready)
            inputs = torch.tensor(
                [before.data_ptr(),
                 self.last.data_ptr(), *(dep.clock.data_ptr() for dep in dependencies)], device=self.device,
                dtype=torch.int64)
            completion = Completion(self.device, torch.empty_like(self.baseline))
            begin, end = self._boundary(inputs, len(dependencies) + 2, after, completion.clock)
            # Commit the reservation only after every fallible preparation step.
            # External launchers must serialize preparation and submission.
            sync._reserve_stream_clocks(self.device, stream)
        return GraphLaunch(self, stream, inputs, dependencies, completion, begin, end, before, after)

    def close(self):
        """Release caller ownership; submitted launches retain their storage."""
        self.closed = True


def record_completion(device: int, stream: int) -> Completion:
    """Copy stream ordering before the caller records its CUDA completion event."""
    clock = sync._current_stream_clock(device, stream)
    with sync._clock_storage(device, stream):
        result = Completion(device, clock.clone())
    _retain_until_stream_completes(device, stream, result)
    return result


def acquire_completions(device: int, stream: int, dependencies):
    """Import snapshots after the caller has enqueued the matching CUDA waits."""
    import torch

    dependencies = tuple(dependencies)
    if not dependencies:
        return
    if any(dep.device != device for dep in dependencies):
        raise ValueError("GSan completion belongs to another device")
    before, after = sync._next_stream_clocks(device, stream)
    with sync._clock_storage(device, stream), triton.knobs.compilation.scope():
        triton.knobs.compilation.instrumentation_mode = ""
        inputs = torch.tensor([before.data_ptr(), *(dep.clock.data_ptr() for dep in dependencies)], device=device,
                              dtype=torch.int64)
        _merge_clocks[(triton.cdiv(before.numel(), 128), )](inputs, after, len(dependencies) + 1, N=before.numel(),
                                                            BLOCK=128, num_warps=4)
        sync._reserve_stream_clocks(device, stream)
    _retain_until_stream_completes(device, stream, (inputs, dependencies))


def _check_reset():
    _collect_pending()
    if _pending or any(not plan.closed for plan in _live_plans) or _live_completions:
        raise AssertionError("Release GSan graphs and completion snapshots before reset")
