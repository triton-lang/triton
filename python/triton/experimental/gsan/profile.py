"""Observed allocation access properties, not a race detector.

Profile dedicated profiling-pool storage lifetimes, including instrumented initialization.
Recommendations cover only observed paths, shapes and replays. Validate the
resulting allocation policy in a separate GSan run.
"""
from __future__ import annotations

from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import RLock, get_ident
from typing import TYPE_CHECKING
import weakref

import triton
import triton.language as tl
from triton._C.libtriton import gsan_profile as _layout
from triton.language.extra.cuda import libdevice

from ._profile_allocator import (ShareableHandleType, configure, install_allocator, is_installed, get_allocator, malloc,
                                 free_allocation, export_allocation_handles, import_allocation_handles,
                                 has_live_allocations, reset)

__all__ = ["AllocationProfile", "ShareableHandleType", "configure", "install_allocator", "is_installed", "get_allocator", "malloc",
           "free_allocation", "export_allocation_handles", "import_allocation_handles", "has_live_allocations", "reset"]

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist

_active: ContextVar[AllocationProfile | None] = ContextVar("gsan_allocation_profile", default=None)
# Compilation knobs are process-global, even though launch state is contextual.
_context_lock = RLock()


@triton.jit
def _or(left, right):
    return left | right


@triton.jit
def _summarize(cells, output, count: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(cells + offsets, offsets < count, 0).to(tl.uint32)
    flags = tl.reduce(values & _layout.FLAGS_MASK, 0, _or)
    written = tl.sum(libdevice.popc((values & _layout.WRITTEN_MASK).to(tl.int32)), 0)
    repeated = tl.sum(((values & _layout.REPEATED_WRITE) != 0).to(tl.int32), 0)
    uninitialized = tl.sum(((values & _layout.READ_BEFORE_WRITE) != 0).to(tl.int32), 0)
    read = tl.sum(((values & _layout.READ) != 0).to(tl.int32), 0)
    base = output + tl.program_id(0) * 5
    tl.store(base, flags)
    tl.store(base + 1, written)
    tl.store(base + 2, repeated)
    tl.store(base + 3, uninitialized)
    tl.store(base + 4, read)


@dataclass
class _Allocation:
    key: tuple
    nbytes: int
    allocation_id: str | None
    cells: dict[torch.device, torch.Tensor] = field(default_factory=dict)
    mappings: list[dict[str, int]] = field(default_factory=list)
    owners: list[object] = field(default_factory=list)
    # Used for alias identity only; native allocator hooks observe deallocation.
    storages: dict[int, weakref.ReferenceType] = field(default_factory=dict)
    summaries: dict[torch.device, tuple] = field(default_factory=dict)
    stats: list[int] | None = None
    lifetime: object | None = None
    lifetime_device: torch.device | None = None
    lifetime_enabled: bool = False
    owns_storage: bool = False
    errors: list[str] = field(default_factory=list)
    names: list[str] = field(default_factory=list)


def _launch_state() -> int:
    profile = _active.get()
    if profile is None:
        raise RuntimeError("gsan-profile requires an active AllocationProfile context")
    return profile.state_pointer


def _add_stats(total, part):
    return [total[0] | part[0], *(a + b for a, b in zip(total[1:], part[1:]))]


def _allocation_report(entry, stats, warnings):
    flags, written, repeated, uninitialized, read = stats
    observed = bool(flags & (_layout.READ | _layout.WRITE | _layout.ATOMIC))
    candidates = [1 << i for i in range(_layout.NUM_GRANULARITIES)
                  if not flags & (1 << (_layout.GRANULARITY_SHIFT + i))]
    granularity = max(candidates) if observed and candidates and not warnings else None
    reasons = list(warnings)
    for bit, reason in ((_layout.REPEATED_WRITE, "repeated writes"),
                        (_layout.READ_BEFORE_WRITE, "reads before an observed write"),
                        (_layout.ATOMIC, "atomic accesses")):
        if flags & bit:
            reasons.append(reason)
    if not observed:
        reasons.append("no accesses observed")
    elif not flags & _layout.WRITE:
        reasons.append("no non-atomic writes observed")
    if observed and not candidates:
        reasons.append("no supported access granularity")
    write_once = observed and not reasons
    nbytes = entry["nbytes"]
    mappings = entry["mappings"]
    return {
        "allocation_id": entry["allocation_id"],
        "names": entry["names"],
        # Preserve the single-device report field, without implying addresses
        # in different processes share a virtual address space.
        "address": mappings[0]["address"] if len(mappings) == 1 else None,
        "mappings": mappings,
        "nbytes": nbytes,
        "max_granularity": granularity,
        "write_once_candidate": write_once,
        "reasons": reasons,
        "bytes_written": written,
        "cells_read": read,
        "cells_with_repeated_writes": repeated,
        "cells_read_before_write": uninitialized,
        "profiling_metadata_bytes": triton.cdiv(nbytes, _layout.GRANULARITY) * _layout.CELL_SIZE_BYTES,
        "default_gsan_shadow_bytes": triton.cdiv(nbytes, 4) * 24,
        "recommended_normal_shadow_bytes": triton.cdiv(nbytes, granularity) * 24 if granularity else None,
        "recommended_write_once_shadow_bytes": triton.cdiv(nbytes, granularity) * 4 if write_once else None,
    }


class AllocationProfile:
    """Measure access granularity and write-once candidates on CUDA devices.

    Call :func:`install_allocator` at startup, before CUDA initialization.
    PyTorch tensor storage allocations on this context's creating thread and
    selected devices use profiling storage while the context is active. Raw-pointer
    allocations use the original allocator and are ignored, as are ordinary
    tensors created before entry. Internal DataPtr workspaces using the same
    allocation entry point as tensor storage may also use the profiling pool.
    Explicit third-party MemPools
    continue to use their original allocator.
    Select whole storages for reporting with :meth:`track`, or imported profiling-pool
    mappings with :meth:`track_pointer`. ``device`` on registration selects the *launch*
    device observing the mapping. ``allocation_id`` identifies one physical
    allocation across devices/processes; use a different ID for each replica.
    Without an ID, allocations in different processes are always distinct.

    ``devices`` enables multiple launch devices in one process. ``group`` is an
    optional torch.distributed process group (Gloo or NCCL); :meth:`report` is
    then collective and returns the same combined report on every group rank.
    Reporting is allowed only after the context exits. All participating ranks
    must exit their contexts before calling the collective report.

    Register each storage before its first instrumented access. Local storages
    can be registered throughout the context, including reused cache addresses.
    Shared allocation IDs and external mappings must be registered before
    :meth:`prepare` or the first launch. Preparation is collective with a group.
    Call prepare on every rank before conditional launches or graph capture.
    Validate policies in a separate GSan run using the proposed race-checking
    allocation pool. The race checker ignores profiling-pool allocations.
    Tracked PyTorch allocations are not retained. Their storage deleters queue
    summary and shadow reset before reuse. Side-stream dependencies are joined
    on a separate retirement stream, without blocking the allocation stream. Follow PyTorch's normal stream synchronization and record_stream
    rules for side-stream uses. Views and C++ references extend the storage
    lifetime naturally. Do not resize a
    registered storage in place. External mappings must remain alive through
    reporting; all remote accesses must finish before their owner frees storage.
    Keep this object and graph storage alive until captured executions finish.
    Allocate storage before capture. Releasing tracked storage during capture
    is unsupported and invalidates its
    recommendations. Contexts are single-use; registration and reporting must
    run on the creating thread, but storage destruction may run on another thread.
    """

    def __init__(self, device=None, *, devices=None, group: dist.ProcessGroup | None = None):
        from . import _profile_allocator as _allocator

        self._hooks = _allocator.get_allocator()
        if triton.runtime.driver.active.get_current_target().backend != "cuda":
            raise NotImplementedError("gsan-profile supports NVIDIA CUDA only")
        if device is not None and devices is not None:
            raise ValueError("specify device or devices, not both")
        requested = [device] if devices is None else list(devices)
        if not requested:
            raise ValueError("at least one CUDA device is required")
        self.devices = tuple(dict.fromkeys(self._device(d) for d in requested))
        self.device = self.devices[0]
        self.group = group
        if group is not None:
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank(group) < 0:
                raise ValueError("group must be an initialized process group containing this rank")
            if dist.get_backend(group) not in ("gloo", "nccl"):
                raise ValueError("allocation profiling requires a Gloo or NCCL process group")
        self._check_capture()
        self._allocations: dict[tuple, _Allocation] = {}
        self._reserve_base = _allocator.get_reserve_pointer()
        self._prepared = False
        self._stack = None
        self._entered = False
        self._failed = False
        self._thread = get_ident()
        self._lifetime_lock = RLock()
        self._next_lifetime = 0

    @staticmethod
    def _device(device):
        import torch
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", device)
        else:
            device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("AllocationProfile requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.index >= torch.cuda.device_count():
            raise ValueError("CUDA device index is out of range")
        return device

    def _check_thread(self):
        if get_ident() != self._thread:
            raise RuntimeError("use AllocationProfile from its creating host thread")

    def _check_capture(self):
        import torch
        for device in self.devices:
            with torch.cuda.device(device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("call AllocationProfile.prepare() before CUDA graph capture; "
                                       "registration and reporting cannot run during capture")

    def track(self, tensor, *, name: str, device=None, allocation_id: str | None = None):
        """Select the tensor's entire profiling-pool storage for reporting and return it.

        Tensors outside the profiling pool (including empty storages) are ignored.

        For a peer tensor, pass the device whose kernels will use its pointer.
        Peer access/mapping setup remains the caller's responsibility.
        """
        if tensor.device.type != "cuda":
            raise ValueError("profiled tensors must be CUDA tensors")
        storage = tensor.untyped_storage()
        observer = tensor.device if device is None else device
        self._track(storage.data_ptr(), storage.nbytes(), tensor, name, observer, allocation_id, storage)
        return tensor

    def track_pointer(self, address: int, nbytes: int, *, owner, name: str, device=None,
                      allocation_id: str | None = None) -> None:
        """Select a complete profiling-pool mapping on a launch device for reporting.

        Pointers outside the profiling pool are ignored.

        ``owner`` must keep the mapping alive. Give aliases of the same physical
        storage the same allocation_id, even when their virtual addresses differ.
        """
        observer = self.device if device is None else self._device(device)
        self._track(address, nbytes, owner, name, observer, allocation_id, None)

    def _track(self, address, nbytes, owner, name, observer, allocation_id, storage):
        # A destructor on another host thread must not race registration.
        from ._profile_allocator import _suspend
        with self._lifetime_lock, _suspend():
            self._track_locked(address, nbytes, owner, name, observer, allocation_id, storage)

    def _track_locked(self, address, nbytes, owner, name, observer, allocation_id, storage):
        import torch
        from . import _profile_allocator as _allocator
        from ._utils import uint8_cuda_tensor_from_ptr
        self._check_thread()
        if self._entered and self._stack is None:
            raise RuntimeError("cannot register allocations after the profile context exits")
        if self._prepared and (allocation_id is not None or storage is None):
            raise RuntimeError("register shared allocations and external mappings before prepare()")
        self._check_capture()
        observer = self._device(observer)
        if observer not in self.devices:
            raise ValueError("register the launch device in AllocationProfile(devices=...)")
        if type(address) is not int or type(nbytes) is not int or nbytes < 0 or address < 0 or \
                address + nbytes > (1 << 63) - 1 or (nbytes and (not address or address % _layout.GRANULARITY)):
            raise ValueError(f"profiled storage must have a nonnegative size and a {_layout.GRANULARITY}-byte-aligned base "
                             "within the CUDA address space")
        if not nbytes:
            return
        layout = _allocator.get_allocation_profile_layout(address)
        if layout is None:
            return
        base, size, shadow = layout
        if address + nbytes > base + size:
            raise ValueError("profiled storage extends beyond its GSan allocation")
        if owner is None:
            raise ValueError("an owner is required for a tracked allocation")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a nonempty string")
        key = ("lifetime", self._next_lifetime)
        self._next_lifetime += 1
        if allocation_id is not None:
            if not isinstance(allocation_id, str) or not allocation_id:
                raise ValueError("allocation_id must be a nonempty string")
            key = ("shared", allocation_id)
        elif storage is not None:
            for candidate in self._allocations.values():
                reference = candidate.storages.get(storage._cdata)
                if reference is not None and reference() is storage and not self._is_retired(candidate):
                    key = candidate.key
                    break
        else:
            key = ("mapping", observer.index, address, nbytes)
        for allocation in self._allocations.values():
            if self._is_retired(allocation):
                continue
            for mapping in allocation.mappings:
                if mapping["device"] != observer.index:
                    continue
                start = mapping["address"]
                if (address, nbytes) == (start, allocation.nbytes):
                    if allocation.key != key:
                        raise ValueError("aliases must use the same allocation_id")
                elif max(address, start) < min(address + nbytes, start + allocation.nbytes):
                    raise ValueError("overlapping ranges must be registered as one complete allocation")
        allocation = self._allocations.get(key)
        if allocation is None:
            allocation = _Allocation(key, nbytes, allocation_id)
        elif self._is_retired(allocation):
            raise ValueError("use a new allocation_id for each storage lifetime")
        elif allocation.nbytes != nbytes:
            raise ValueError("all mappings of an allocation_id must have the same size")
        if observer not in allocation.cells:
            with torch.cuda.device(observer):
                count = triton.cdiv(nbytes, _layout.GRANULARITY)
                allocation.cells[observer] = uint8_cuda_tensor_from_ptr(
                    shadow, count * _layout.CELL_SIZE_BYTES, observer.index).view(torch.int32)
        mapping = {"device": observer.index, "address": address}
        if mapping not in allocation.mappings:
            allocation.mappings.append(mapping)
        lifetime_storage = storage
        if lifetime_storage is None and isinstance(owner, torch.Tensor):
            lifetime_storage = owner.untyped_storage()
            if not (lifetime_storage.data_ptr() <= address and
                    address + nbytes <= lifetime_storage.data_ptr() + lifetime_storage.nbytes()):
                raise ValueError("the tensor owner must contain the tracked range")
        if lifetime_storage is not None and lifetime_storage._cdata not in allocation.storages:
            # Native profiling storages carry their own retirement deleter.
            # Borrowed/imported mappings retain the external owner contract.
            owns_storage = self._hooks.owns(lifetime_storage.data_ptr())
            if owns_storage and allocation.owns_storage:
                raise ValueError("an allocation_id must identify one physical allocation, not replicas")
            allocation.owns_storage |= owns_storage
            allocation.storages[lifetime_storage._cdata] = weakref.ref(lifetime_storage)
        elif lifetime_storage is None:
            allocation.owners.append(owner)
        if name not in allocation.names:
            allocation.names.append(name)
        self._allocations[key] = allocation
        self._prepare_summary(allocation, observer)
        if allocation.owns_storage and allocation.lifetime is None:
            device = lifetime_storage.device
            if device not in allocation.cells:
                allocation.cells[device] = uint8_cuda_tensor_from_ptr(
                    shadow, triton.cdiv(nbytes, _layout.GRANULARITY) * _layout.CELL_SIZE_BYTES,
                    device.index).view(torch.int32)
            self._prepare_summary(allocation, device)
            summary, kernel = allocation.summaries[device]
            with torch.cuda.device(device):
                allocation.lifetime = self._hooks.watch(
                    lifetime_storage.data_ptr(), shadow, device.index, kernel.function,
                    summary.data_ptr(), summary.shape[0],
                    kernel.metadata.num_warps * 32, kernel.metadata.shared,
                    (allocation.cells[device], summary, kernel))
            allocation.lifetime_device = device
            if self._prepared:
                allocation.lifetime_enabled = self._hooks.enable(allocation.lifetime)
        # Storage deleters reset untracked allocations too. Reuse follows the
        # reset without a host-side synchronization.

    def _is_retired(self, allocation):
        return allocation.stats is not None or (allocation.lifetime is not None and
                                               self._hooks.released(allocation.lifetime))

    def _prepare_summary(self, allocation, device):
        import torch
        if device in allocation.summaries:
            return
        cells = allocation.cells[device]
        with torch.cuda.device(device), triton.knobs.compilation.scope():
            triton.knobs.compilation.instrumentation_mode = ""
            summary = torch.empty((triton.cdiv(cells.numel(), 1024), 5), dtype=torch.int64, device=device)
            kernel = _summarize.warmup(cells, summary, cells.numel(), BLOCK=1024, grid=(summary.shape[0],))
            kernel._init_handles()
        allocation.summaries[device] = (summary, kernel)

    def _release_buffers(self, allocation):
        # Drop the native record first: it owns metadata buffers until any
        # queued reduction completes, without retaining the profiled storage.
        allocation.lifetime = None
        allocation.storages.clear()
        allocation.cells.clear()
        allocation.summaries.clear()

    def _finalize(self, allocation):
        import torch
        if allocation.stats is not None:
            return allocation.stats
        if allocation.lifetime is not None:
            try:
                device = allocation.lifetime_device
                with torch.cuda.device(device):
                    self._hooks.finalize(allocation.lifetime, torch.cuda.current_stream().cuda_stream)
                if not allocation.lifetime_enabled:
                    return [0, 0, 0, 0, 0]
                summary = allocation.summaries[device][0]
                result = [0, 0, 0, 0, 0]
                for row in summary.cpu().tolist():
                    result = _add_stats(result, row)
                return result
            except RuntimeError as exc:
                allocation.errors.append(f"storage retirement failed: {exc}")
                allocation.stats = [0, 0, 0, 0, 0]
                return allocation.stats
        return self._device_stats(allocation) if self._prepared else [0, 0, 0, 0, 0]

    def _manifest(self):
        rank = 0
        if self.group is not None:
            import torch.distributed as dist
            rank = dist.get_rank(self.group)
        local = {}
        for i, allocation in enumerate(self._allocations.values()):
            key = ("shared", allocation.allocation_id) if allocation.allocation_id is not None else ("local", rank, i)
            local[key] = allocation
        manifest = [{
            "key": key, "allocation_id": a.allocation_id, "nbytes": a.nbytes, "names": list(a.names),
            "mappings": [dict(m, rank=rank) for m in a.mappings], "owner_rank": rank,
            "owns_storage": a.owns_storage, "errors": list(a.errors)
        } for key, a in local.items()]
        return rank, local, manifest

    @staticmethod
    def _combine_manifests(packets):
        entries = {}
        for packet in packets:
            for entry in packet:
                key = entry["key"]
                if key not in entries:
                    entries[key] = entry
                    continue
                previous = entries[key]
                if previous["nbytes"] != entry["nbytes"]:
                    raise ValueError("all mappings of an allocation_id must have the same size across ranks")
                if previous["owns_storage"] and entry["owns_storage"]:
                    raise ValueError("an allocation_id must identify one physical allocation, not replicas")
                if entry["owns_storage"]:
                    previous["owner_rank"] = entry["owner_rank"]
                    previous["owns_storage"] = True
                previous["names"] = list(dict.fromkeys(previous["names"] + entry["names"]))
                previous["mappings"].extend(entry["mappings"])
                previous["errors"].extend(entry["errors"])
        return entries

    def prepare(self) -> None:
        """Reset selected shadow cells before launching; collective with a group.

        Exactly one rank initializes each shared allocation. A completion
        collective publishes these resets before any rank can launch kernels.
        """
        self._check_thread()
        if self._prepared:
            return
        if _active.get() not in (None, self):
            raise RuntimeError("cannot prepare while another AllocationProfile is active")
        with ExitStack() as stack:
            if _active.get() is not self:
                if not _context_lock.acquire(blocking=False):
                    raise RuntimeError("cannot prepare while another AllocationProfile is active")
                stack.callback(_context_lock.release)
            with self._lifetime_lock:
                self._prepare()

    def _prepare(self):
        import torch
        self._check_capture()
        # Finish allocator initialization (including async shadow zeroing) on
        # every rank before the elected owner resets any shared shadow.
        for device in self.devices:
            torch.cuda.synchronize(device)
        rank, local, manifest = self._manifest()
        self._entries = self._combine_manifests(self._gather(manifest))
        # Also clear live native storage that may be registered later in this
        # context. A tensor kept from an earlier context must not inherit its
        # observations merely because registration follows prepare().
        self._hooks.reset_metadata([device.index for device in self.devices])
        for key, allocation in local.items():
            if self._entries[key]["owner_rank"] == rank and not self._is_retired(allocation):
                # Local mappings are aliases of the same physical shadow.
                next(iter(allocation.cells.values())).zero_()
        for device in self.devices:
            torch.cuda.synchronize(device)
        self._gather(None)
        for allocation in local.values():
            if allocation.lifetime is not None:
                allocation.lifetime_enabled = self._hooks.enable(allocation.lifetime)
        self._prepared = True

    @property
    def state_pointer(self) -> int:
        """Hidden instrumentation argument for the current CUDA launch device."""
        import torch
        self._check_thread()
        if _active.get() is not self:
            raise RuntimeError("gsan-profile requires an active AllocationProfile context")
        device = torch.device("cuda", torch.cuda.current_device())
        if device not in self.devices:
            raise RuntimeError("gsan-profile kernel launched on an unregistered CUDA device")
        self.prepare()
        return self._reserve_base

    def __enter__(self):
        self._check_thread()
        if self._entered or _active.get() is not None:
            raise RuntimeError("AllocationProfile contexts cannot be nested or re-entered")
        if not _context_lock.acquire(blocking=False):
            raise RuntimeError("concurrent AllocationProfile contexts are unsupported")
        with ExitStack() as stack:
            stack.callback(_context_lock.release)
            if triton.knobs.compilation.instrumentation_mode not in ("", "gsan-profile"):
                raise RuntimeError("gsan-profile must run separately from other instrumentation")
            stack.enter_context(triton.knobs.compilation.scope())
            triton.knobs.compilation.instrumentation_mode = "gsan-profile"
            token = _active.set(self)
            stack.callback(_active.reset, token)
            self._hooks.set_enabled(True, [device.index for device in self.devices])
            stack.callback(self._hooks.set_enabled, False, [])
            self._stack = stack.pop_all()
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        import torch
        self._check_thread()
        stack, self._stack = self._stack, None
        self._failed |= exc_type is not None
        self._hooks.set_enabled(False, [])
        with stack:
            try:
                for device in self.devices:
                    torch.cuda.synchronize(device)
                with self._lifetime_lock:
                    for allocation in self._allocations.values():
                        # Shared metadata needs the report's completion collective.
                        if self.group is None or allocation.allocation_id is None:
                            allocation.stats = self._finalize(allocation)
                            self._release_buffers(allocation)
            except BaseException:
                self._failed = True
                raise
        return False

    def _gather(self, value):
        if self.group is None:
            return [value]
        import torch.distributed as dist
        result = [None] * dist.get_world_size(self.group)
        import torch
        from ._profile_allocator import _suspend
        with torch.cuda.device(self.device), _suspend():
            dist.all_gather_object(result, value, group=self.group)
        return result

    def _device_stats(self, allocation):
        import torch
        cells = next(iter(allocation.cells.values()))
        summary, kernel = allocation.summaries[cells.device]
        with torch.cuda.device(cells.device):
            # The compiled kernel has instrumentation disabled. Calling it
            # directly avoids changing process-global knobs from a destructor.
            kernel[(summary.shape[0], 1, 1)](cells, summary, cells.numel(), 1024)
            result = [0, 0, 0, 0, 0]
            for row in summary.cpu().tolist():
                result = _add_stats(result, row)
            return result

    def report(self) -> dict:
        """Return final results after context exit; collective with a group.

        Calls before entry or while the context is active are rejected. Results
        remain available on repeated calls after exit.
        """
        self._check_thread()
        if not self._entered or self._stack is not None:
            raise RuntimeError("report() is only available after the AllocationProfile context exits")
        if _active.get() is not None:
            raise RuntimeError("cannot report while another AllocationProfile is active")
        with ExitStack() as stack:
            if not _context_lock.acquire(blocking=False):
                raise RuntimeError("cannot report while another AllocationProfile is active")
            stack.callback(_context_lock.release)
            stack.enter_context(triton.knobs.compilation.scope())
            # The JIT reads its mode from knobs, not from launch keyword args.
            triton.knobs.compilation.instrumentation_mode = ""
            from ._profile_allocator import _suspend
            with self._lifetime_lock, _suspend():
                return self._report()

    def _report(self) -> dict:
        """Finalize shared shadow after every participating device finishes."""
        import torch
        self._check_capture()
        for device in self.devices:
            torch.cuda.synchronize(device)
        # This collective must precede shadow reads: another rank may still be
        # writing the same physical cells after this rank's local synchronization.
        failures = self._gather(self._failed)
        warnings = [(f"rank {rank}: " if self.group is not None else "") +
                    "profile context exited with an exception" for rank, failed in enumerate(failures) if failed]
        rank, local, manifest = self._manifest()
        # Local lifetimes may have been registered or retired since prepare().
        self._entries = self._combine_manifests(self._gather(manifest))
        stats = {key: self._finalize(allocation)
                 for key, allocation in local.items() if self._entries[key]["owner_rank"] == rank}
        for key, allocation in local.items():
            if key in stats:
                allocation.stats = stats[key]
            self._release_buffers(allocation)
        combined = {}
        local_errors = {key: list(allocation.errors) for key, allocation in local.items() if key in stats}
        for packet, errors in self._gather((stats, local_errors)):
            combined.update(packet)
            for key, values in errors.items():
                self._entries[key]["errors"] = list(dict.fromkeys(self._entries[key]["errors"] + values))
        return {
            "schema_version": 1, "mode": "gsan-profile", "scope": "observed execution only; not a race check",
            "world_size": len(failures), "warnings": warnings,
            "allocations": [_allocation_report(entry, combined[key], warnings + entry["errors"])
                            for key, entry in self._entries.items()]
        }
