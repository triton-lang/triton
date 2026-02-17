from __future__ import annotations

import atexit
import array
import contextlib
import functools
import math
import os
import socket
import struct
import tempfile
import time
import uuid
import weakref
from collections.abc import Sequence
from typing import TypeAlias
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

from ._allocator import create_mem_pool, export_allocation_handles, free_allocation, import_allocation_handles

_CONSTRUCT_STORAGE_FROM_DATA_POINTER = getattr(torch._C, "_construct_storage_from_data_pointer", None)
_RendezvousCacheKey: TypeAlias = tuple[int, int, int]


def _uint8_cuda_tensor_from_ptr(data_ptr: int, numel: int, device_index: int) -> torch.Tensor:
    if _CONSTRUCT_STORAGE_FROM_DATA_POINTER is None:
        raise RuntimeError("torch._C._construct_storage_from_data_pointer is unavailable.")
    device = torch.device("cuda", device_index)
    storage = _CONSTRUCT_STORAGE_FROM_DATA_POINTER(data_ptr, device, numel)
    return torch.empty(0, dtype=torch.uint8, device=device).set_(storage, 0, (numel, ), (1, ))


def _normalize_size(size: tuple[object, ...]) -> tuple[int, ...]:
    if len(size) == 1 and isinstance(size[0], Sequence):
        size = size[0]
    return tuple(int(v) for v in size)


@functools.lru_cache()
def _get_mem_pool(device_index: int):
    _ = device_index
    return create_mem_pool()


@atexit.register
def _clear_mem_pool_cache() -> None:
    _get_mem_pool.cache_clear()


def empty(*size, dtype: torch.dtype | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    shape = _normalize_size(size)
    if dtype is None:
        dtype = torch.get_default_dtype()

    dev: torch.device
    if device is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    else:
        dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError("GSan symmetric memory requires CUDA tensors.")
    device_index = torch.cuda.current_device() if dev.index is None else dev.index
    dev = torch.device("cuda", device_index)

    with torch.cuda.device(dev), torch.cuda.use_mem_pool(_get_mem_pool(device_index)):
        return torch.empty(shape, dtype=dtype, device=dev)


def _resolve_group(group) -> tuple[dist.ProcessGroup, str]:
    if isinstance(group, dist.ProcessGroup):
        return group, str(group.group_name)
    if isinstance(group, str):
        return c10d._resolve_process_group(group), group
    raise TypeError(f"rendezvous: unsupported group type: {type(group)}")


def _send_fds(sock: socket.socket, src_rank: int, real_fd: int, shadow_fd: int) -> None:
    msg = struct.pack("i", src_rank)
    fds = array.array("i", [real_fd, shadow_fd])
    sock.sendmsg([msg], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())])


def _recv_fds(sock: socket.socket) -> tuple[int, int, int]:
    int_size = array.array("i").itemsize
    msg, ancdata, _, _ = sock.recvmsg(4, socket.CMSG_SPACE(2 * int_size))
    if len(msg) != 4:
        raise RuntimeError("Failed to receive rank metadata during rendezvous.")
    src_rank = struct.unpack("i", msg)[0]
    recv_fds: list[int] = []
    for level, ctype, data in ancdata:
        if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
            arr = array.array("i")
            arr.frombytes(data[:len(data) - (len(data) % int_size)])
            recv_fds.extend(arr.tolist())
    if len(recv_fds) < 2:
        raise RuntimeError("Failed to receive file descriptors during rendezvous.")
    return src_rank, recv_fds[0], recv_fds[1]


def _import_peer_ptrs(
    *,
    received_fds: dict[int, tuple[int, int]],
    metas: list[dict],
    rank: int,
    world_size: int,
    device_index: int,
    base_ptr: int,
) -> tuple[int, ...]:
    peer_ptrs = [0] * world_size
    peer_ptrs[rank] = base_ptr
    try:
        for peer in range(world_size):
            if peer == rank:
                continue
            peer_real_fd, peer_shadow_fd = received_fds[peer]
            ptr = import_allocation_handles(
                peer_real_fd,
                peer_shadow_fd,
                int(metas[peer]["alloc_size"]),
                device_index,
            )
            peer_ptrs[peer] = ptr
    except Exception:
        for peer, ptr in enumerate(peer_ptrs):
            if peer != rank:
                free_allocation(ptr, device_index)
        raise
    finally:
        pending_fds = tuple(received_fds.values())
        for peer_real_fd, peer_shadow_fd in pending_fds:
            os.close(peer_real_fd)
            os.close(peer_shadow_fd)
    return tuple(peer_ptrs)


class GSanSymmetricMemoryHandle:

    def __init__(
        self,
        *,
        group: dist.ProcessGroup,
        rank: int,
        world_size: int,
        device_index: int,
        buffer_size: int,
        peer_ptrs: tuple[int, ...],
        cache_key: _RendezvousCacheKey | None = None,
    ):
        self._group = group
        self._rank = rank
        self._world_size = world_size
        self._device_index = device_index
        self._buffer_size = buffer_size
        self._peer_ptrs = tuple(peer_ptrs)
        self._cache_key = cache_key
        self._closed = False

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def barrier(self, channel: int = 0, timeout_ms: int = 0) -> None:
        if self._closed:
            raise RuntimeError("GSanSymmetricMemoryHandle has been closed.")
        if channel != 0:
            raise NotImplementedError("Only channel=0 is supported in GSan symmetric memory.")
        _ = timeout_ms
        dist.barrier(group=self._group)

    def get_buffer(
        self,
        rank: int,
        sizes: Sequence[int],
        dtype: torch.dtype,
        storage_offset: int = 0,
    ) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("GSanSymmetricMemoryHandle has been closed.")
        if rank < 0 or rank >= self._world_size:
            raise ValueError(f"Invalid peer rank: {rank}")
        if storage_offset < 0:
            raise ValueError(f"storage_offset must be >= 0, got {storage_offset}")

        shape = tuple(int(v) for v in sizes)
        element_size = torch.empty((), dtype=dtype).element_size()
        offset_bytes = storage_offset * element_size
        req_bytes = math.prod(shape) * element_size
        if offset_bytes + req_bytes > self._buffer_size:
            raise ValueError(
                f"Requested slice ({offset_bytes + req_bytes} bytes) exceeds buffer size {self._buffer_size} bytes.")

        base_ptr = self._peer_ptrs[rank]
        if base_ptr == 0:
            raise RuntimeError(f"Peer rank {rank} has no mapped buffer.")

        byte_tensor = _uint8_cuda_tensor_from_ptr(base_ptr + offset_bytes, req_bytes, self._device_index)
        return byte_tensor.view(dtype=dtype).reshape(shape)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for rank, ptr in enumerate(self._peer_ptrs):
            if rank != self._rank:
                free_allocation(ptr, self._device_index)
        if self._cache_key is not None:
            _RENDEZVOUS_CACHE.pop(self._cache_key)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _wait_to_exist(path, timeout=10):
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.001)


_RENDEZVOUS_CACHE: weakref.WeakValueDictionary[_RendezvousCacheKey,
                                               GSanSymmetricMemoryHandle] = weakref.WeakValueDictionary()


def rendezvous(tensor: torch.Tensor, group) -> GSanSymmetricMemoryHandle:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("rendezvous: tensor must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise RuntimeError("rendezvous: tensor must be on CUDA device")
    if tensor.storage_offset() != 0:
        raise RuntimeError("rendezvous: tensor must have storage_offset() == 0")
    if not tensor.is_contiguous():
        raise RuntimeError("rendezvous: tensor must be contiguous")

    process_group, _ = _resolve_group(group)
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    device_index = tensor.device.index
    if device_index is None:
        raise RuntimeError("rendezvous: CUDA tensor must have concrete device index")

    storage = tensor.untyped_storage()
    base_ptr = storage.data_ptr()
    storage_key = int(getattr(storage, "_cdata", base_ptr))
    cache_key = (base_ptr, storage_key, id(process_group))
    cached = _RENDEZVOUS_CACHE.get(cache_key)
    if cached is not None and not cached._closed:
        return cached
    _RENDEZVOUS_CACHE.pop(cache_key, None)

    buffer_size = tensor.untyped_storage().nbytes()

    if world_size == 1:
        handle = GSanSymmetricMemoryHandle(
            group=process_group,
            rank=rank,
            world_size=world_size,
            device_index=device_index,
            buffer_size=buffer_size,
            peer_ptrs=[base_ptr],
            imported_peer_ptrs=[],
            cache_key=cache_key,
        )
        _RENDEZVOUS_CACHE[cache_key] = handle
        return handle

    with contextlib.ExitStack() as stack:
        real_fd, shadow_fd, alloc_size = export_allocation_handles(base_ptr)

        stack.callback(os.close, real_fd)
        stack.callback(os.close, shadow_fd)

        local_meta = {
            "hostname": socket.gethostname(),
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device_type": tensor.device.type,
            "nbytes": buffer_size,
            "alloc_size": int(alloc_size),
        }

        metas: list[dict] = [None] * world_size  # type: ignore[assignment]
        dist.all_gather_object(metas, local_meta, group=process_group)

        first = metas[0]
        for i, meta in enumerate(metas):
            if meta["hostname"] != first["hostname"]:
                raise RuntimeError(
                    f"rendezvous: rank {i} is on host {meta['hostname']}, expected single-node host {first['hostname']}"
                )
            if meta["shape"] != first["shape"] or meta["dtype"] != first["dtype"]:
                raise RuntimeError("rendezvous: all ranks must use tensors with identical shape and dtype.")
            if meta["device_type"] != "cuda":
                raise RuntimeError("rendezvous: all ranks must use CUDA tensors.")
            if meta["nbytes"] != first["nbytes"]:
                raise RuntimeError("rendezvous: all ranks must use tensors with identical byte size.")
            if meta["alloc_size"] != first["alloc_size"]:
                raise RuntimeError("rendezvous: all ranks must use identical GSan allocation sizes.")

        token_holder = [uuid.uuid4().hex if rank == 0 else None]
        dist.broadcast_object_list(token_holder, group=process_group, group_src=0)
        token = str(token_holder[0])

        def make_socket_path(rank):
            return Path(tempfile.gettempdir()) / f"triton-gsan-{token[:16]}-{rank}.sock"

        socket_path = make_socket_path(rank)
        socket_path.unlink(missing_ok=True)
        stack.callback(lambda: socket_path.unlink(missing_ok=True))

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        stack.push(listener)
        listener.bind(str(socket_path))
        listener.listen(world_size)

        received_fds: dict[int, tuple[int, int]] = {}
        pending: dict[int, tuple[socket.socket, int, int]] = {}

        try:
            for peer in range(rank):
                while peer not in pending:
                    conn, _ = listener.accept()
                    src_rank, peer_real_fd, peer_shadow_fd = _recv_fds(conn)
                    pending[src_rank] = (conn, peer_real_fd, peer_shadow_fd)
                conn, peer_real_fd, peer_shadow_fd = pending.pop(peer)
                received_fds[peer] = (peer_real_fd, peer_shadow_fd)
                _send_fds(conn, rank, real_fd, shadow_fd)
                conn.close()
            for peer in range(rank + 1, world_size):
                peer_socket_path = make_socket_path(peer)
                # Connection is initiated by lower ranked peer
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                    _wait_to_exist(peer_socket_path)
                    conn.settimeout(10)
                    conn.connect(str(peer_socket_path))
                    _send_fds(conn, rank, real_fd, shadow_fd)
                    src_rank, peer_real_fd, peer_shadow_fd = _recv_fds(conn)
                    if src_rank != peer:
                        raise RuntimeError(f"Unexpected peer rank {src_rank}, expected {peer}")
                    received_fds[peer] = (peer_real_fd, peer_shadow_fd)
        except Exception:
            for peer_real_fd, peer_shadow_fd in received_fds.values():
                os.close(peer_real_fd)
                os.close(peer_shadow_fd)
            received_fds.clear()
            raise
        finally:
            for conn, _, _ in pending.values():
                conn.close()

        peer_ptrs = _import_peer_ptrs(
            received_fds=received_fds,
            metas=metas,
            rank=rank,
            world_size=world_size,
            device_index=int(device_index),
            base_ptr=base_ptr,
        )
        dist.barrier(group=process_group)

    handle = GSanSymmetricMemoryHandle(
        group=process_group,
        rank=rank,
        world_size=world_size,
        device_index=device_index,
        buffer_size=buffer_size,
        peer_ptrs=peer_ptrs,
        cache_key=cache_key,
    )
    _RENDEZVOUS_CACHE[cache_key] = handle
    return handle
