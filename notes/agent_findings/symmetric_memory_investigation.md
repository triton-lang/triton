# Symmetric Memory Investigation

## Confirmed Facts

### Two Available Implementations

**1. `torch.distributed._symmetric_memory` (PyTorch built-in)**
- Production-ready API available from PyTorch 2.3+
- Core functions: `empty()`, `rendezvous()`, `get_buffer()`, `barrier()`
- Backed by CUDA driver APIs: `cuMemAddressReserve`, `cuMemMap` for UVA mapping
- `rendezvous()` coordinates buffer sharing across ranks via torch.distributed

**2. `triton.experimental.gsan` (Triton's GSan layer)**
- Located at `python/triton/experimental/gsan/symmetric_memory.py`
- Wraps PyTorch's implementation with shadow memory for race detection
- Adds vector clock tracking for happens-before analysis
- Unix domain socket handshake for FD exchange in rendezvous (when `USE_TORCH_DIST=False`)
- `GSanSymmetricMemoryHandle` wraps the real handle + shadow memory

### API Usage

```python
import torch.distributed._symmetric_memory as symm_mem

# Allocate: all participating ranks allocate same-shape buffer
buf = symm_mem.empty((size,), dtype=torch.float16, device=device)

# Rendezvous: establishes peer-to-peer memory mappings
hdl = symm_mem.rendezvous(buf, group=dist.group.WORLD)

# Get peer buffer: returns a tensor view backed by peer GPU's memory
peer_buf = hdl.get_buffer(rank=r, sizes=[size], dtype=torch.float16)

# Barrier: waits for all ranks to reach this point
# channel=0 for the first barrier, channel=1 for the second, etc.
hdl.barrier(channel=0)
```

### How Peer Buffer Pointers Work

`hdl.get_buffer(r, sizes, dtype)` returns a PyTorch tensor whose `data_ptr()` is the UVA-mapped address of GPU r's buffer. This address is accessible from the current GPU via NVLink (intra-node) or PCIe (with GPU Direct RDMA for inter-node, if supported).

Inside a Triton kernel:
```python
peer_tensor = hdl.get_buffer(r, ...)
# Pass peer_tensor directly as a kernel argument — Triton receives its data_ptr
# as the pointer argument. The kernel's tl.store to this pointer writes directly
# to the peer GPU's memory.
```

### Barrier Semantics

`hdl.barrier(channel=N)` is a **host-side operation**:
1. Calls `torch.distributed.barrier()` to synchronize all CPU threads
2. Syncs CUDA streams: `torch.cuda.synchronize()`
3. This creates an "epoch boundary" — no GPU-side visibility guarantee exists
   unless CUDA synchronization has completed before the next kernel launch

**Critical implication:** The pipeliner cannot reason across this barrier. It is not an IR-level op; it is invisible to the compiler. The compiler treats the two kernels (gemm_scatter and reduce) as completely independent compilation units.

### No Native Distributed Ops in Triton IR

Search results confirm: there is **no** `tt.ReduceScatterOp`, `tt.AllReduceOp`, or `tt.AllGatherOp` in Triton's IR dialects.

The Triton IR ops `TT_ReduceOp`, `TT_GatherOp`, and `TT_ScatterOp` are **intra-GPU** operations (reduce across warps/threads within one kernel). They are not distributed collectives.

Cross-GPU communication in Triton is achieved entirely via:
- Peer buffer pointers passed as regular pointer arguments
- `tl.store` / `tl.atomic_add` to peer-mapped addresses
- Host-side barriers between kernel launches

### Reference Implementation: `triton_kernels/distributed.py`

The `_convert_dp_to_ep` kernel (lines ~155-175) shows the pattern:
```python
@triton.jit
def _convert_dp_to_ep(peer_dst_ptrs, ...):
    for dst_rank in tl.static_range(N_RANKS):
        # peer_dst_ptrs[dst_rank] is a pointer-typed kernel argument (one per rank)
        peer_dst_ptr = peer_dst_ptrs[dst_rank].to(tl.int64, bitcast=True)
        dst_row_ptrs = tl.where(dst_rank == expt_ranks, peer_dst_ptr, dst_row_ptrs)
    # ... tl.store(dst_row_ptrs, data)
```

The `tl.static_range(N_RANKS)` unrolls at compile time; each `peer_dst_ptrs[dst_rank]` is a separate variadic pointer argument. The bitcast to `tl.int64` allows arithmetic on the address.

## Observed Behavior

- `symm_mem.empty()` requires a CUDA device and `torch.distributed` to be initialized (or at least `WORLD_SIZE=1` locally)
- For single-GPU simulation (world_size=1), the peer buffer for rank 0 is the same tensor as the local buffer — no actual peer access occurs
- `rendezvous()` with world_size=1 may be a no-op or may require a trivial barrier

## Hypotheses

- The symmetric memory barrier uses `cudaStreamSynchronize` under the hood, which means the GPU must be idle before the barrier completes. This confirms no kernel-level overlap across the barrier.
- For NVLink-connected GPUs, `tl.store` to a peer address has comparable latency to a local L2 cache miss (~100-300 ns). For PCIe, it is 10-100x slower.

## Open Questions

- Does `hdl.get_buffer()` return a tensor that Triton can pass directly as a typed pointer argument, or must it be passed as int64 and cast inside the kernel?
- For world_size=1, does `rendezvous()` require the process group to be initialized?

## Next Actions

- Test `symm_mem.empty()` with world_size=1 after `torch.distributed.init_process_group()` with "nccl" or "gloo" backend
- Verify peer buffer pointer type when passed to Triton kernel warmup
