"""
GEMM + Reduce-Scatter fused kernel in Triton.
Uses torch.distributed symmetric memory for peer-to-peer buffer access.

Design Overview
---------------
Each GPU holds:
  - A_local : (M, K_local)  where K_local = K // WORLD_SIZE
  - B       : (K_local, N)  (full weight shard for column-parallel linear)

After GEMM, each GPU has a partial result P_r = A_local @ B of shape (M, N).
Reduce-scatter assigns GPU r ownership of rows [r*M_SHARD : (r+1)*M_SHARD] of
sum_r P_r.

Implementation (two kernels + host barrier):

  Kernel 1 (gemm_scatter_kernel):
    - Inner k-loop: software-pipelined GEMM (loads hidden behind dots)
    - Post-loop:    scatter store to destination rank's symmetric memory buffer
                    → pipeliner CANNOT touch this (accumulator dependency)

  [host] hdl.barrier(channel=0)  ← epoch boundary between GPUs

  Kernel 2 (reduce_kernel):
    - Reads WORLD_SIZE partial tiles from local symm mem buffer
    - Sums them into final output (each GPU owns its M_SHARD rows)

IR Analysis Goal
----------------
Run with MLIR_ENABLE_DUMP=1 to observe:
  BEFORE TritonGPUPipeline: tt.load × 2, tt.dot, tt.store (all synchronous)
  AFTER  TritonGPUPipeline: ttg.async_copy_global_to_local × 2, ttng.async_wait,
                             tt.dot (now overlapped with loads)
  UNCHANGED:                 the tt.store to peer symm mem buffer
                             (depends on fully-accumulated acc, not pipelineable)
"""

from __future__ import annotations

import os
import sys
import argparse
import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Problem shape
# ---------------------------------------------------------------------------
M = 1024
N = 1024
K = 1024
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
NUM_STAGES = 3  # pipeline depth for inner GEMM loop


# ===========================================================================
# Kernel 1: GEMM + Scatter Store
# ===========================================================================
@triton.jit
def gemm_scatter_kernel(
    # Input matrices
    a_ptr,          # float16 (M, K_local)
    b_ptr,          # float16 (K_local, N)
    # Scatter output buffer
    # Single-GPU: pointer to local output buffer
    # Multi-GPU:  pre-selected peer symm mem buffer pointer for this CTA's
    #             destination rank (selected by host before launch)
    out_ptr,        # float16 (WORLD_SIZE * M_SHARD * N,)  — flat symm mem buf
    scatter_offset,  # element offset for this rank's slot: RANK * M_SHARD * N
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    # Dimensions
    M, N, K,
    # Tile sizes (compile-time constants)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Computes one (BLOCK_M, BLOCK_N) output tile via a pipelined inner k-loop,
    then writes the result to a (potentially peer GPU's) symmetric memory buffer.

    The software pipeline pass operates on the tl.range loop below:
      - tt.load(a) and tt.load(b) become ttg.async_copy_global_to_local
      - commit/wait groups are inserted to overlap loads with tt.dot

    The tl.store at the end (scatter step) is NOT touched by the pipeliner:
      - 'acc' is only valid after all K iterations complete
      - There is no async scatter op in Triton's IR
      - The pipeliner assigns tt.latency only to loads and MMA ops, not stores
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ── Phase 1: GEMM inner loop ────────────────────────────────────────────
    # num_stages=NUM_STAGES instructs the pipeliner to hide load latency by
    # issuing loads NUM_STAGES iterations ahead of the dot that consumes them.
    #
    # IR before pipelining:
    #   scf.for %k = 0 to K//BLOCK_K step 1 {tt.num_stages = 3} {
    #     %a = tt.load %a_ptrs : ...
    #     %b = tt.load %b_ptrs : ...
    #     %c = tt.dot %a, %b, %acc : ...
    #     ...
    #   }
    #
    # IR after pipelining:
    #   [prologue: 2 async copies issued]
    #   scf.for %k = 0 to K//BLOCK_K step 1 {
    #     ttg.async_copy_global_to_local %a_ptrs -> %smem_a[next_buf]
    #     ttg.async_copy_global_to_local %b_ptrs -> %smem_b[next_buf]
    #     ttng.async_commit_group
    #     ttng.async_wait {num = NUM_STAGES-1}
    #     %a = ttg.local_load %smem_a[cur_buf]
    #     %b = ttg.local_load %smem_b[cur_buf]
    #     %c = tt.dot %a, %b, %acc : ...
    #     ...
    #   }
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    result = acc.to(tl.float16)

    # ── Phase 2: Scatter store ──────────────────────────────────────────────
    # This store is OUTSIDE the inner loop and CANNOT be pipelined because:
    #   (a) 'acc' is only valid after all k iterations — it is the loop-carried
    #       value that accumulates across iterations; no earlier partial value
    #       is correct to scatter.
    #   (b) There is no async scatter op in Triton's IR. The pipeliner only
    #       assigns tt.latency to tt.LoadOp / tt.DescriptorLoadOp and to
    #       tt.DotOp / ttng.MMAv5OpInterface (see AssignLatencies.cpp).
    #   (c) A new pass would need to model cross-GPU communication latency
    #       and insert an async "commit" before the K-loop and a "wait" before
    #       the next tile — analogous to how loads are handled today.
    #
    # In multi-GPU deployment:
    #   out_ptr is the destination rank's symmetric memory buffer (peer GPU
    #   memory mapped via NVLink/UVA).  scatter_offset encodes this rank's
    #   contribution slot: RANK * M_SHARD * N elements from the buffer base.
    out_ptrs = (out_ptr + scatter_offset
                + offs_m[:, None] * N
                + offs_n[None, :])
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, result, mask=out_mask)


# ===========================================================================
# Kernel 2: Local Reduce
# ===========================================================================
@triton.jit
def reduce_kernel(
    # Local symm mem buffer: (WORLD_SIZE, M_SHARD, N) laid out flat
    partial_buf_ptr,     # float16
    out_ptr,             # float16 (M_SHARD, N)
    M_SHARD, N,
    WORLD_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Reads WORLD_SIZE partial tiles from the local symm mem buffer and sums them.
    Each GPU calls this after the host-side barrier that ensures all peers have
    finished writing their contributions.

    The tl.static_range unrolls the WORLD_SIZE loop at compile time.
    The pipeliner sees no dynamic loop here, so there is no opportunity to
    pipeline across the WORLD_SIZE reduction.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # tl.static_range: compile-time unrolled loop over WORLD_SIZE partials
    for r in tl.static_range(WORLD_SIZE):
        elem_off = r * M_SHARD * N + offs_m[:, None] * N + offs_n[None, :]
        mask = (offs_m[:, None] < M_SHARD) & (offs_n[None, :] < N)
        partial = tl.load(partial_buf_ptr + elem_off, mask=mask, other=0.0)
        acc += partial.to(tl.float32)

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M_SHARD) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float16), mask=out_mask)


# ===========================================================================
# Single-GPU simulation (world_size=1) for IR capture
# ===========================================================================
def run_warmup(dev: torch.device, dump_dir: str | None = None):
    """
    Compiles both kernels without executing them on the GPU.
    Sets environment variables to capture MLIR IR dumps to files.

    Triton compiles the kernel on the first warmup() call regardless of GPU
    execution — the full MLIR pipeline (assign-latencies → schedule-loops →
    pipeline) runs, and MLIR_ENABLE_DUMP captures every pass boundary.
    """
    world_size = 1
    rank = 0
    K_local = K
    M_shard = M

    a_local = torch.zeros((M, K_local), dtype=torch.float16, device=dev)
    b = torch.zeros((K_local, N), dtype=torch.float16, device=dev)

    # For world_size=1, the "peer" buffer is just a local tensor.
    # scatter_offset = 0 (rank 0's slot at the start of the buffer)
    symm_buf = torch.zeros((world_size * M_shard * N,), dtype=torch.float16, device=dev)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    print("[warmup] Compiling gemm_scatter_kernel ...")
    gemm_scatter_kernel.warmup(
        a_local, b,
        symm_buf,
        0,  # scatter_offset = RANK * M_SHARD * N = 0
        a_local.stride(0), a_local.stride(1),
        b.stride(0), b.stride(1),
        M, N, K_local,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_STAGES=NUM_STAGES,
        grid=grid,
    )
    print("[warmup] gemm_scatter_kernel compiled.")

    out = torch.zeros((M_shard, N), dtype=torch.float16, device=dev)
    grid2 = (triton.cdiv(M_shard, BLOCK_M), triton.cdiv(N, BLOCK_N))

    print("[warmup] Compiling reduce_kernel ...")
    reduce_kernel.warmup(
        symm_buf, out,
        M_shard, N,
        WORLD_SIZE=world_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        grid=grid2,
    )
    print("[warmup] reduce_kernel compiled.")


def run_single_gpu(dev: torch.device):
    """
    Single-GPU functional test (world_size=1).
    Results should match torch.matmul(A, B) to within fp16 tolerance.
    """
    world_size = 1
    rank = 0
    K_local = K
    M_shard = M

    a_local = torch.randn((M, K_local), dtype=torch.float16, device=dev)
    b = torch.randn((K_local, N), dtype=torch.float16, device=dev)
    symm_buf = torch.zeros((world_size * M_shard * N,), dtype=torch.float16, device=dev)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gemm_scatter_kernel[grid](
        a_local, b,
        symm_buf,
        rank * M_shard * N,
        a_local.stride(0), a_local.stride(1),
        b.stride(0), b.stride(1),
        M, N, K_local,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_STAGES=NUM_STAGES,
    )

    out = torch.zeros((M_shard, N), dtype=torch.float16, device=dev)
    grid2 = (triton.cdiv(M_shard, BLOCK_M), triton.cdiv(N, BLOCK_N))
    reduce_kernel[grid2](
        symm_buf, out,
        M_shard, N,
        WORLD_SIZE=world_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )

    ref = (a_local.float() @ b.float()).half()
    max_diff = (out - ref).abs().max().item()
    print(f"[single-gpu] max |triton - torch| = {max_diff:.4f}")
    assert max_diff < 1.0, f"Numerical error too large: {max_diff}"
    print("[single-gpu] PASSED")
    return out


def run_multi_gpu(world_size: int):
    """
    Multi-GPU Reduce-Scatter using torch.distributed symmetric memory.

    This function requires:
      - NCCL-capable GPU hardware (NVIDIA)
      - torch.distributed initialized
      - world_size GPUs visible

    The symmetric memory model:
      - torch.distributed._symmetric_memory allocates UVA-mapped buffers
        that are directly accessible from all GPUs via NVLink or PCIe
      - get_buffer(rank, ...) materializes a view of peer GPU's buffer
      - Triton kernel writes to peer's buffer via NVLink/UVA (tl.store)
      - Host-side hdl.barrier(channel=0) ensures all writes are visible
        before the reduce step
    """
    import torch.distributed as dist
    import torch.multiprocessing as mp

    def worker(rank: int, world_size: int):
        import torch.distributed._symmetric_memory as symm_mem

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dev = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(dev)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        K_local = K // world_size
        M_shard = M // world_size

        a_local = torch.randn((M, K_local), dtype=torch.float16, device=dev)
        b = torch.randn((K_local, N), dtype=torch.float16, device=dev)

        # Allocate symmetric memory buffer: each GPU holds partials from all ranks
        # Layout: (WORLD_SIZE, M_SHARD, N) fp16
        buf_raw = symm_mem.empty(
            (world_size * M_shard * N,),
            dtype=torch.float16,
            device=dev,
        )
        hdl = symm_mem.rendezvous(buf_raw, group=dist.group.WORLD)

        # Get peer buffer pointers
        # peer_bufs[r] is a tensor backed by GPU r's symm mem buffer
        peer_bufs = [
            hdl.get_buffer(r, (world_size * M_shard * N,), torch.float16)
            for r in range(world_size)
        ]

        # Barrier: ensure all GPUs have initialized their buffers
        hdl.barrier(channel=0)

        # Phase 1+2: GEMM + Scatter
        # For each destination rank d, launch the kernel writing to peer_bufs[d]
        # (In practice, a more efficient implementation would select the destination
        #  inside the kernel using tl.static_range, but the two-launch approach
        #  is clearer for illustration)
        for dst_rank in range(world_size):
            # Only CTAs whose row range belongs to dst_rank write in this pass
            # Here we simplify: each GPU writes all of its output to each dst_rank
            # A real reduce-scatter would partition by row range
            grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
            gemm_scatter_kernel[grid](
                a_local, b,
                peer_bufs[dst_rank],
                rank * M_shard * N,    # this rank's slot in the peer buffer
                a_local.stride(0), a_local.stride(1),
                b.stride(0), b.stride(1),
                M, N, K_local,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                NUM_STAGES=NUM_STAGES,
            )

        # Host-side barrier: wait for all GPUs to finish writing
        # This is NOT a GPU-side barrier; it's torch.distributed + CUDA stream sync
        # The pipeliner cannot model across this boundary
        hdl.barrier(channel=1)

        # Phase 4: Local reduce (sum WORLD_SIZE partials in local buffer)
        out = torch.zeros((M_shard, N), dtype=torch.float16, device=dev)
        grid2 = (triton.cdiv(M_shard, BLOCK_M), triton.cdiv(N, BLOCK_N))
        reduce_kernel[grid2](
            buf_raw, out,
            M_shard, N,
            WORLD_SIZE=world_size,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        dist.destroy_process_group()
        print(f"[rank {rank}] DONE, out shape = {out.shape}")

    mp.spawn(worker, args=(world_size,), nprocs=world_size, join=True)


# ===========================================================================
# IR capture utilities
# ===========================================================================
def setup_ir_capture(dump_file: str, always_compile: bool = True):
    """Set environment variables for MLIR pass dump."""
    os.environ["MLIR_ENABLE_DUMP"] = "1"
    os.environ["MLIR_DUMP_PATH"] = dump_file
    if always_compile:
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"


def teardown_ir_capture():
    """Remove environment variables after IR capture."""
    for k in ["MLIR_ENABLE_DUMP", "MLIR_DUMP_PATH", "TRITON_ALWAYS_COMPILE"]:
        os.environ.pop(k, None)


def extract_pipeline_ir(dump_file: str, before_file: str, after_file: str):
    """
    Parse the full MLIR dump file and extract the IR immediately before
    and immediately after the TritonGPUPipeline pass.
    """
    if not os.path.exists(dump_file):
        print(f"[extract] dump file not found: {dump_file}", file=sys.stderr)
        return False

    content = open(dump_file).read()
    marker = "// -----// IR Dump "
    chunks = content.split(marker)

    before_ir = None
    after_ir = None

    for i, chunk in enumerate(chunks):
        if chunk.startswith("Before TritonGPUPipeline"):
            before_ir = marker + chunk
        elif chunk.startswith("After TritonGPUPipeline"):
            after_ir = marker + chunk

    if before_ir:
        os.makedirs(os.path.dirname(before_file), exist_ok=True)
        open(before_file, "w").write(before_ir)
        print(f"[extract] Wrote before-pipeline IR to {before_file}")
    else:
        print("[extract] WARNING: 'Before TritonGPUPipeline' section not found")

    if after_ir:
        os.makedirs(os.path.dirname(after_file), exist_ok=True)
        open(after_file, "w").write(after_ir)
        print(f"[extract] Wrote after-pipeline IR to {after_file}")
    else:
        print("[extract] WARNING: 'After TritonGPUPipeline' section not found")

    return bool(before_ir and after_ir)


def generate_diff(before_file: str, after_file: str, diff_file: str):
    """Generate unified diff between before and after pipeline IR."""
    import subprocess
    if not (os.path.exists(before_file) and os.path.exists(after_file)):
        print("[diff] Missing IR files, skipping diff generation")
        return
    result = subprocess.run(
        ["diff", "-u", before_file, after_file],
        capture_output=True, text=True
    )
    open(diff_file, "w").write(result.stdout)
    print(f"[diff] Wrote diff ({len(result.stdout)} chars) to {diff_file}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="GEMM + Reduce-Scatter overlap analysis in Triton"
    )
    parser.add_argument(
        "--mode",
        choices=["warmup", "single", "multi"],
        default="warmup",
        help=(
            "warmup: compile kernels and capture IR only (no GPU execution)\n"
            "single: single-GPU functional test\n"
            "multi:  multi-GPU reduce-scatter (requires NCCL and multiple GPUs)"
        ),
    )
    parser.add_argument("--world-size", type=int, default=2,
                        help="Number of GPUs for multi mode")
    parser.add_argument("--no-ir-capture", action="store_true",
                        help="Skip MLIR IR dump even in warmup mode")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device available. Cannot compile Triton kernels.")
        print("Triton requires a CUDA device for JIT compilation.")
        sys.exit(1)

    dev = torch.device("cuda:0")

    if args.mode == "warmup":
        dump_file = "artifacts/pass_logs/full_mlir_dump.mlir"
        os.makedirs("artifacts/pass_logs", exist_ok=True)

        if not args.no_ir_capture:
            setup_ir_capture(dump_file)
            print(f"[main] IR dump enabled → {dump_file}")

        run_warmup(dev)

        if not args.no_ir_capture:
            teardown_ir_capture()
            extract_pipeline_ir(
                dump_file,
                "artifacts/ir_before/before_pipeline.mlir",
                "artifacts/ir_after/after_pipeline.mlir",
            )
            generate_diff(
                "artifacts/ir_before/before_pipeline.mlir",
                "artifacts/ir_after/after_pipeline.mlir",
                "diff_ir.txt",
            )

    elif args.mode == "single":
        run_single_gpu(dev)

    elif args.mode == "multi":
        run_multi_gpu(args.world_size)


if __name__ == "__main__":
    main()
