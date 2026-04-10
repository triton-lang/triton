# GEMM + Reduce-Scatter Overlap: Analysis Report

## Executive Summary

Triton의 software pipeline pass는 **GEMM의 inner K-loop 내부 load/dot 연산의 overlap에는 성공**하지만, **GEMM과 Reduce-Scatter 간의 overlap은 불가능**하다. 이는 (1) scatter store가 완전히 누적된 accumulator에 의존하여 loop 외부에서 실행되고, (2) Triton IR에 비동기 scatter 연산이 없으며, (3) host-side barrier가 컴파일러가 볼 수 없는 epoch boundary를 만들기 때문이다.

실제 `triton-opt` 실행 결과, pipeline pass 이후 `tt.store` (scatter)는 **완전히 변경되지 않음**을 확인했다.

---

## 1. System Overview

### 1.1 Triton Software Pipeline Infrastructure

Triton의 software pipeline은 세 개의 순차적 MLIR pass로 구현된다:

1. **`TritonGPUAssignLatencies`** (`lib/Dialect/TritonGPU/Transforms/Pipeliner/AssignLatencies.cpp`)
   - `tt.LoadOp`, `tt.DescriptorLoadOp`, `tt.DotOp` 등에 `tt.latency` attribute를 할당
   - **`tt.StoreOp`에는 latency를 할당하지 않음** — store는 pipelining 대상이 아님

2. **`TritonGPUScheduleLoops`** (`lib/Dialect/TritonGPU/Transforms/Pipeliner/ScheduleLoops.cpp`)
   - 각 op에 `(loop.stage, loop.cluster)` 쌍을 할당하여 coarse schedule 생성
   - 전제조건: innermost loop, loop-carried distance ≤ 1, no barriers/asserts/prints

3. **`TritonGPUPipeline`** (`lib/Dialect/TritonGPU/Transforms/Pipeliner/SoftwarePipeliner.cpp`)
   - `tt.load` → `ttg.async_copy_global_to_local` 변환
   - 다중 버퍼 shared memory 할당 (`ttg.local_alloc`)
   - Prologue/epilogue 생성 및 `ttg.async_wait`/`ttg.async_commit_group` 삽입

### 1.2 Symmetric Memory Model

`torch.distributed._symmetric_memory`를 통해 UVA(Unified Virtual Addressing) 기반의 peer GPU 메모리 접근을 제공한다.

- `symm_mem.empty()`: 모든 참여 GPU에서 동일한 버퍼 할당
- `symm_mem.rendezvous()`: 메모리 매핑 설정 (NVLink/PCIe를 통한 peer 접근)
- `hdl.get_buffer(rank, ...)`: peer GPU의 버퍼를 PyTorch tensor로 반환
- `hdl.barrier(channel)`: host-side 동기화 (CPU barrier + CUDA stream sync)

Triton 커널에서 peer buffer pointer를 일반 `tl.store` 인자로 전달하면, NVLink/UVA를 통해 peer GPU 메모리에 직접 기록한다. **Triton IR에는 분산 collective 연산(`reduce_scatter`, `all_reduce` 등)이 존재하지 않는다.**

### 1.3 Reduce-Scatter Semantics

Row-parallel linear에서의 reduce-scatter:

```
GPU r:  A_r (M, K_local) @ B (K_local, N) = P_r (M, N)   ← partial sum
Reduce-Scatter:  GPU r gets rows [r*M_SHARD : (r+1)*M_SHARD] of Σ_r P_r
```

구현 방식 (two-kernel + host barrier):
```
Kernel 1 (gemm_scatter): GEMM + tl.store to peer buffer
   ↓
hdl.barrier(channel=0)   ← host-side epoch boundary
   ↓
Kernel 2 (reduce):       Σ over WORLD_SIZE partials in local buffer
```

---

## 2. Kernel Implementation

### 2.1 Design Choices

| 결정 사항 | 선택 | 이유 |
|-----------|------|------|
| 커널 수 | 2개 (gemm_scatter + reduce) | GPU-side 크로스-GPU 세마포어 없음 |
| Pipeline depth | NUM_STAGES=3 | 2 in-flight loads + 1 computing |
| Tile sizes | 128×128 output, 64(32) K reduction | 표준 fp16 GEMM 타일 |
| Peer pointer 전달 | out_ptr (typed pointer) | 단순성; multi-GPU에서 host가 pre-compute |
| 동기화 | hdl.barrier(channel=0) | host-side; 컴파일러 투명 |

### 2.2 Kernel Code Structure

**`gemm_scatter_kernel`** (`gemm_reduce_scatter_triton.py`):
```python
@triton.jit
def gemm_scatter_kernel(a_ptr, b_ptr, out_ptr, scatter_offset, ...):
    # Phase 1: GEMM inner loop — PIPELINED by software pipeline pass
    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl.load(a_ptrs, ...)   # → ttg.async_copy_global_to_local
        b = tl.load(b_ptrs, ...)   # → ttg.async_copy_global_to_local
        acc = tl.dot(a, b, acc)    # → tt.dot (unchanged, reads from shared mem)
    
    # Phase 2: Scatter store — NOT PIPELINED (depends on acc_final)
    tl.store(out_ptrs, acc.to(tl.float16))  # → tt.store (UNCHANGED)
```

**`reduce_kernel`** (`gemm_reduce_scatter_triton.py`):
```python
@triton.jit
def reduce_kernel(partial_buf_ptr, out_ptr, ...):
    for r in tl.static_range(WORLD_SIZE):  # compile-time unrolled
        partial = tl.load(partial_buf_ptr + offset_r)
        acc += partial
    tl.store(out_ptr, acc)
```

### 2.3 IR Capture Strategy

CUDA GPU가 없는 환경에서 `triton-opt`를 직접 사용하여 IR을 생성:
```bash
triton-opt input.mlir \
  -tritongpu-assign-latencies \
  -tritongpu-schedule-loops \
  -tritongpu-pipeline=num-stages=3 \
  -canonicalize \
  -o after_pipeline.mlir
```

Input MLIR은 `test/TritonGPU/loop-pipeline.mlir`의 matmul pattern을 기반으로, `tt.store`를 loop 외부에 추가하여 scatter를 모델링했다.

---

## 3. IR Analysis

### 3.1 IR Before Pipelining (TTGIR, after ScheduleLoops)

`artifacts/ir_before/before_pipeline.mlir` (37 lines):

```mlir
// Inner loop with schedule attributes:
%10:3 = scf.for %arg6 = %arg0 to %arg1 step %arg2
    iter_args(%arg7, %arg8, %arg9) -> (...) {
  %13 = tt.load %arg7 : tensor<128x32x!tt.ptr<f16>>        // A tile load
  %14 = ttg.convert_layout %13 {loop.stage = 2, loop.cluster = 0}
  %15 = tt.load %arg8 : tensor<32x128x!tt.ptr<f16>>        // B tile load
  %16 = ttg.convert_layout %15 {loop.stage = 2, loop.cluster = 0}
  %17 = arith.mulf %16, %cst {loop.stage = 2, loop.cluster = 0}
  %18 = tt.dot %14, %17, %arg9 {loop.stage = 2, loop.cluster = 0}  // MMA
  %19 = tt.addptr %arg7, ... {loop.stage = 1, loop.cluster = 2}    // ptr advance
  %20 = tt.addptr %arg8, ... {loop.stage = 1, loop.cluster = 2}
  scf.yield ...
} {tt.scheduled_max_stage = 2}

// POST-LOOP: scatter store (no schedule attributes)
%11 = ttg.convert_layout %10#2 : ... #mma -> ... #blocked1
%12 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
tt.store %12, %11 : tensor<128x128x!tt.ptr<f32>>          // ← SCATTER
```

### 3.2 IR After Pipelining (TTGIR, after TritonGPUPipeline)

`artifacts/ir_after/after_pipeline.mlir` (92 lines):

**새로 추가된 요소:**
```mlir
// Shared memory allocations (double-buffered)
%10 = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
%11 = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable>

// Prologue: 2 iterations of async copies issued ahead
%15 = ttg.async_copy_global_to_local %4, %13 mask %14 other %cst_4 ...
%16 = ttg.async_commit_group tokens %15
...

// Main loop: consume from shared mem, issue new async copies
%33:9 = scf.for ... iter_args(..., %arg10, %arg11, %arg12, %arg13, %arg14, %arg15) {
  %43 = ttg.async_wait %arg12, %arg14 {num = 2}    // wait for oldest
  %45 = ttg.local_load %44 token %43 : ... -> tensor<128x32xf16, #dot_op>
  %47 = ttg.local_load %46 token %43 : ... -> tensor<32x128xf16, #dot_op>
  %49 = tt.dot %45, %48, %arg9 : ...               // MMA (unchanged)
  %57 = ttg.async_copy_global_to_local %50, %55 ... // issue next async copy
  %58 = ttg.async_commit_group tokens %57
  ...
}

// Epilogue: drain all pending copies
%34 = ttg.async_wait {num = 0}
ttg.local_dealloc %11
ttg.local_dealloc %10
```

**변경되지 않은 부분 (scatter store):**
```mlir
// POST-LOOP: scatter store — IDENTICAL to before-pipeline version
%35 = ttg.convert_layout %33#2 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked1>
%36 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked1>
tt.store %36, %35 : tensor<128x128x!tt.ptr<f32>, #blocked1>    // ← UNCHANGED
```

### 3.3 Key IR Diff Observations

| 변경 내역 | Before → After |
|-----------|----------------|
| `tt.load` (global mem) | 2개 → 0개 (제거됨) |
| `ttg.async_copy_global_to_local` | 0개 → 6개 (비동기 복사) |
| `ttg.async_commit_group` | 0개 → 4개 (복사 그룹 커밋) |
| `ttg.async_wait` | 0개 → 2개 (대기) |
| `ttg.local_alloc` | 0개 → 2개 (공유 메모리 더블 버퍼) |
| `ttg.local_load` | 0개 → 2개 (공유 메모리 → 레지스터) |
| `tt.dot` | 1개 → 1개 (변경 없음) |
| **`tt.store` (scatter)** | **1개 → 1개 (완전히 동일)** |
| `scf.for` iter_args | 3개 → 9개 (+6: 버퍼 인덱스, 토큰) |

---

## 4. What the Pipeliner Does and Does Not Do

### 4.1 성공: GEMM A/B Tile Load 파이프라이닝

Pipeliner는 inner K-loop의 `tt.load` 연산을 성공적으로 변환한다:

1. `tt.load` → `ttg.async_copy_global_to_local` (cp.async 명령어 사용)
2. 다중 버퍼 공유 메모리 할당 (2x 버퍼, `ttg.local_alloc`)
3. Prologue에서 첫 2 iteration의 load를 미리 발행
4. Main loop에서 iteration k의 dot과 iteration k+2의 load를 중첩
5. Epilogue에서 남은 pending copy를 drain

이를 통해 global memory latency (~100-300 cycles)를 tensor core 연산으로 숨길 수 있다.

### 4.2 실패: Scatter Store 파이프라이닝

`tt.store` (scatter write)는 **전혀 변경되지 않았다.** 이유:

1. **위치:** `tt.store`는 `scf.for` **외부**에 위치 — pipeliner는 loop body 내부의 op만 변환
2. **의존성:** `%33#2` (accumulator)는 `scf.for`의 loop-carried value이며, 모든 iteration이 완료된 후에만 유효
3. **AssignLatencies 범위:** `tt.StoreOp`에는 `tt.latency`를 할당하지 않음 (source: `AssignLatencies.cpp`)
4. **비동기 op 부재:** `ttg.AsyncStoreGlobalOp` 같은 비동기 store가 Triton IR에 존재하지 않음

### 4.3 구조적 이유: Accumulator Dependency Chain

```
%acc_0 = zeros(...)
%acc_1 = tt.dot(load_a_0, load_b_0, %acc_0)    ← iteration 0
%acc_2 = tt.dot(load_a_1, load_b_1, %acc_1)    ← iteration 1
...
%acc_final = tt.dot(load_a_{K-1}, load_b_{K-1}, %acc_{K-2})
tt.store(peer_ptr, %acc_final)                  ← ALL iterations에 의존
```

`tt.store`는 `%acc_final`에 data dependency가 있고, 이는 **모든** `tt.dot`에 순차적으로 의존한다. 이 chain은 load latency hiding과 다르게, pipeline scheduling으로 해결할 수 없는 **근본적 ordering constraint**이다.

---

## 5. Failure Analysis: 왜 True GEMM-Comm Overlap이 불가능한가

### 5.1 Dependency Analysis

GEMM의 K-축 reduction은 **순차적 accumulation**이다. 중간 partial sum을 scatter하면 receiver는 incorrect value를 받게 된다 (다른 GPU의 partial sum과 합산 시 의미 없음). True overlap을 위해서는 알고리즘 변경이 필요하다.

### 5.2 Host-Barrier Epoch Boundary

`hdl.barrier(channel=0)`는 host-side operation이다:
1. `torch.distributed.barrier()` — 모든 CPU 프로세스 동기화
2. `torch.cuda.synchronize()` — 모든 CUDA 스트림 drain

이 barrier는 **컴파일러에 투명하지 않다.** 두 커널은 독립적으로 컴파일되며, inter-kernel scheduling은 Triton의 지원 범위 밖이다.

### 5.3 Triton IR에 Communication Ops 부재

| 필요한 기능 | Triton에서의 현재 상태 |
|------------|----------------------|
| Async reduce-scatter | 없음 (`tt.ReduceOp`는 intra-GPU) |
| Cross-GPU semaphore | 없음 (`mbarrier`는 intra-device) |
| `__threadfence_system()` | Triton에서 접근 불가 |
| Cooperative group 동기화 | 지원하지 않음 |

Peer buffer write (`tl.store to peer_ptr`)는 **일반 pointer dereference**로 IR에 표현된다. Pipeliner는 이것이 cross-GPU communication인지 알 수 없다.

---

## 6. Proposed Architecture for True Overlap

### 6.1 Algorithm Change: Streaming Atomic-Add Protocol

Full K reduction 후 scatter 대신, **K를 chunk 단위로 분할하여 partial sum을 streaming**:

```python
# 제안하는 알고리즘 (현재 Triton에서 직접 구현 불가):
for k_chunk in range(K // BLOCK_K // PIPELINE_CHUNKS):
    for kk in range(PIPELINE_CHUNKS):
        a = load(a_ptrs)
        b = load(b_ptrs)
        acc_partial += dot(a, b)
    
    # PIPELINE_CHUNKS tiles 후, partial sum을 peer에 atomic_add
    tl.atomic_add(peer_ptr + offset, acc_partial)
    signal_semaphore(peer_gpu, chunk_id=k_chunk)
    acc_partial = zeros(...)  # reset for next chunk
```

Receiver side:
```python
for k_chunk in range(total_chunks):
    wait_semaphore(src_gpu, chunk_id=k_chunk)  # spin-wait
    local_acc += tl.load(local_buf + chunk_offset)
```

이 접근법은 compute와 comm의 50% overlap을 달성할 수 있으나, `tl.atomic_add`의 overhead와 semaphore의 부재가 장벽이다.

### 6.2 New IR Ops Required

| Op 이름 | 설명 | 용도 |
|---------|------|------|
| `tt.AsyncReduceScatterOp` | 비동기 scatter + reduce | `tt.latency` 할당 가능 |
| `tt.SignalSemaphoreOp` | GPU-side inter-CTA signal | Cross-GPU spin-wait |
| `tt.WaitSemaphoreOp` | GPU-side spin-wait on signal | Receiver side |
| `tt.AsyncAtomicAddOp` | 비동기 atomic add to peer | Streaming partial store |

### 6.3 Extended Pipeline Passes

1. **Extended `AssignLatencies`:**
   - `tt.AsyncReduceScatterOp`에 NVLink latency 할당 (~1μs for NVLink4)
   - `tt.WaitSemaphoreOp`를 blocking op으로 인식

2. **Extended `ScheduleLoops`:**
   - Scatter ops와 compute ops를 pipeline stage에 interleave
   - Stage k: scatter for chunk k-2, compute chunk k-1, load chunk k

3. **New pass: `TritonGPUOverlapComputeComm`:**
   - Compute-communication overlap 전용 최적화
   - Full accumulation → partial accumulation 알고리즘 변환
   - 동기화 op 삽입 및 correctness 검증

### 6.4 기존 Warp-Specialized WGMMA Pipeline과의 비교

`WGMMAPipeline.cpp`는 WGMMA를 memory load와 overlap한다:
- WGMMA는 비동기 `wgmma_wait_group`으로 completion을 추적
- Load는 `ttg.async_copy_global_to_local`로 비동기 발행
- 두 비동기 연산이 동일 warp group 내에서 중첩

GEMM-RS overlap은 구조적으로 유사하지만 더 복잡:
- Completion signal이 **다른 GPU에서** 온다 (local wait counter가 아님)
- Latency가 network-dependent이고 compile-time에 정확히 알 수 없음
- 알고리즘 변경 (partial accumulation)에 대한 semantic correctness 분석 필요

---

## 7. Conclusions

### 확인된 결론

1. **Triton의 software pipeline pass는 GEMM inner loop의 load/compute overlap에 성공한다.**
   - `tt.load` → `ttg.async_copy_global_to_local` 변환
   - `NUM_STAGES=3`으로 2 iteration 선행 load 발행
   - Global memory latency를 tensor core 연산으로 효과적으로 숨김

2. **Scatter store (peer buffer write)는 pipeline pass에 의해 전혀 변경되지 않는다.**
   - 실제 `triton-opt` 출력에서 `tt.store`가 before/after 완전히 동일함을 확인
   - `diff_ir.txt`에서 scatter store 관련 변경 0건

3. **True GEMM-RS overlap은 현재 Triton으로 불가능하다.**
   - Accumulator dependency: full K reduction 없이는 correct value를 scatter할 수 없음
   - Host barrier: compiler-invisible epoch boundary
   - IR gaps: 비동기 scatter, cross-GPU semaphore, `__threadfence_system()` 미지원

4. **이를 해결하기 위해 필요한 것:**
   - **알고리즘 변경:** streaming partial scatter (atomic_add 기반)
   - **새로운 IR ops:** `tt.AsyncReduceScatterOp`, `tt.SignalSemaphoreOp`, `tt.WaitSemaphoreOp`
   - **확장된 pass:** `AssignLatencies`에 communication latency 모델 추가, `ScheduleLoops`에 compute-comm interleaving 추가
   - **새로운 pass:** `TritonGPUOverlapComputeComm` — compute-comm overlap 전용 최적화

### 현재 달성 가능한 Overlap

- **GEMM 내부:** load/compute overlap (pipeliner가 처리) ✅
- **Spatial concurrency:** 다수의 CTA가 동시 실행되어, 일부 CTA가 NVLink write를 수행하는 동안 다른 CTA는 GEMM을 계산 (implicit, pipeliner와 무관) ✅
- **Cross-kernel overlap:** 현재 Triton에서 불가능 ❌
- **Intra-kernel GEMM-RS overlap:** 현재 Triton에서 불가능 ❌
