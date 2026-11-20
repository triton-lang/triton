// RUN: triton-opt %s --triton-nvidia-gpu-membar='compute-capability=107 ptx-version=94' --allocate-shared-memory-nv=compute-capability=107 --convert-triton-gpu-to-llvm='compute-capability=107 ptx-version=94' --convert-warp-specialize-to-llvm -reconcile-unrealized-casts | FileCheck %s
// RUN: triton-opt %s --triton-nvidia-gpu-membar='compute-capability=90 ptx-version=93' --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=93' --convert-warp-specialize-to-llvm -reconcile-unrealized-casts | FileCheck %s
// RUN: not triton-opt %s --triton-nvidia-gpu-membar='compute-capability=90 ptx-version=86' --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=86' --convert-warp-specialize-to-llvm -reconcile-unrealized-casts 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not triton-opt %s --triton-nvidia-gpu-membar='compute-capability=80 ptx-version=93' --allocate-shared-memory-nv=compute-capability=80 --convert-triton-gpu-to-llvm='compute-capability=80 ptx-version=93' --convert-warp-specialize-to-llvm -reconcile-unrealized-casts 2>&1 | FileCheck --check-prefix=ERROR %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: barrier_test_wait_report
  tt.func @barrier_test_wait_report(%out: !tt.ptr<i1>, %phase: i32, %pred: i1) {
    // Keep an explicit barrier allocation; query results need no scratch.
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %alloc, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: mov.pred $0, 0;
    // CHECK: mov.pred $1, 0;
    // CHECK: @$4 mbarrier.test_wait.parity.phase_type::primary.shared::cta.b64 $0|$1, [$2], $3;
    // CHECK: "=b,=b,r,r,b"
    %done, %reported = ttng.barrier_test_wait_report %alloc, %phase, %pred : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> (i1, i1)
    // CHECK: barrier.cta.red.and.pred $0, $1, 128, $2;
    // CHECK: llvm.and
    // CHECK-NOT: st.shared
    // CHECK-NOT: llvm.load

    // CHECK: @$3 mbarrier.test_wait.parity.phase_type::conditional.shared::cta.b64 $0, [$1], $2;
    %conditional = ttng.barrier_test_wait %alloc, %phase, %pred, conditional : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> i1
    // CHECK: barrier.cta.red.and.pred $0, $1, 128, $2;
    // CHECK-NOT: st.shared
    // CHECK-NOT: llvm.load
    // Keep both completion and report results live through the reductions.
    %combined = arith.andi %done, %reported : i1
    %result = arith.andi %combined, %conditional : i1
    tt.store %out, %result : !tt.ptr<i1>
    tt.return
  }
}

// ERROR: primary mbarrier report requires mbarrier v1 layout support
