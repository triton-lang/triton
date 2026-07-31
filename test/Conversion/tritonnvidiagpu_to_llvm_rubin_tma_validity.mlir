// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=107 --convert-triton-gpu-to-llvm='compute-capability=107 ptx-version=94' -reconcile-unrealized-casts | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=93' -reconcile-unrealized-casts | FileCheck %s
// RUN: not triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=86' -reconcile-unrealized-casts 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=80 --convert-triton-gpu-to-llvm='compute-capability=80 ptx-version=93' -reconcile-unrealized-casts 2>&1 | FileCheck --check-prefix=ERROR %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: barrier_test_wait_report
  tt.func @barrier_test_wait_report(%alloc: !ttg.memdesc<1xi64, #shared, #smem>, %phase: i32, %pred: i1) {
    // CHECK: mov.pred $0, 0;
    // CHECK: mov.pred $1, 0;
    // CHECK: @!$4 bra.uni skipTest;
    // CHECK: mbarrier.test_wait.parity.phase_type::primary.shared::cta.b64 $0|$1, [$2], $3;
    // CHECK: "=b,=b,r,r,b"
    %done, %reported = ttng.barrier_test_wait_report %alloc, %phase, %pred : !ttg.memdesc<1xi64, #shared, #smem> -> (i1, i1)
    // CHECK: st.shared::cta.b8
    // CHECK: nvvm.barrier
    // CHECK: llvm.load
    // CHECK: nvvm.barrier

    // CHECK: mbarrier.test_wait.parity.phase_type::conditional.shared::cta.b64 $0, [$1], $2;
    %conditional = ttng.barrier_test_wait %alloc, %phase, %pred, conditional : !ttg.memdesc<1xi64, #shared, #smem> -> i1
    // CHECK: st.shared::cta.b8
    // CHECK: nvvm.barrier
    // CHECK: llvm.load
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: barrier_test_wait_report_cluster
  tt.func @barrier_test_wait_report_cluster(%alloc: !ttg.memdesc<1xi64, #shared, #smem>, %phase: i32) {
    // CHECK: nvg.cluster_id
    // CHECK: mbarrier.test_wait.parity.phase_type::primary.shared::cta.b64
    %done, %reported = ttng.barrier_test_wait_report %alloc, %phase : !ttg.memdesc<1xi64, #shared, #smem> -> (i1, i1)
    // CHECK: st.shared::cta.b8
    // CHECK: nvvm.cluster.arrive
    // CHECK: nvvm.cluster.wait
    // CHECK: nvvm.mapa
    // CHECK: llvm.load {{.*}} : !llvm.ptr<7> -> i8
    tt.return
  }
}

// ERROR: primary mbarrier report requires mbarrier v1 layout support
