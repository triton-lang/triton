// RUN: triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=107 ptx-version=94' -reconcile-unrealized-casts | FileCheck %s
// RUN: triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=93' -reconcile-unrealized-casts | FileCheck %s
// RUN: not triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=86' -reconcile-unrealized-casts 2>&1 | FileCheck --check-prefix=ERROR %s
// RUN: not triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=80 ptx-version=93' -reconcile-unrealized-casts 2>&1 | FileCheck --check-prefix=ERROR %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: barrier_test_wait_report
  tt.func @barrier_test_wait_report(%alloc: !ttg.memdesc<1xi64, #shared, #smem>, %phase: i32, %pred: i1) {
    // CHECK: mov.u32 $0, 0;
    // CHECK: mov.u32 $1, 0;
    // CHECK: @!$4 bra.uni skipTest;
    // CHECK: mbarrier.test_wait.parity.phase_type::primary.shared::cta.b64 complete|reported, [$2], $3;
    // CHECK: selp.u32 $0, 1, 0, complete;
    // CHECK: selp.u32 $1, 1, 0, reported;
    %done, %reported = ttng.barrier_test_wait_report %alloc, %phase, %pred : !ttg.memdesc<1xi64, #shared, #smem> -> (i32, i32)

    // CHECK: mbarrier.test_wait.parity.phase_type::conditional.shared::cta.b64 complete, [$1], $2;
    %conditional = ttng.barrier_test_wait %alloc, %phase, %pred, conditional : !ttg.memdesc<1xi64, #shared, #smem> -> i32
    tt.return
  }
}

// ERROR: primary mbarrier report requires mbarrier v1 layout support
