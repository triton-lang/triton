// RUN: triton-opt %s --convert-triton-gpu-to-llvm=compute-capability=80 -reconcile-unrealized-casts | FileCheck %s

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: wait_barrier_sm80
  tt.func @wait_barrier_sm80(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>, %phase: i32, %pred: i1) {
    // CHECK: waitLoop:
    // CHECK: mbarrier.test_wait.parity.shared::cta.b64
    // CHECK-NOT: @!complete bra.uni waitLoop
    // CHECK: @!complete bra waitLoop
    ttng.wait_barrier %alloc, %phase : !ttg.memdesc<1xi64, #shared0, #smem>

    // CHECK: @!$2 bra.uni skipWait
    // CHECK: waitLoop:
    // CHECK: mbarrier.test_wait.parity.shared::cta.b64
    // CHECK-NOT: @!complete bra.uni waitLoop
    // CHECK: @!complete bra waitLoop
    // CHECK: skipWait:
    ttng.wait_barrier %alloc, %phase, %pred : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }
}
