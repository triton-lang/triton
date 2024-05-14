// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm | FileCheck %s

#shared0 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: init_barrier
  tt.func @init_barrier(%alloc: !tt.memdesc<1xi64, #shared0>) {
    // CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void
    triton_nvidia_gpu.init_barrier %alloc, 1 : !tt.memdesc<1xi64, #shared0>
    tt.return
  }
}

// -----

#shared0 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: wait_barrier
  tt.func @wait_barrier(%alloc: !tt.memdesc<1xi64, #shared0>, %phase: i32) {
    // CHECK: waitLoop:
    // CHECK: mbarrier.try_wait.parity.shared.b64
    // CHECK: @!P1 bra.uni waitLoop
    triton_nvidia_gpu.wait_barrier %alloc, %phase : !tt.memdesc<1xi64, #shared0>
    tt.return
  }
}


// -----

#shared0 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: tma_copy_global_to_local
  // CHECK: elect.sync
  // CHECK: "@$0 mbarrier.arrive.expect_tx.shared.b64 _, [$1], 65536;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void
  // CHECK: nvvm.barrier0
  // CHECK: "@$0 cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2, {$3, $4}], [$5];", "b,r,l,r,r,r" {{.*}} : (i1, !llvm.ptr<3>, !llvm.ptr<1>, i32, i32, !llvm.ptr<3>) -> !llvm.void
  tt.func @tma_copy_global_to_local(%tma: !tt.ptr<i64>, %alloc: !tt.memdesc<128x128xf32, #shared1>, %x: i32, %barrier: !tt.memdesc<1xi64, #shared0>, %pred: i1) {
    triton_nvidia_gpu.async_tma_copy_global_to_local %tma[%x, %x] %alloc, %barrier, %pred : !tt.ptr<i64>, !tt.memdesc<1xi64, #shared0> -> !tt.memdesc<128x128xf32, #shared1>
    tt.return
  }
}

// -----

#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: tma_copy_local_to_global
  // CHECK: elect.sync
  // CHECK: "@$0 cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$1, {$2, $3}], [$4];", "b,l,r,r,r" {{.*}} : (i1, !llvm.ptr<1>, i32, i32, !llvm.ptr<3>) -> !llvm.void
  // CHECK: cp.async.bulk.commit_group
  tt.func @tma_copy_local_to_global(%tma: !tt.ptr<i64>, %alloc: !tt.memdesc<128x128xf32, #shared1>, %x: i32) {
    triton_nvidia_gpu.async_tma_copy_local_to_global %tma[%x, %x] %alloc : <i64>, <128x128xf32, #shared1>
    tt.return
  }
}

// -----

#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: async_tma_store_wait
  // CHECK: "cp.async.bulk.wait_group.read 0x0;", ""  : () -> !llvm.void
  tt.func @async_tma_store_wait() {
    triton_nvidia_gpu.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}
