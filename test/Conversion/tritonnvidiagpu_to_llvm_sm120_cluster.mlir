// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=120' --convert-triton-gpu-to-llvm='compute-capability=120' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[0]]}>
#shared_barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#shared_store = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:120", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sm120_async_shared_store
  // CHECK: fence.mbarrier_init.release.cluster
  // CHECK: st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32
  tt.func public @sm120_async_shared_store(
      %src: tensor<128xi32, #blocked>,
      %dst: !ttg.memdesc<128xi32, #shared_store, #smem, mutable>,
      %mbarrier: !ttg.memdesc<2xi64, #shared_barrier, #smem, mutable>) {
    ttng.fence_mbarrier_init_release_cluster
    ttng.async_shared_store %src, %dst, %mbarrier : tensor<128xi32, #blocked> -> !ttg.memdesc<128xi32, #shared_store, #smem, mutable>, !ttg.memdesc<2xi64, #shared_barrier, #smem, mutable>
    tt.return
  }
}
