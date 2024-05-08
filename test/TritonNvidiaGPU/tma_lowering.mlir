// RUN: triton-opt %s -split-input-file --triton-nvidia-tma-lowering | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tma_load
// CHECK: triton_gpu.local_alloc  : ()
// CHECK: triton_gpu.local_alloc  : ()
// CHECK: triton_nvidia_gpu.init_barrier
// CHECK: triton_nvidia_gpu.async_tma_copy_global_to_local
// CHECK: triton_nvidia_gpu.wait_barrier
// CHECK: triton_nvidia_gpu.inval_barrier
// CHECK: triton_gpu.local_load
  tt.func public @tma_load(%arg0: !tt.ptr<i8>, %arg1: i32) -> tensor<128x64xf16, #blocked> {
    %l = tt.experimental_descriptor_load %arg0[%arg1, %arg1] : !tt.ptr<i8> -> tensor<128x64xf16, #blocked>
    tt.return %l : tensor<128x64xf16, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: tma_store
//       CHECK: triton_gpu.local_alloc
//       CHECK: triton_nvidia_gpu.fence_async_shared {bCluster = false}
//       CHECK: triton_nvidia_gpu.async_tma_copy_local_to_global
  tt.func public @tma_store(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: tensor<128x256xf32, #blocked>) {
    tt.experimental_descriptor_store %arg0[%arg1, %arg1], %arg2 : !tt.ptr<i8>, tensor<128x256xf32, #blocked>
    tt.return
  }
}
