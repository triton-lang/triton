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

// -----

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: make_tensor_descriptor
  // CHECK: %0 = arith.extsi %arg2 : i32 to i64
  // CHECK: %1 = triton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : <i8>
  // CHECK: %2 = arith.shrsi %0, %c4_i64 : i64
  // CHECK: tt.experimental_tensormap_create %1, %arg0, [%c32_i32, %c8_i32], [%arg2, %arg1], [%2], [%c1_i32, %c1_i32] {elem_type = 0 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 1 : i32} : (!tt.ptr<i8>, !tt.ptr<i8>, i32, i32, i32, i32, i64, i32, i32) -> ()
  // CHECK: tt.experimental_tensormap_fenceproxy_acquire %1 : !tt.ptr<i8>
  tt.func public @make_tensor_descriptor(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32} , %arg1: i32 {tt.divisibility = 16 : i32} , %arg2: i32 {tt.divisibility = 16 : i32} ) -> !tt.ptr<i8> {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<32> : tensor<8x1xi32>
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.extsi %arg2 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] {tensorShape = array<i32: 8, 32>} : <i8>, <i8>
    tt.return %1 : !tt.ptr<i8>
  }
}
