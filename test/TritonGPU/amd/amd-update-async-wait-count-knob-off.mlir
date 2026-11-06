// With staged async marks disabled, gfx1250 keeps the previous behavior:
// UpdateAsyncWaitCount counts outstanding async instructions and rewrites
// ttg.async_wait to amdg.async_wait.
// RUN: env TRITON_HIP_USE_STAGED_ASYNC_MARKS=0 triton-opt %s -split-input-file --tritonamdgpu-update-async-wait-count=gfx-arch=gfx1250 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [1, 0]}>
#shared1 = #ttg.padded_shared<[4:+4] {order = [1, 0], shape=[16, 256]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: simple_waitcnt_knob_off
  tt.func public @simple_waitcnt_knob_off(%arg1: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<16x256xf16, #shared1, #smem, mutable>, %arg3: tensor<128x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}, %arg4: tensor<16x256x!tt.ptr<f16>, #blocked1> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    %0 = ttg.async_copy_global_to_local %arg3, %arg1 : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    %1 = ttg.async_commit_group tokens %0
    %2 = ttg.async_copy_global_to_local %arg4, %arg2 : tensor<16x256x!tt.ptr<f16>, #blocked1> -> <16x256xf16, #shared1, #smem, mutable>
    %3 = ttg.async_commit_group tokens %2

    // CHECK: amdg.async_wait {{.*}} {num_inst = 4
    %9 = ttg.async_wait %1 {num = 0 : i32}
    // CHECK: amdg.async_wait {{.*}} {num_inst = 0
    %10 = ttg.async_wait %3 {num = 0 : i32}
    tt.return
  }
}
