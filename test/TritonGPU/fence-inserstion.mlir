// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-fence-insertion | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.target" = "cuda:90", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: matmul_like_fence
  tt.func public @matmul_like_fence(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x64xf16, #blocked2>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %0 = triton_gpu.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !triton_gpu.memdesc<128x128xf16, #shared>
    %1 = triton_gpu.local_alloc %arg1 : (tensor<128x64xf16, #blocked2>) -> !triton_gpu.memdesc<128x64xf16, #shared1>
    // CHECK: triton_nvidia_gpu.fence_async_shared
    %2 = triton_nvidia_gpu.warp_group_dot %0, %1, %cst : !triton_gpu.memdesc<128x128xf16, #shared> * !triton_gpu.memdesc<128x64xf16, #shared1> -> tensor<128x64xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.target" = "cuda:90", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fence_outside_loop
  tt.func public @fence_outside_loop(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x64xf16, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = triton_gpu.local_alloc %arg0 : (tensor<128x128xf16, #blocked>) -> !triton_gpu.memdesc<128x128xf16, #shared>
    %1 = triton_gpu.local_alloc %arg1 : (tensor<128x64xf16, #blocked>) -> !triton_gpu.memdesc<128x64xf16, #shared1>
    // CHECK: triton_nvidia_gpu.fence_async_shared
    // CHECK: scf.for
    // CHECK-NOT: triton_nvidia_gpu.fence_async_shared
    // CHECK:   triton_nvidia_gpu.warp_group_dot
    scf.for %iv0 = %c0_i32 to %c64_i32 step %c32_i32 : i32 {
      scf.for %iv1 = %c0_i32 to %c64_i32 step %c32_i32 : i32 {
        %2 = triton_nvidia_gpu.warp_group_dot %0, %1, %cst : !triton_gpu.memdesc<128x128xf16, #shared> * !triton_gpu.memdesc<128x64xf16, #shared1> -> tensor<128x64xf32, #mma>
      }
    }
    tt.return
  }
}
