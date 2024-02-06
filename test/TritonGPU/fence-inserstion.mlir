// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-fence-insertion | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#mma1 = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: matmul_like_fence
  tt.func public @matmul_like_fence(%arg0: tensor<128x128xf16, #blocked>, %arg1: tensor<128x64xf16, #blocked2>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %0 = triton_gpu.convert_layout %arg0 : (tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #shared>
    %1 = triton_gpu.convert_layout %arg1 : (tensor<128x64xf16, #blocked2>) -> tensor<128x64xf16, #shared1>
    // CHECK: triton_nvidia_gpu.fence_async_shared
    %2 = tt.dot %0, %1, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x128xf16, #shared> * tensor<128x64xf16, #shared1> -> tensor<128x64xf32, #mma>
    tt.return
  }
}
