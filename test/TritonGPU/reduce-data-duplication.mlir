// RUN: triton-opt %s -split-input-file -tritongpu-reduce-data-duplication | FileCheck %s

//       CHECK:   #[[SHARED:.*]] = #triton_gpu.shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [0, 1], hasLeadingOffset = false}
//       CHECK:   apply_swizzle
//       CHECK:   %{{.*}} = triton_gpu.local_alloc %{{.*}} : (tensor<16x256xf16, #{{.*}}>) -> !tt.memdesc<16x256xf16, #[[SHARED]], #triton_gpu.shared_memory>

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @apply_swizzle(%arg0: tensor<16x256xf16, #blocked>) {
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x256xf16, #blocked> -> tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
}


// -----

//       CHECK:   apply_swizzle1
//       CHECK:   %{{.*}} = tt.load %{{.*}} {shouldHoist = true}
//       CHECK:   %{{.*}} = triton_gpu.local_alloc %{{.*}} : (tensor<16x256xf16, #{{.*}}>) -> !tt.memdesc<16x256xf16, #[[SHARED]], #triton_gpu.shared_memory>
//       CHECK:   %{{.*}} = triton_gpu.local_load %{{.*}} shouldHoist = true : !tt.memdesc<16x256xf16, #[[SHARED]], #triton_gpu.shared_memory> -> tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @apply_swizzle1(%arg0: tensor<16x256x!tt.ptr<f16>, #blocked>) {
    %0 = tt.load %arg0 {shouldHoist = true} : tensor<16x256x!tt.ptr<f16>, #blocked>
    %1 = triton_gpu.convert_layout %0 : tensor<16x256xf16, #blocked> -> tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
}
