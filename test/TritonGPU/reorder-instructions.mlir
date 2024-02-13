// RUN: triton-opt %s -split-input-file -tritongpu-reorder-instructions | FileCheck %s

// check that we don't hoist convert_layout above its operand definition.
// CHECK-LABEL: convert_cannot_hoist
//       CHECK:   %[[CVTS:.+]] = triton_gpu.convert_layout
//       CHECK:   triton_gpu.convert_layout %[[CVTS]]
//       CHECK:   tt.dot
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @convert_cannot_hoist(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %9 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #blocked>
    %10 = triton_gpu.convert_layout %9 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #shared>
    %11 = triton_gpu.convert_layout %10 : tensor<32x32xf32, #shared> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %11, %cst_0, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %13 = triton_gpu.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 {cache = 1 : i32, evict = 1 : i32} : tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: sink_convert_dealloc
//       CHECK: triton_gpu.async_wait {num = 0 : i32}
//       CHECK: triton_gpu.dealloc_tensor %0 : tensor<4x128x64xf16, #shared>
//       CHECK: triton_gpu.dealloc_tensor %1 : tensor<4x128x64xf16, #shared>
//       CHECK: %2 = triton_gpu.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @sink_convert_dealloc(%arg0: tensor<32x32xf32, #blocked>) attributes {noinline = false} {
    %0 = triton_gpu.alloc_tensor : tensor<4x128x64xf16, #shared>
    %1 = triton_gpu.alloc_tensor : tensor<4x128x64xf16, #shared>
    %2 = triton_gpu.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    triton_gpu.async_wait {num = 0 : i32}
    triton_gpu.dealloc_tensor %0 : tensor<4x128x64xf16, #shared>
    triton_gpu.dealloc_tensor %1 : tensor<4x128x64xf16, #shared>
    %3 = arith.addf %2, %2 : tensor<32x32xf32, #blocked1>
    tt.return
  }
}

// -----

// CHECK-LABEL: sink_convert_idx_1
//       CHECK: triton_gpu.convert_layout %{{.*}} : tensor<32x32xf32, #shared> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
//       CHECK: triton_gpu.convert_layout %{{.*}} : tensor<32x32xf32, #shared> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
//       CHECK: tt.dot
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @sink_convert_idx_1(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %B = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #blocked>
    %BS = triton_gpu.convert_layout %B : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #shared>
    %BD = triton_gpu.convert_layout %BS : tensor<32x32xf32, #shared> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
    %A = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #blocked>
    %AS = triton_gpu.convert_layout %A : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #shared>
    %AD = triton_gpu.convert_layout %AS : tensor<32x32xf32, #shared> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %12 = tt.dot %AD, %BD, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
    %13 = triton_gpu.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 {cache = 1 : i32, evict = 1 : i32} : tensor<32x32xf32, #blocked>
    tt.return
  }
}
