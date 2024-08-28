// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s

// CHECK-LABEL: blocked_to_dot_op_shortcut_warp32
#blocked = #triton_gpu.blocked<{sizePerThread = [32, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [0, 1]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @blocked_to_dot_op_shortcut_warp32(%arg0: tensor<32x32xf16, #blocked>, %arg1: tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>) {
    %0 = triton_gpu.convert_layout %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK-NOT: load
    tt.return
  }
}

// -----

// CHECK-LABEL: blocked_to_dot_op_shortcut_warp64
#blocked = #triton_gpu.blocked<{sizePerThread = [32, 1], threadsPerWarp = [2, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "hip:gfx940", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @blocked_to_dot_op_shortcut_warp64(%arg0: tensor<32x32xf16, #blocked>) {
    %0 = triton_gpu.convert_layout %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK-NOT: load
    tt.return
  }
}

// -----

// CHECK-LABEL: blocked_to_dot3d_op_shortcut_warp32
#blocked = #triton_gpu.blocked<{sizePerThread = [2, 32, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 1, 2], order = [1, 2, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @blocked_to_dot3d_op_shortcut_warp32(%arg0: tensor<8x32x32xf16, #blocked>) {
    %0 = triton_gpu.convert_layout %arg0 : tensor<8x32x32xf16, #blocked> -> tensor<8x32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK-NOT: load
    tt.return
  }
}

// -----

// CHECK-LABEL: blocked_to_dot3d_op_shortcut_warp64
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 32, 1], threadsPerWarp = [1, 2, 32], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "hip:gfx940", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @blocked_to_dot3d_op_shortcut_warp64(%arg0: tensor<8x32x32xf16, #blocked>) {
    %0 = triton_gpu.convert_layout %arg0 : tensor<8x32x32xf16, #blocked> -> tensor<8x32x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK-NOT: load
    tt.return
  }
}
