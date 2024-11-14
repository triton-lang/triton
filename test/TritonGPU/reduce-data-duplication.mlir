// RUN: triton-opt %s -split-input-file -tritongpu-reduce-data-duplication | FileCheck %s

//       CHECK:   #[[$SHARED:.*]] = #triton_gpu.shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [0, 1], hasLeadingOffset = false, inThreadTranspose = false}
//       CHECK-LABEL: apply_swizzle
//       CHECK:   %{{.*}} = triton_gpu.local_alloc %{{.*}} : (tensor<16x256xf16, #{{.*}}>) -> !tt.memdesc<16x256xf16, #[[$SHARED]], #triton_gpu.shared_memory>

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @apply_swizzle(%arg0: tensor<16x256xf16, #blocked>) {
    %0 = triton_gpu.convert_layout %arg0 : tensor<16x256xf16, #blocked> -> tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
}

// -----

//       CHECK-LABEL:   conversion_shortcut_blocked_dotop_warp32
//       CHECK-NOT:  triton_gpu.local_alloc
//       CHECK: triton_gpu.convert_layout
//       CHECK-NOT:  triton_gpu.local_alloc
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 64], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [0, 1]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @conversion_shortcut_blocked_dotop_warp32(%arg0: tensor<64x64xf16, #blocked>) {
    %0 = triton_gpu.convert_layout %arg0 : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

//       CHECK-LABEL:   conversion_shortcut_blocked_dotop_warp64
//       CHECK-NOT:  triton_gpu.local_alloc
//       CHECK: triton_gpu.convert_layout
//       CHECK-NOT:  triton_gpu.local_alloc
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 2], warpsPerCTA = [2, 2], order = [0, 1]}>
module attributes {"triton_gpu.target" = "hip:gfx940", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @conversion_shortcut_blocked_dotop_warp64(%arg0: tensor<64x64xf16, #blocked>) {
    %0 = triton_gpu.convert_layout %arg0 : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
    tt.return
  }
}

// -----

// CHECK: #[[$shared_layout_transpose:.*]] = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [0, 1], hasLeadingOffset = false, inThreadTranspose = true}>
// CHECK-LABEL: threadRake_shared
// CHECK: [[shared_ptr:%.*]] = triton_gpu.local_alloc {{.*}} -> !tt.memdesc<32x128xf16, #[[$shared_layout_transpose]], #triton_gpu.shared_memory>
// CHECK: [[opB:%.*]] = triton_gpu.local_load [[shared_ptr]] : !tt.memdesc<32x128xf16, #[[$shared_layout_transpose]], #triton_gpu.shared_memory> -> {{.*}}
// CHECK: {{.*}} = tt.dot {{.*}}, [[opB]], {{.*}}
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [32, 32], isTransposed = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @threadRake_shared(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x32x!tt.ptr<f16>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %2 = tt.load %0 : tensor<256x32x!tt.ptr<f16>, #blocked>
    %3 = tt.load %1 : tensor<32x128x!tt.ptr<f16>, #blocked1>
    %4 = triton_gpu.convert_layout %2 : tensor<256x32xf16, #blocked> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %5 = triton_gpu.convert_layout %3 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %6 = tt.dot %4, %5, %cst_0 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
    tt.return
  }
}