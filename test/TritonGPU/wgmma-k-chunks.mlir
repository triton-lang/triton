// RUN: triton-opt %s -split-input-file -tritongpu-pipeline -canonicalize | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @rs_wgmma_k_chunks_4
  tt.func public @rs_wgmma_k_chunks_4(%a: tensor<128x64xf16, #dot>, %b: !ttg.memdesc<64x64xf16, #shared, #smem>, %ub: i32) -> tensor<128x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %0 = scf.for %i = %c0 to %ub step %c1 iter_args(%acc_iter = %acc) -> (tensor<128x64xf32, #mma>)  : i32 {
      // CHECK: tt.reshape
      // CHECK: = ttng.warp_group_dot %{{.*}} : tensor<128x16xf16
      // CHECK: = ttng.warp_group_dot %{{.*}} : tensor<128x16xf16
      // CHECK: = ttng.warp_group_dot %{{.*}} : tensor<128x16xf16
      // CHECK: = ttng.warp_group_dot %{{.*}} : tensor<128x16xf16
      // CHECK-NOT: = ttng.warp_group_dot %
      // CHECK: scf.yield
      %dot_result = ttng.warp_group_dot %a, %b, %acc_iter {kChunksHint = 4 : i32} : tensor<128x64xf16, #dot> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
      scf.yield %dot_result : tensor<128x64xf32, #mma>
    }
    tt.return %0 : tensor<128x64xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @rs_wgmma_k_chunks_1
  tt.func public @rs_wgmma_k_chunks_1(%a: tensor<128x64xf16, #dot>, %b: !ttg.memdesc<64x64xf16, #shared, #smem>, %ub: i32) -> tensor<128x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %0 = scf.for %i = %c0 to %ub step %c1 iter_args(%acc_iter = %acc) -> (tensor<128x64xf32, #mma>)  : i32 {
      // CHECK: scf.for
      // CHECK-NOT: tt.reshape
      // CHECK: = ttng.warp_group_dot %{{.*}}kChunksHint = 1 : i32{{.*}} : tensor<128x64xf16
      // CHECK-NOT: = ttng.warp_group_dot %
      // CHECK: scf.yield
      %dot_result = ttng.warp_group_dot %a, %b, %acc_iter {kChunksHint = 1 : i32} : tensor<128x64xf16, #dot> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
      scf.yield %dot_result : tensor<128x64xf32, #mma>
    }
    tt.return %0 : tensor<128x64xf32, #mma>
  }
}
