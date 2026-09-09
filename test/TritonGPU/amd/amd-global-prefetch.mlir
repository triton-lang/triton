// RUN: triton-opt %s -split-input-file -tritonamdgpu-schedule-loops=num_stages=2 -tritonamdgpu-pipeline="use_async_copy=true use_pingpong=false use_l2_prefetch=true" | FileCheck %s --check-prefixes=NS2_GPD1

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // NS2_GPD1-LABEL: gemm_tdm_prefetch_ns2_gpd1
  tt.func public @gemm_tdm_prefetch_ns2_gpd1(
    %desc0: !tt.tensordesc<64x64xbf16, #shared>,
    %desc1: !tt.tensordesc<64x64xbf16, #shared>) -> tensor<64x64xf32, #mma> attributes {noinline = false} {

    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 8 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>

    // NS2_GPD1: scf.for
    %loop = scf.for %iter_var = %c0_i32 to %c512_i32 step %c64_i32 iter_args(%acc = %zero) -> (tensor<64x64xf32, #mma>)  : i32 {
      // NS2_GPD1: amdg.tdm_prefetch %[[DESC0:.+]][
      // NS2_GPD1: amdg.tdm_prefetch %[[DESC1:.+]][

      // NS2_GPD1: %[[HANDLE0:.+]] = amdg.update_tensor_descriptor %[[DESC0]]
      // NS2_GPD1: amdg.async_tdm_copy_global_to_local %[[HANDLE0]]
      %t0 = tt.descriptor_load %desc0[%c0_i32, %iter_var] : !tt.tensordesc<64x64xbf16, #shared> -> tensor<64x64xbf16, #blocked>

      // NS2_GPD1: %[[HANDLE1:.+]] = amdg.update_tensor_descriptor %[[DESC1]]
      // NS2_GPD1: amdg.async_tdm_copy_global_to_local %[[HANDLE1]]
      %t1 = tt.descriptor_load %desc1[%iter_var, %c0_i32] : !tt.tensordesc<64x64xbf16, #shared> -> tensor<64x64xbf16, #blocked>

      %a = ttg.convert_layout %t0 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b = ttg.convert_layout %t1 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %res = tt.dot %a, %b, %acc : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x64xf32, #mma>
      scf.yield %res : tensor<64x64xf32, #mma>
    }

    tt.return %loop#0 : tensor<64x64xf32, #mma>
  }
}
