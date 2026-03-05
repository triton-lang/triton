// RUN: triton-opt %s -split-input-file -tritonamdgpu-schedule-loops="num_stages=2" -tritonamdgpu-pipeline="use_async_copy=1" -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0], [4, 0]]}, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @matmul_kernel_make_tensor_descriptor(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) {
    %c512_i32 = arith.constant 512 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<512x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c512_i32 : i32
    %3 = arith.muli %1, %c64_i32 : i32
    %4 = arith.extsi %K : i32 to i64
    %5 = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%4, %c1_i64] : <f16>, <tensor<512x32xf16>>
    %6 = arith.extsi %N : i32 to i64
    %7 = tt.make_tensor_descriptor %b_ptr, [%K, %N], [%6, %c1_i64] : <f16>, <tensor<32x64xf16>>
    %8 = tt.make_tensor_descriptor %c_ptr, [%M, %N], [%6, %c1_i64] : <f16>, <tensor<512x64xf16>>
    %9 = arith.addi %K, %c31_i32 : i32
    %10 = arith.divsi %9, %c32_i32 : i32
    %accumulator:2 = scf.for %accumulator_0 = %c0_i32 to %10 step %c1_i32 iter_args(%arg7 = %c0_i32, %arg8 = %cst) -> (i32, tensor<512x64xf32, #mma>)  : i32 {
      %13 = tt.descriptor_load %5[%2, %arg7] : !tt.tensordesc<tensor<512x32xf16>> -> tensor<512x32xf16, #blocked>
      %14 = tt.descriptor_load %7[%arg7, %3] : !tt.tensordesc<tensor<32x64xf16>> -> tensor<32x64xf16, #blocked1>
      %15 = ttg.convert_layout %13 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %16 = ttg.convert_layout %14 : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %17 = tt.dot %15, %16, %arg8 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
      %18 = arith.addi %arg7, %c32_i32 : i32
      scf.yield %18, %17 : i32, tensor<512x64xf32, #mma>
    }
    %11 = arith.truncf %accumulator#1 : tensor<512x64xf32, #mma> to tensor<512x64xf16, #mma>
    %12 = ttg.convert_layout %11 : tensor<512x64xf16, #mma> -> tensor<512x64xf16, #blocked1>
    tt.descriptor_store %8[%2, %3], %12 : !tt.tensordesc<tensor<512x64xf16>>, tensor<512x64xf16, #blocked1>
    tt.return
  }
}

// CHECK-LABEL: tt.func @matmul_kernel_make_tensor_descriptor
// CHECK: async_tdm_copy_global_to_local
// CHECK: ttg.async_commit_group tokens
// CHECK: async_tdm_copy_global_to_local
// CHECK: ttg.async_commit_group tokens
// CHECK: scf.for
// CHECK: amdg.async_tdm_wait
// CHECK: async_tdm_copy_global_to_local
// CHECK: ttg.async_commit_group tokens
// CHECK: }
