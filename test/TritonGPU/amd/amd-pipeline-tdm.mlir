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

// Operand A (opIdx=0, order=[1,0]): loadTransposed = (1 != 1) = false → non-transposed
//   padAmount = min(kWidth=8, 128/16) = min(8, 8) = 8
//   innerDimLength = shape[order[0]] = shape[1] = 32 (K dim)
//   → padded_shared<[32:+8]>
//
// Operand B (opIdx=1, order=[1,0]): loadTransposed = (1 != 0) = true → transposed
//   queryLDSTransLoadParams(16) → instBitWidth=128, padAmount = 2*128/16 = 16
//   innerDimLength = shape[order[0]] = shape[1] = 64 (N dim)
//   → padded_shared<[64:+16]>
// CHECK:     #ttg.padded_shared<[128:+8] {
// CHECK-NOT: #ttg.padded_shared
// CHECK:     #ttg.padded_shared<[128:+16] {
// CHECK-NOT: #ttg.padded_shared

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

// -----

// Test TDM padding for fp8 (f8E5M2) matmul on gfx1250.
//
// Operand A (opIdx=0, order=[1,0]): loadTransposed = (1 != 1) = false → non-transposed
//   padAmount = 128/8 = 16 (sub-dword: no min with vecWidth, full 4-dword
//   stride separation needed for ds_load_2addr_b64 cross-address conflicts)
//   innerDimLength = shape[1] = 64 (K dim)
//   → padded_shared<[64:+16]>
//
// Operand B (opIdx=1, order=[1,0]): loadTransposed = (1 != 0) = true → transposed
//   queryLDSTransLoadParams(8) → instBitWidth=64, padAmount = 2*64/8 = 16
//   innerDimLength = shape[1] = 64 (N dim)
//   → padded_shared<[64:+16]>
// CHECK:     #ttg.padded_shared<[256:+16] {
// CHECK-NOT: #ttg.padded_shared
// CHECK:     #ttg.padded_shared<[256:+16] {
// CHECK-NOT: #ttg.padded_shared

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0], [4, 0]]}, instrShape = [16, 16, 64]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tdm_padding_fp8(%a_ptr: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32},
    %b_ptr: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c256_i32 : i32
    %3 = arith.muli %1, %c64_i32 : i32
    %4 = arith.extsi %K : i32 to i64
    %5 = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%4, %c1_i64] : <f8E5M2>, <tensor<256x64xf8E5M2>>
    %6 = arith.extsi %N : i32 to i64
    %7 = tt.make_tensor_descriptor %b_ptr, [%K, %N], [%6, %c1_i64] : <f8E5M2>, <tensor<64x64xf8E5M2>>
    %8 = tt.make_tensor_descriptor %c_ptr, [%M, %N], [%6, %c1_i64] : <f16>, <tensor<256x64xf16>>
    %9 = arith.addi %K, %c63_i32 : i32
    %10 = arith.divsi %9, %c64_i32 : i32
    %accumulator:2 = scf.for %iv = %c0_i32 to %10 step %c1_i32 iter_args(%k_off = %c0_i32, %acc = %cst) -> (i32, tensor<256x64xf32, #mma>)  : i32 {
      %a = tt.descriptor_load %5[%2, %k_off] : !tt.tensordesc<tensor<256x64xf8E5M2>> -> tensor<256x64xf8E5M2, #blocked>
      %b = tt.descriptor_load %7[%k_off, %3] : !tt.tensordesc<tensor<64x64xf8E5M2>> -> tensor<64x64xf8E5M2, #blocked1>
      %a_dot = ttg.convert_layout %a : tensor<256x64xf8E5M2, #blocked> -> tensor<256x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_dot = ttg.convert_layout %b : tensor<64x64xf8E5M2, #blocked1> -> tensor<64x64xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_dot, %b_dot, %acc : tensor<256x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x64xf32, #mma>
      %next_k = arith.addi %k_off, %c64_i32 : i32
      scf.yield %next_k, %d : i32, tensor<256x64xf32, #mma>
    }
    %out = arith.truncf %accumulator#1 : tensor<256x64xf32, #mma> to tensor<256x64xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<256x64xf16, #mma> -> tensor<256x64xf16, #blocked1>
    tt.descriptor_store %8[%2, %3], %out_blocked : !tt.tensordesc<tensor<256x64xf16>>, tensor<256x64xf16, #blocked1>
    tt.return
  }
}

// CHECK-LABEL: tt.func @tdm_padding_fp8
// CHECK: async_tdm_copy_global_to_local
// CHECK: async_tdm_copy_global_to_local
// CHECK: scf.for
// CHECK: }

// -----

// Test TDM padding for f32 matmul on gfx1250.
//
// Operand A (opIdx=0, order=[1,0]): loadTransposed = (1 != 1) = false → non-transposed
//   padAmount = min(kWidth=8, 128/32) = min(8, 4) = 4
//   innerDimLength = shape[1] = 16 (K dim)
//   → padded_shared<[16:+4]>
//
// Operand B (opIdx=1, order=[1,0]): loadTransposed = (1 != 0) = true → transposed
//   queryLDSTransLoadParams(32) → nullopt, falls back to padAmount = 128/32 = 4
//   innerDimLength = shape[1] = 64 (N dim)
//   → padded_shared<[64:+4]>
// CHECK:     #ttg.padded_shared<[64:+4] {
// CHECK-NOT: #ttg.padded_shared
// CHECK:     #ttg.padded_shared<[64:+4] {
// CHECK-NOT: #ttg.padded_shared

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0], [4, 0]]}, instrShape = [16, 16, 4]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tdm_padding_f32(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %c15_i32 = arith.constant 15 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c256_i32 : i32
    %3 = arith.muli %1, %c64_i32 : i32
    %4 = arith.extsi %K : i32 to i64
    %5 = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%4, %c1_i64] : <f32>, <tensor<256x16xf32>>
    %6 = arith.extsi %N : i32 to i64
    %7 = tt.make_tensor_descriptor %b_ptr, [%K, %N], [%6, %c1_i64] : <f32>, <tensor<16x64xf32>>
    %8 = tt.make_tensor_descriptor %c_ptr, [%M, %N], [%6, %c1_i64] : <f32>, <tensor<256x64xf32>>
    %9 = arith.addi %K, %c15_i32 : i32
    %10 = arith.divsi %9, %c16_i32 : i32
    %accumulator:2 = scf.for %iv = %c0_i32 to %10 step %c1_i32 iter_args(%k_off = %c0_i32, %acc = %cst) -> (i32, tensor<256x64xf32, #mma>)  : i32 {
      %a = tt.descriptor_load %5[%2, %k_off] : !tt.tensordesc<tensor<256x16xf32>> -> tensor<256x16xf32, #blocked>
      %b = tt.descriptor_load %7[%k_off, %3] : !tt.tensordesc<tensor<16x64xf32>> -> tensor<16x64xf32, #blocked1>
      %a_dot = ttg.convert_layout %a : tensor<256x16xf32, #blocked> -> tensor<256x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_dot = ttg.convert_layout %b : tensor<16x64xf32, #blocked1> -> tensor<16x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_dot, %b_dot, %acc, inputPrecision = tf32 : tensor<256x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x64xf32, #mma>
      %next_k = arith.addi %k_off, %c16_i32 : i32
      scf.yield %next_k, %d : i32, tensor<256x64xf32, #mma>
    }
    %out_blocked = ttg.convert_layout %accumulator#1 : tensor<256x64xf32, #mma> -> tensor<256x64xf32, #blocked1>
    tt.descriptor_store %8[%2, %3], %out_blocked : !tt.tensordesc<tensor<256x64xf32>>, tensor<256x64xf32, #blocked1>
    tt.return
  }
}

// CHECK-LABEL: tt.func @tdm_padding_f32
// CHECK: async_tdm_copy_global_to_local
// CHECK: async_tdm_copy_global_to_local
// CHECK: scf.for
// CHECK: }
