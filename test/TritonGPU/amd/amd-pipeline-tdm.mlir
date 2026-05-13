// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-descriptor-encoding  -tritonamdgpu-schedule-loops="num_stages=2" -tritonamdgpu-pipeline="use_async_copy=1" -canonicalize | FileCheck %s

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
    %5 = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%4, %c1_i64] : <f16>, <512x32xf16>
    %6 = arith.extsi %N : i32 to i64
    %7 = tt.make_tensor_descriptor %b_ptr, [%K, %N], [%6, %c1_i64] : <f16>, <32x64xf16>
    %8 = tt.make_tensor_descriptor %c_ptr, [%M, %N], [%6, %c1_i64] : <f16>, <512x64xf16>
    %9 = arith.addi %K, %c31_i32 : i32
    %10 = arith.divsi %9, %c32_i32 : i32
    %accumulator:2 = scf.for %accumulator_0 = %c0_i32 to %10 step %c1_i32 iter_args(%arg7 = %c0_i32, %arg8 = %cst) -> (i32, tensor<512x64xf32, #mma>)  : i32 {
      %13 = tt.descriptor_load %5[%2, %arg7] : !tt.tensordesc<512x32xf16> -> tensor<512x32xf16, #blocked>
      %14 = tt.descriptor_load %7[%arg7, %3] : !tt.tensordesc<32x64xf16> -> tensor<32x64xf16, #blocked1>
      %15 = ttg.convert_layout %13 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %16 = ttg.convert_layout %14 : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %17 = tt.dot %15, %16, %arg8 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
      %18 = arith.addi %arg7, %c32_i32 : i32
      scf.yield %18, %17 : i32, tensor<512x64xf32, #mma>
    }
    %11 = arith.truncf %accumulator#1 : tensor<512x64xf32, #mma> to tensor<512x64xf16, #mma>
    %12 = ttg.convert_layout %11 : tensor<512x64xf16, #mma> -> tensor<512x64xf16, #blocked1>
    tt.descriptor_store %8[%2, %3], %12 : !tt.tensordesc<512x64xf16>, tensor<512x64xf16, #blocked1>
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
//
// tt.desciptor_store -> padded_shared<[64:+8]>
// CHECK: #[[$PADDED_A:.*]] = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [512, 32]}>
// CHECK: #[[$PADDED_B:.*]] = #ttg.padded_shared<[128:+16] {order = [1, 0], shape = [32, 64]}>
// CHECK: #[[$PADDED_C:.*]] = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [512, 64]}>
// CHECK-NOT: #ttg.padded_shared

// The loop body and epilogue each emit two adjacent amdg.async_tdm_wait ops
// (one per descriptor_load) which combineRedundantWaitOps folds into a single
// wait taking both tokens; the matched two-operand wait below proves the fold.
// CHECK-LABEL: tt.func @matmul_kernel_make_tensor_descriptor
// CHECK: async_tdm_copy_global_to_local {{.*}} : !tt.tensordesc<512x32xf16, #[[$PADDED_A]]> -> !ttg.memdesc<512x32xf16, #[[$PADDED_A]], #smem, mutable>
// CHECK-NOT: ttg.async_commit_group
// CHECK: async_tdm_copy_global_to_local {{.*}} : !tt.tensordesc<32x64xf16, #[[$PADDED_B]]> -> !ttg.memdesc<32x64xf16, #[[$PADDED_B]], #smem, mutable>
// CHECK-NOT: ttg.async_commit_group
// CHECK: scf.for
// CHECK: amdg.async_tdm_wait %{{[^,]+}}, %{{[^,]+}} {num = 0 : i32}
// CHECK-NOT: amdg.async_tdm_wait
// CHECK: async_tdm_copy_global_to_local {{.*}} : !tt.tensordesc<512x32xf16, #[[$PADDED_A]]> -> !ttg.memdesc<512x32xf16, #[[$PADDED_A]], #smem, mutable>
// CHECK-NOT: ttg.async_commit_group
// CHECK: async_tdm_copy_global_to_local {{.*}} : !tt.tensordesc<32x64xf16, #[[$PADDED_B]]> -> !ttg.memdesc<32x64xf16, #[[$PADDED_B]], #smem, mutable>
// CHECK-NOT: ttg.async_commit_group
// CHECK: }
// CHECK: amdg.async_tdm_wait %{{[^,]+}}, %{{[^,]+}} {num = 0 : i32}
// CHECK-NOT: amdg.async_tdm_wait
// CHECK: tt.descriptor_store {{.*}} : !tt.tensordesc<512x64xf16, #[[$PADDED_C]]>

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
//
// tt.descriptor_store -> padded_shared<[64:+8]>
// CHECK: #[[$PADDED_A:.*]] = #ttg.padded_shared<[256:+16] {order = [1, 0], shape = [256, 64]}>
// CHECK: #[[$PADDED_B:.*]] = #ttg.padded_shared<[256:+16] {order = [1, 0], shape = [64, 64]}>
// CHECK: #[[$PADDED_C:.*]] = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [256, 64]}>
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
    %5 = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%4, %c1_i64] : <f8E5M2>, <256x64xf8E5M2>
    %6 = arith.extsi %N : i32 to i64
    %7 = tt.make_tensor_descriptor %b_ptr, [%K, %N], [%6, %c1_i64] : <f8E5M2>, <64x64xf8E5M2>
    %8 = tt.make_tensor_descriptor %c_ptr, [%M, %N], [%6, %c1_i64] : <f16>, <256x64xf16>
    %9 = arith.addi %K, %c63_i32 : i32
    %10 = arith.divsi %9, %c64_i32 : i32
    %accumulator:2 = scf.for %iv = %c0_i32 to %10 step %c1_i32 iter_args(%k_off = %c0_i32, %acc = %cst) -> (i32, tensor<256x64xf32, #mma>)  : i32 {
      %a = tt.descriptor_load %5[%2, %k_off] : !tt.tensordesc<256x64xf8E5M2> -> tensor<256x64xf8E5M2, #blocked>
      %b = tt.descriptor_load %7[%k_off, %3] : !tt.tensordesc<64x64xf8E5M2> -> tensor<64x64xf8E5M2, #blocked1>
      %a_dot = ttg.convert_layout %a : tensor<256x64xf8E5M2, #blocked> -> tensor<256x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_dot = ttg.convert_layout %b : tensor<64x64xf8E5M2, #blocked1> -> tensor<64x64xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_dot, %b_dot, %acc : tensor<256x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x64xf32, #mma>
      %next_k = arith.addi %k_off, %c64_i32 : i32
      scf.yield %next_k, %d : i32, tensor<256x64xf32, #mma>
    }
    %out = arith.truncf %accumulator#1 : tensor<256x64xf32, #mma> to tensor<256x64xf16, #mma>
    %out_blocked = ttg.convert_layout %out : tensor<256x64xf16, #mma> -> tensor<256x64xf16, #blocked1>
    tt.descriptor_store %8[%2, %3], %out_blocked : !tt.tensordesc<256x64xf16>, tensor<256x64xf16, #blocked1>
    tt.return
  }
}

// CHECK-LABEL: tt.func @tdm_padding_fp8
// CHECK: async_tdm_copy_global_to_local {{.*}} : !tt.tensordesc<256x64xf8E5M2, #[[$PADDED_A]]> -> !ttg.memdesc<256x64xf8E5M2, #[[$PADDED_A]], #smem, mutable>
// CHECK-NOT: ttg.async_commit_group
// CHECK: async_tdm_copy_global_to_local {{.*}} : !tt.tensordesc<64x64xf8E5M2, #[[$PADDED_B]]> -> !ttg.memdesc<64x64xf8E5M2, #[[$PADDED_B]], #smem, mutable>
// CHECK-NOT: ttg.async_commit_group
// CHECK: scf.for
// CHECK: async_tdm_copy_global_to_local {{.*}} : !tt.tensordesc<256x64xf8E5M2, #[[$PADDED_A]]> -> !ttg.memdesc<256x64xf8E5M2, #[[$PADDED_A]], #smem, mutable>
// CHECK-NOT: ttg.async_commit_group
// CHECK: async_tdm_copy_global_to_local {{.*}} : !tt.tensordesc<64x64xf8E5M2, #[[$PADDED_B]]> -> !ttg.memdesc<64x64xf8E5M2, #[[$PADDED_B]], #smem, mutable>
// CHECK-NOT: ttg.async_commit_group
// CHECK: }
// CHECK: tt.descriptor_store {{.*}} : !tt.tensordesc<256x64xf16, #[[$PADDED_C]]>

// -----

// Test TDM padding for f32 matmul on gfx1250.
//
// Operand A (opIdx=0, order=[1,0]): loadTransposed = (1 != 1) = false → non-transposed
//   padAmount = min(kWidth=8, 128/32) = min(8, 4) = 4
//   innerDimLength = shape[1] = 16 (K dim)
//   → padded_shared<[16:+4]>
//
// Operand B (opIdx=1, order=[1,0]): loadTransposed = (1 != 0) = true → transposed
//   queryLDSTransLoadParams(32) → empty, falls back to padAmount = 128/32 = 4
//   innerDimLength = shape[1] = 64 (N dim)
//   → padded_shared<[64:+4]>
//
// tt.desciptor_store -> padded_shared<[64:+4]>
// CHECK: #[[$PADDED_A:.*]] = #ttg.padded_shared<[64:+4] {order = [1, 0], shape = [256, 16]}>
// CHECK: #[[$PADDED_B:.*]] = #ttg.padded_shared<[64:+4] {order = [1, 0], shape = [16, 64]}>
// CHECK: #[[$PADDED_C:.*]] = #ttg.padded_shared<[64:+4] {order = [1, 0], shape = [256, 64]}>
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
    %5 = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%4, %c1_i64] : <f32>, <256x16xf32>
    %6 = arith.extsi %N : i32 to i64
    %7 = tt.make_tensor_descriptor %b_ptr, [%K, %N], [%6, %c1_i64] : <f32>, <16x64xf32>
    %8 = tt.make_tensor_descriptor %c_ptr, [%M, %N], [%6, %c1_i64] : <f32>, <256x64xf32>
    %9 = arith.addi %K, %c15_i32 : i32
    %10 = arith.divsi %9, %c16_i32 : i32
    %accumulator:2 = scf.for %iv = %c0_i32 to %10 step %c1_i32 iter_args(%k_off = %c0_i32, %acc = %cst) -> (i32, tensor<256x64xf32, #mma>)  : i32 {
      %a = tt.descriptor_load %5[%2, %k_off] : !tt.tensordesc<256x16xf32> -> tensor<256x16xf32, #blocked>
      %b = tt.descriptor_load %7[%k_off, %3] : !tt.tensordesc<16x64xf32> -> tensor<16x64xf32, #blocked1>
      %a_dot = ttg.convert_layout %a : tensor<256x16xf32, #blocked> -> tensor<256x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_dot = ttg.convert_layout %b : tensor<16x64xf32, #blocked1> -> tensor<16x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_dot, %b_dot, %acc, inputPrecision = tf32 : tensor<256x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x64xf32, #mma>
      %next_k = arith.addi %k_off, %c16_i32 : i32
      scf.yield %next_k, %d : i32, tensor<256x64xf32, #mma>
    }
    %out_blocked = ttg.convert_layout %accumulator#1 : tensor<256x64xf32, #mma> -> tensor<256x64xf32, #blocked1>
    tt.descriptor_store %8[%2, %3], %out_blocked : !tt.tensordesc<256x64xf32>, tensor<256x64xf32, #blocked1>
    tt.return
  }
}

// CHECK-LABEL: tt.func @tdm_padding_f32
// CHECK: async_tdm_copy_global_to_local {{.*}} : !tt.tensordesc<256x16xf32, #[[$PADDED_A]]> -> !ttg.memdesc<256x16xf32, #[[$PADDED_A]], #smem, mutable>
// CHECK-NOT: ttg.async_commit_group
// CHECK: async_tdm_copy_global_to_local {{.*}} : !tt.tensordesc<16x64xf32, #[[$PADDED_B]]> -> !ttg.memdesc<16x64xf32, #[[$PADDED_B]], #smem, mutable>
// CHECK-NOT: ttg.async_commit_group
// CHECK: scf.for
// CHECK: async_tdm_copy_global_to_local {{.*}} : !tt.tensordesc<256x16xf32, #[[$PADDED_A]]> -> !ttg.memdesc<256x16xf32, #[[$PADDED_A]], #smem, mutable>
// CHECK-NOT: ttg.async_commit_group
// CHECK: async_tdm_copy_global_to_local {{.*}} : !tt.tensordesc<16x64xf32, #[[$PADDED_B]]> -> !ttg.memdesc<16x64xf32, #[[$PADDED_B]], #smem, mutable>
// CHECK-NOT: ttg.async_commit_group
// CHECK: }
// CHECK: tt.descriptor_store {{.*}} : !tt.tensordesc<256x64xf32, #[[$PADDED_C]]>

// -----

// Test TDM pipeline for gather + dot on gfx1250.
// Two gathers (A and B) inside a loop feed into a dot. The pipeline pass
// should convert them to async_tdm_gather ops and software-pipeline the loop.

#blocked_ga = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_gb = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma_g = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0], [4, 0]]}, instrShape = [16, 16, 32]}>
#padded_ga = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [1, 32]}>
#padded_gb = #ttg.padded_shared<[16:+16] {order = [1, 0], shape = [1, 16]}>
#padded_gc = #ttg.padded_shared<[16:+8] {order = [1, 0], shape = [16, 16]}>
#idx_enc = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 8], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @gather_dot_pipeline(
      %a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %K: i32 {tt.divisibility = 16 : i32}) {
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma_g>
    %4 = arith.extsi %K : i32 to i64
    %a_desc = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%4, %c1_i64] : <f16>, <1x32xf16, #padded_ga>
    %6 = arith.extsi %N : i32 to i64
    %b_desc = tt.make_tensor_descriptor %b_ptr, [%K, %N], [%6, %c1_i64] : <f16>, <1x16xf16, #padded_gb>
    %c_desc = tt.make_tensor_descriptor %c_ptr, [%M, %N], [%6, %c1_i64] : <f16>, <16x16xf16, #padded_gc>
    %a_indices = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #idx_enc}>>
    %b_indices = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #idx_enc}>>
    %9 = arith.addi %K, %c31_i32 : i32
    %10 = arith.divsi %9, %c32_i32 : i32
    %accumulator:2 = scf.for %iv = %c0_i32 to %10 step %c1_i32 iter_args(%k_off = %c0_i32, %acc = %cst) -> (i32, tensor<16x16xf32, #mma_g>)  : i32 {
      %a = tt.descriptor_gather %a_desc[%a_indices, %k_off] : (!tt.tensordesc<1x32xf16, #padded_ga>, tensor<16xi32, #ttg.slice<{dim = 0, parent = #idx_enc}>>, i32) -> tensor<16x32xf16, #blocked_ga>
      %b = tt.descriptor_gather %b_desc[%b_indices, %c0_i32] : (!tt.tensordesc<1x16xf16, #padded_gb>, tensor<32xi32, #ttg.slice<{dim = 0, parent = #idx_enc}>>, i32) -> tensor<32x16xf16, #blocked_gb>
      %a_dot = ttg.convert_layout %a : tensor<16x32xf16, #blocked_ga> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_g, kWidth = 8}>>
      %b_dot = ttg.convert_layout %b : tensor<32x16xf16, #blocked_gb> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_g, kWidth = 8}>>
      %d = tt.dot %a_dot, %b_dot, %acc : tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma_g, kWidth = 8}>> * tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma_g, kWidth = 8}>> -> tensor<16x16xf32, #mma_g>
      %next_k = arith.addi %k_off, %c32_i32 : i32
      scf.yield %next_k, %d : i32, tensor<16x16xf32, #mma_g>
    }
    %out = arith.truncf %accumulator#1 : tensor<16x16xf32, #mma_g> to tensor<16x16xf16, #mma_g>
    %out_blocked = ttg.convert_layout %out : tensor<16x16xf16, #mma_g> -> tensor<16x16xf16, #blocked_gb>
    tt.descriptor_store %c_desc[%c0_i32, %c0_i32], %out_blocked : !tt.tensordesc<16x16xf16, #padded_gc>, tensor<16x16xf16, #blocked_gb>
    tt.return
  }
}

// CHECK-LABEL: tt.func @gather_dot_pipeline
// Prologue: two async_tdm_gather ops before the loop
// CHECK: amdg.async_tdm_gather
// CHECK-NOT: ttg.async_commit_group
// CHECK: amdg.async_tdm_gather
// CHECK-NOT: ttg.async_commit_group
// Loop body: two more async_tdm_gather ops (pipelined next iteration)
// CHECK: scf.for
// CHECK: amdg.async_tdm_gather
// CHECK-NOT: ttg.async_commit_group
// CHECK: amdg.async_tdm_gather
// CHECK-NOT: ttg.async_commit_group
// CHECK: }

// -----

// A/B descriptor loads are converted to TDM copies, but a descriptor load used
// as the dot accumulator operand remains a raw tt.descriptor_load. Dynamic loop
// predication must guard that raw load directly.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0], [4, 0]]}, instrShape = [16, 16, 4]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_load_accumulator_predicated(
      %a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %acc_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %K: i32 {tt.divisibility = 16 : i32}) {
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %c15_i32 = arith.constant 15 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #mma>
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %m = arith.muli %pid_m, %c256_i32 : i32
    %n = arith.muli %pid_n, %c64_i32 : i32
    %k_stride = arith.extsi %K : i32 to i64
    %a_desc = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%k_stride, %c1_i64] : <f32>, <256x16xf32>
    %n_stride = arith.extsi %N : i32 to i64
    %b_desc = tt.make_tensor_descriptor %b_ptr, [%K, %N], [%n_stride, %c1_i64] : <f32>, <16x64xf32>
    %acc_desc = tt.make_tensor_descriptor %acc_ptr, [%M, %N], [%n_stride, %c1_i64] : <f32>, <256x64xf32>
    %c_desc = tt.make_tensor_descriptor %c_ptr, [%M, %N], [%n_stride, %c1_i64] : <f32>, <256x64xf32>
    %k_plus = arith.addi %K, %c15_i32 : i32
    %num_k = arith.divsi %k_plus, %c16_i32 : i32
    %accumulator:2 = scf.for %iv = %c0_i32 to %num_k step %c1_i32 iter_args(%k_off = %c0_i32, %acc = %zero) -> (i32, tensor<256x64xf32, #mma>)  : i32 {
      %a = tt.descriptor_load %a_desc[%m, %k_off] : !tt.tensordesc<256x16xf32> -> tensor<256x16xf32, #blocked>
      %b = tt.descriptor_load %b_desc[%k_off, %n] : !tt.tensordesc<16x64xf32> -> tensor<16x64xf32, #blocked1>
      %acc_load = tt.descriptor_load %acc_desc[%m, %n] : !tt.tensordesc<256x64xf32> -> tensor<256x64xf32, #mma>
      %a_dot = ttg.convert_layout %a : tensor<256x16xf32, #blocked> -> tensor<256x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_dot = ttg.convert_layout %b : tensor<16x64xf32, #blocked1> -> tensor<16x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %d = tt.dot %a_dot, %b_dot, %acc_load, inputPrecision = tf32 : tensor<256x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x64xf32, #mma>
      %next_k = arith.addi %k_off, %c16_i32 : i32
      scf.yield %next_k, %d : i32, tensor<256x64xf32, #mma>
    }
    %out = ttg.convert_layout %accumulator#1 : tensor<256x64xf32, #mma> -> tensor<256x64xf32, #blocked1>
    tt.descriptor_store %c_desc[%m, %n], %out : !tt.tensordesc<256x64xf32>, tensor<256x64xf32, #blocked1>
    tt.return
  }
}

// CHECK-LABEL: tt.func @descriptor_load_accumulator_predicated
// CHECK: scf.if {{.*}} -> (tensor<256x64xf32
// CHECK-NEXT: tt.descriptor_load
// CHECK-NEXT: tt.dot
// CHECK-NEXT: scf.yield
// CHECK-NEXT: } else {
// CHECK-NEXT: scf.yield

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [1, 1, 1, 32], warpsPerCTA = [1, 2, 4, 1], order = [3, 2, 1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked11 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @descriptor_store_predicate(%arg0: !tt.tensordesc<32x16xbf16>, %arg1: !tt.tensordesc<1x16x4x32xbf16>, %arg2: tensor<16x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked8}>>, %ub: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #blocked8>
    %0 = scf.for %iv = %c0_i32 to %ub step %c32_i32 iter_args(%idx = %c0_i32) -> (i32) : i32 {
      %load = tt.descriptor_load %arg0[%idx, %c0_i32] : !tt.tensordesc<32x16xbf16> -> tensor<32x16xbf16, #blocked3>
      %lhs = ttg.convert_layout %load : tensor<32x16xbf16, #blocked3> -> tensor<32x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked8}>>
      %dot = tt.dot %lhs, %arg2, %cst : tensor<32x16xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked8}>> * tensor<16x64xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked8}>> -> tensor<32x64xf32, #blocked8>
      %dot_blocked = ttg.convert_layout %dot : tensor<32x64xf32, #blocked8> -> tensor<32x64xf32, #blocked1>
      %trans = tt.trans %dot_blocked {order = array<i32: 1, 0>} : tensor<32x64xf32, #blocked1> -> tensor<64x32xf32, #blocked11>
      %store_layout = ttg.convert_layout %trans : tensor<64x32xf32, #blocked11> -> tensor<64x32xf32, #blocked5>
      %reshaped = tt.reshape %store_layout : tensor<64x32xf32, #blocked5> -> tensor<1x16x4x32xf32, #blocked6>
      %out = arith.truncf %reshaped : tensor<1x16x4x32xf32, #blocked6> to tensor<1x16x4x32xbf16, #blocked6>
      tt.descriptor_store %arg1[%c0_i32, %c0_i32, %c0_i32, %idx], %out : !tt.tensordesc<1x16x4x32xbf16>, tensor<1x16x4x32xbf16, #blocked6>
      %next = arith.addi %idx, %c32_i32 : i32
      scf.yield %next : i32
    }
    tt.return
  }
}

// CHECK-LABEL: tt.func @descriptor_store_predicate
// CHECK-NOT: ttg.mask
// CHECK: scf.if %{{[0-9]+}} {
// CHECK-NEXT: tt.descriptor_store
// CHECK-NEXT: }
