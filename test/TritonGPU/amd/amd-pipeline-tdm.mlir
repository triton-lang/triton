// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-descriptor-encoding  -tritonamdgpu-schedule-loops="num_stages=2" -tritonamdgpu-pipeline="use_async_copy=1" -canonicalize | FileCheck %s

// Re-run with num_stages=3 to exercise the LDS-prefetch schedule (only enabled
// on the TDM path when numStages >= 3).
// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-descriptor-encoding  -tritonamdgpu-schedule-loops="num_stages=3" -tritonamdgpu-pipeline="use_async_copy=1" -canonicalize | FileCheck %s --check-prefix=LDS_PREFETCH

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

// With num_stages=3 the TDM path uses the LDS-prefetch schedule: ttg.local_load
// is hoisted one stage ahead of the tt.dot so the dot consumes operands that
// were prefetched into registers the previous iteration. The loop body leads
// with the dot, followed by the next-tile TDM copies, the async_wait, and the
// local_loads that feed the next iteration (carried as iter_args). The copies
// are issued before the wait so they are already in flight across the stall.
// LDS_PREFETCH-LABEL: tt.func @matmul_kernel_make_tensor_descriptor
// Prologue: dot operands for the first iteration are pre-loaded from LDS.
// LDS_PREFETCH: %[[PRO_A:.+]] = ttg.local_load {{.*}} -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = {{.*}}, kWidth = 8}>>
// LDS_PREFETCH: %[[PRO_B:.+]] = ttg.local_load {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = {{.*}}, kWidth = 8}>>
// LDS_PREFETCH: scf.for
// LDS_PREFETCH-SAME: %[[OP_A:[^ ]+]] = %[[PRO_A]]
// LDS_PREFETCH-SAME: %[[OP_B:[^ ]+]] = %[[PRO_B]]
// LDS_PREFETCH: tt.dot %[[OP_A]], %[[OP_B]]
// LDS_PREFETCH: amdg.async_tdm_copy_global_to_local {{.*}} -> !ttg.memdesc<512x32xf16
// LDS_PREFETCH: amdg.async_tdm_copy_global_to_local {{.*}} -> !ttg.memdesc<32x64xf16
// LDS_PREFETCH: amdg.async_tdm_wait {{.*}} {num = 0 : i32}
// LDS_PREFETCH: %[[NXT_A:.+]] = ttg.local_load {{.*}} -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = {{.*}}, kWidth = 8}>>
// LDS_PREFETCH: %[[NXT_B:.+]] = ttg.local_load {{.*}} -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = {{.*}}, kWidth = 8}>>
// LDS_PREFETCH: scf.yield {{.*}}, %[[NXT_A]], %[[NXT_B]]

// -----

// Test TDM padding for fp8 (f8E5M2) matmul on CDNA5.
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

// The LDS-prefetch schedule applies to any non-scaled dot on the TDM path, so
// fp8 operands feeding a plain tt.dot get the prefetch schedule too.
// LDS_PREFETCH-LABEL: tt.func @tdm_padding_fp8
// LDS_PREFETCH: %[[PRO_A:.+]] = ttg.local_load {{.*}} -> tensor<256x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = {{.*}}, kWidth = 8}>>
// LDS_PREFETCH: %[[PRO_B:.+]] = ttg.local_load {{.*}} -> tensor<64x64xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = {{.*}}, kWidth = 8}>>
// LDS_PREFETCH: scf.for
// LDS_PREFETCH-SAME: %[[OP_A:[^ ]+]] = %[[PRO_A]]
// LDS_PREFETCH-SAME: %[[OP_B:[^ ]+]] = %[[PRO_B]]
// LDS_PREFETCH: tt.dot %[[OP_A]], %[[OP_B]]
// LDS_PREFETCH: amdg.async_tdm_copy_global_to_local {{.*}} -> !ttg.memdesc<256x64xf8E5M2
// LDS_PREFETCH: amdg.async_tdm_copy_global_to_local {{.*}} -> !ttg.memdesc<64x64xf8E5M2
// LDS_PREFETCH: amdg.async_tdm_wait {{.*}} {num = 0 : i32}
// LDS_PREFETCH: %[[NXT_A:.+]] = ttg.local_load {{.*}} -> tensor<256x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = {{.*}}, kWidth = 8}>>
// LDS_PREFETCH: %[[NXT_B:.+]] = ttg.local_load {{.*}} -> tensor<64x64xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = {{.*}}, kWidth = 8}>>
// LDS_PREFETCH: scf.yield {{.*}}, %[[NXT_A]], %[[NXT_B]]

// -----

// Test TDM padding for f32 matmul on CDNA5.
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

// Test TDM pipeline for gather + dot on CDNA5.
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

// -----

// Two descriptor stores in the same loop iteration with identical source
// shape/type share a single LDS allocation. The pipeliner must serialize
// them so that the second `local_store` does not overwrite the buffer while
// the first TDM store is still consuming it.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tdm_two_same_shape_stores_share_alloc(
      %x_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %y_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %src: tensor<32x64xf16, #blocked>,
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %ub: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %n_stride = arith.extsi %N : i32 to i64
    %x_desc = tt.make_tensor_descriptor %x_ptr, [%M, %N], [%n_stride, %c1_i64] : <f16>, <32x64xf16>
    %y_desc = tt.make_tensor_descriptor %y_ptr, [%M, %N], [%n_stride, %c1_i64] : <f16>, <32x64xf16>
    scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%off = %c0_i32) -> (i32) : i32 {
      tt.descriptor_store %x_desc[%off, %c0_i32], %src : !tt.tensordesc<32x64xf16>, tensor<32x64xf16, #blocked>
      tt.descriptor_store %y_desc[%off, %c0_i32], %src : !tt.tensordesc<32x64xf16>, tensor<32x64xf16, #blocked>
      %next = arith.addi %off, %c32_i32 : i32
      scf.yield %next : i32
    }
    tt.return
  }
}

// CHECK-LABEL: tt.func @tdm_two_same_shape_stores_share_alloc
// One shared LDS allocation hoisted out of the loop.
// CHECK: %[[ALLOC:.+]] = ttg.local_alloc :
// CHECK-NOT: ttg.local_alloc
// One seed wait before the loop.
// CHECK: %[[SEED:.+]] = amdg.async_tdm_wait  {num = 0 : i32}
// CHECK-NOT: amdg.async_tdm_wait{{.*}}{num = 0
// Loop carries one async token iter_arg.
// CHECK: scf.for {{.*}} iter_args({{.*}}, %[[ITER_TOK:[^ )]+]] = %[[SEED]])
// CHECK-SAME: -> (i32, !ttg.async.token)
// First store: wait on previous iteration's tail token, then write LDS.
// CHECK: amdg.async_tdm_wait %[[ITER_TOK]] {num = 0 : i32}
// CHECK: ttg.local_store {{.*}}, %[[ALLOC]]
// CHECK: %[[TOK1:.+]] = amdg.async_tdm_copy_local_to_global {{.*}} from %[[ALLOC]]
// Second store: wait on first store's token to prevent overwriting LDS while the first TDM store is still consuming it.
// CHECK: amdg.async_tdm_wait %[[TOK1]] {num = 0 : i32}
// CHECK: ttg.local_store {{.*}}, %[[ALLOC]]
// CHECK: %[[TOK2:.+]] = amdg.async_tdm_copy_local_to_global {{.*}} from %[[ALLOC]]
// Yield the second store's token
// CHECK: scf.yield {{.*}}, %[[TOK2]] : i32, !ttg.async.token
// CHECK: }
// Drain the in-flight TDM store and deallocate the shared buffer
// CHECK: amdg.async_tdm_wait %{{[^,]+}} {num = 0 : i32}
// CHECK: ttg.local_dealloc %[[ALLOC]]
// CHECK-NOT: ttg.local_dealloc

// -----

// Two descriptor stores with different source element types have to use separate allocations.
#blocked_mix = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tdm_two_stores_diff_elem_type(
      %x_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %y_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32},
      %src_a: tensor<32x64xf16, #blocked_mix>,
      %src_b: tensor<32x64xbf16, #blocked_mix>,
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %ub: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %n_stride = arith.extsi %N : i32 to i64
    %x_desc = tt.make_tensor_descriptor %x_ptr, [%M, %N], [%n_stride, %c1_i64] : <f16>, <32x64xf16>
    %y_desc = tt.make_tensor_descriptor %y_ptr, [%M, %N], [%n_stride, %c1_i64] : <bf16>, <32x64xbf16>
    scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%off = %c0_i32) -> (i32) : i32 {
      tt.descriptor_store %x_desc[%off, %c0_i32], %src_a : !tt.tensordesc<32x64xf16>, tensor<32x64xf16, #blocked_mix>
      tt.descriptor_store %y_desc[%off, %c0_i32], %src_b : !tt.tensordesc<32x64xbf16>, tensor<32x64xbf16, #blocked_mix>
      %next = arith.addi %off, %c32_i32 : i32
      scf.yield %next : i32
    }
    tt.return
  }
}

// CHECK-LABEL: tt.func @tdm_two_stores_diff_elem_type
// CHECK-DAG: %[[AF16:.+]] = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, {{.+}}, #smem, mutable>
// CHECK-DAG: %[[ABF16:.+]] = ttg.local_alloc : () -> !ttg.memdesc<32x64xbf16, {{.+}}, #smem, mutable>
// CHECK-NOT: ttg.local_alloc
// CHECK: scf.for
// CHECK-SAME: -> (i32, !ttg.async.token, !ttg.async.token)
// CHECK: ttg.local_store {{.*}}, %[[AF16]]
// CHECK: amdg.async_tdm_copy_local_to_global {{.*}} from %[[AF16]]
// CHECK: ttg.local_store {{.*}}, %[[ABF16]]
// CHECK: amdg.async_tdm_copy_local_to_global {{.*}} from %[[ABF16]]
// CHECK: ttg.local_dealloc %[[AF16]]
// CHECK: ttg.local_dealloc %[[ABF16]]

// -----

// Same element type but different tile size
#blocked_mix2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tdm_two_stores_diff_tensor_shape(
      %x_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %y_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %src_a: tensor<32x64xf16, #blocked_mix2>,
      %src_b: tensor<32x32xf16, #blocked_mix2>,
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %ub: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c1_i64 = arith.constant 1 : i64
    %n_stride = arith.extsi %N : i32 to i64
    %x_desc = tt.make_tensor_descriptor %x_ptr, [%M, %N], [%n_stride, %c1_i64] : <f16>, <32x64xf16>
    %y_desc = tt.make_tensor_descriptor %y_ptr, [%M, %c32_i32], [%n_stride, %c1_i64] : <f16>, <32x32xf16>
    scf.for %iv = %c0_i32 to %ub step %c1_i32 iter_args(%off = %c0_i32) -> (i32) : i32 {
      tt.descriptor_store %x_desc[%off, %c0_i32], %src_a : !tt.tensordesc<32x64xf16>, tensor<32x64xf16, #blocked_mix2>
      tt.descriptor_store %y_desc[%off, %c0_i32], %src_b : !tt.tensordesc<32x32xf16>, tensor<32x32xf16, #blocked_mix2>
      %next = arith.addi %off, %c32_i32 : i32
      scf.yield %next : i32
    }
    tt.return
  }
}

// CHECK-LABEL: tt.func @tdm_two_stores_diff_tensor_shape
// CHECK-DAG: %[[A64:.+]] = ttg.local_alloc : () -> !ttg.memdesc<32x64xf16, {{.+}}, #smem, mutable>
// CHECK-DAG: %[[A32:.+]] = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, {{.+}}, #smem, mutable>
// CHECK-NOT: ttg.local_alloc
// CHECK: scf.for
// CHECK-SAME: -> (i32, !ttg.async.token, !ttg.async.token)
// CHECK: ttg.local_store {{.*}}, %[[A64]]
// CHECK: amdg.async_tdm_copy_local_to_global {{.*}} from %[[A64]]
// CHECK: ttg.local_store {{.*}}, %[[A32]]
// CHECK: amdg.async_tdm_copy_local_to_global {{.*}} from %[[A32]]
// CHECK: ttg.local_dealloc %[[A64]]
// CHECK: ttg.local_dealloc %[[A32]]

// -----

// TDM load + tt.dot_scaled (mxfp4 x mxfp4) pipeline on CDNA5.
#blocked_sd   = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked_sd_b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma_sd = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0], [4, 0]]}, instrShape = [16, 16, 128]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tdm_scaled_dot_reorder(
      %a_ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %b_ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32},
      %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %scale_a: tensor<128x4xi8, #blocked_sd>,
      %scale_b: tensor<128x4xi8, #blocked_sd>,
      %M: i32 {tt.divisibility = 16 : i32},
      %N: i32 {tt.divisibility = 16 : i32},
      %K: i32 {tt.divisibility = 16 : i32}) {
    %c128_i32 = arith.constant 128 : i32
    %c64_i32  = arith.constant 64  : i32
    %c0_i32   = arith.constant 0   : i32
    %c1_i64   = arith.constant 1   : i64
    %c1_i32   = arith.constant 1   : i32
    %c63_i32  = arith.constant 63  : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma_sd>
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %m = arith.muli %pid_m, %c128_i32 : i32
    %n = arith.muli %pid_n, %c128_i32 : i32
    %k_stride = arith.extsi %K : i32 to i64
    %a_desc = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%k_stride, %c1_i64] : <i8>, <128x64xi8>
    %n_stride = arith.extsi %N : i32 to i64
    %b_desc = tt.make_tensor_descriptor %b_ptr, [%K, %N], [%n_stride, %c1_i64] : <i8>, <64x128xi8>
    %k_ceil = arith.addi %K, %c63_i32 : i32
    %num_k   = arith.divsi %k_ceil, %c64_i32 : i32
    %result:2 = scf.for %iv = %c0_i32 to %num_k step %c1_i32
        iter_args(%k_off = %c0_i32, %acc = %cst) -> (i32, tensor<128x128xf32, #mma_sd>) : i32 {
      %a = tt.descriptor_load %a_desc[%m, %k_off] : !tt.tensordesc<128x64xi8> -> tensor<128x64xi8, #blocked_sd>
      %b = tt.descriptor_load %b_desc[%k_off, %n] : !tt.tensordesc<64x128xi8>  -> tensor<64x128xi8, #blocked_sd_b>
      %a_dot = ttg.convert_layout %a : tensor<128x64xi8, #blocked_sd>   -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma_sd, kWidth = 16}>>
      %b_dot = ttg.convert_layout %b : tensor<64x128xi8, #blocked_sd_b> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma_sd, kWidth = 16}>>
      %d = tt.dot_scaled %a_dot scale %scale_a, %b_dot scale %scale_b, %acc lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma_sd, kWidth = 16}>>, tensor<128x4xi8, #blocked_sd> * tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma_sd, kWidth = 16}>>, tensor<128x4xi8, #blocked_sd> -> tensor<128x128xf32, #mma_sd>
      %next_k = arith.addi %k_off, %c64_i32 : i32
      scf.yield %next_k, %d : i32, tensor<128x128xf32, #mma_sd>
    }
    tt.return
  }
}

// TDM copy reordering is on by default: all TDM copies precede the
// single async_tdm_wait, which precedes the local_loads and dot_scaled.
// CHECK-LABEL: tt.func @tdm_scaled_dot_reorder
// CHECK: scf.for
// CHECK: async_tdm_copy_global_to_local {{.*}} -> !ttg.memdesc<128x64xi8
// CHECK: async_tdm_copy_global_to_local {{.*}} -> !ttg.memdesc<64x128xi8
// CHECK: amdg.async_tdm_wait %{{[^,]+}}, %{{[^,]+}} {num = 0 : i32}
// CHECK-NOT: async_tdm_copy_global_to_local
// CHECK: ttg.local_load
// CHECK: ttg.local_load
// CHECK: tt.dot_scaled
// -----

// An M-sliced GEMM: one dot per A tile, both sharing a loop-invariant B. The
// two 512x32 loads tile a 1024x32 parent, so they share one TDM copy. Only the
// leading slice is prefetched across the backedge; the second is read from a
// subslice in the iteration that uses it, with the fused wait between that
// load and d1. Contrast with the single-dot kernels above, where both operands
// stay prefetched as loop-carried registers.
// LDS_PREFETCH-LABEL: tt.func @sliced_m
// LDS_PREFETCH: ttg.local_alloc : () -> !ttg.memdesc<2x1024x32xf16
// LDS_PREFETCH: scf.for
// LDS_PREFETCH: tt.dot %arg{{[0-9]+}}, %arg1
// LDS_PREFETCH: ttg.memdesc_subslice {{.*}}[512, 0]
// LDS_PREFETCH: %[[TILE:.*]] = ttg.local_load
// LDS_PREFETCH: amdg.async_tdm_wait
// LDS_PREFETCH: tt.dot %[[TILE]], %arg1

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0], [4, 0]]}, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @sliced_m(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %B: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
    %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32})
    -> (tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>) {
    %c512_i32 = arith.constant 512 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<512x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = arith.addi %1, %c512_i32 : i32
    %3 = arith.extsi %K : i32 to i64
    %4 = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%3, %c1_i64] : <f16>, <512x32xf16>
    %5 = arith.divsi %K, %c32_i32 : i32
    %acc:3 = scf.for %iv = %c0_i32 to %5 step %c1_i32
        iter_args(%k = %c0_i32, %acc0 = %cst, %acc1 = %cst)
        -> (i32, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>) : i32 {
      %6 = tt.descriptor_load %4[%1, %k] : !tt.tensordesc<512x32xf16> -> tensor<512x32xf16, #blocked>
      %7 = tt.descriptor_load %4[%2, %k] : !tt.tensordesc<512x32xf16> -> tensor<512x32xf16, #blocked>
      %8 = ttg.convert_layout %6 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %9 = ttg.convert_layout %7 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %10 = tt.dot %8, %B, %acc0 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
      %11 = tt.dot %9, %B, %acc1 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
      %12 = arith.addi %k, %c32_i32 : i32
      scf.yield %12, %10, %11 : i32, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>
    }
    tt.return %acc#1, %acc#2 : tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>
  }
}

// -----

// Three M-slices would tile a 1536-row parent, which is not a power of two, so
// each slice keeps its own TDM copy and buffer.
// CHECK-LABEL: tt.func @sliced_m_three_slices
// CHECK-COUNT-3: ttg.local_alloc : () -> !ttg.memdesc<2x512x32xf16
// CHECK-NOT: ttg.memdesc_subslice
// CHECK: tt.return
// LDS_PREFETCH-LABEL: tt.func @sliced_m_three_slices
// LDS_PREFETCH-COUNT-3: ttg.local_alloc : () -> !ttg.memdesc<2x512x32xf16
// LDS_PREFETCH-NOT: ttg.memdesc_subslice
// LDS_PREFETCH: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0], [4, 0]]}, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @sliced_m_three_slices(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %B: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
    %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32})
    -> (tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>) {
    %c512_i32 = arith.constant 512 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c1536_i32 = arith.constant 1536 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<512x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1536_i32 : i32
    %2 = arith.addi %1, %c512_i32 : i32
    %3 = arith.addi %1, %c1024_i32 : i32
    %4 = arith.extsi %K : i32 to i64
    %5 = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%4, %c1_i64] : <f16>, <512x32xf16>
    %6 = arith.divsi %K, %c32_i32 : i32
    %acc:4 = scf.for %iv = %c0_i32 to %6 step %c1_i32
        iter_args(%k = %c0_i32, %acc0 = %cst, %acc1 = %cst, %acc2 = %cst)
        -> (i32, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>) : i32 {
      %7 = tt.descriptor_load %5[%1, %k] : !tt.tensordesc<512x32xf16> -> tensor<512x32xf16, #blocked>
      %8 = tt.descriptor_load %5[%2, %k] : !tt.tensordesc<512x32xf16> -> tensor<512x32xf16, #blocked>
      %9 = tt.descriptor_load %5[%3, %k] : !tt.tensordesc<512x32xf16> -> tensor<512x32xf16, #blocked>
      %10 = ttg.convert_layout %7 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %11 = ttg.convert_layout %8 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %12 = ttg.convert_layout %9 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %13 = tt.dot %10, %B, %acc0 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
      %14 = tt.dot %11, %B, %acc1 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
      %15 = tt.dot %12, %B, %acc2 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
      %16 = arith.addi %k, %c32_i32 : i32
      scf.yield %16, %13, %14, %15 : i32, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>
    }
    tt.return %acc#1, %acc#2, %acc#3 : tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>
  }
}

// -----

// Rank-reducing loads drop the unit batch dimension of a 1x512x32 block, but
// their indices still address all three descriptor dimensions. Two loads from
// batches 512 apart must not be taken for adjacent 512-row M-slices.
// CHECK-LABEL: tt.func @rank_reducing_loads_not_fused
// CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2x512x32xf16
// CHECK-NOT: ttg.memdesc_subslice
// CHECK: tt.return
// LDS_PREFETCH-LABEL: tt.func @rank_reducing_loads_not_fused
// LDS_PREFETCH-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2x512x32xf16
// LDS_PREFETCH-NOT: ttg.memdesc_subslice
// LDS_PREFETCH: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0], [4, 0]]}, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @rank_reducing_loads_not_fused(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %B: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
    %NB: i32 {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32})
    -> (tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>) {
    %c512_i32 = arith.constant 512 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<512x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.get_program_id y : i32
    %3 = arith.addi %2, %c512_i32 : i32
    %4 = arith.extsi %K : i32 to i64
    %5 = arith.extsi %M : i32 to i64
    %6 = arith.muli %5, %4 : i64
    %7 = tt.make_tensor_descriptor %a_ptr, [%NB, %M, %K], [%6, %4, %c1_i64] : <f16>, <1x512x32xf16>
    %8 = arith.divsi %K, %c32_i32 : i32
    %acc:3 = scf.for %iv = %c0_i32 to %8 step %c1_i32
        iter_args(%k = %c0_i32, %acc0 = %cst, %acc1 = %cst)
        -> (i32, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>) : i32 {
      %9 = tt.descriptor_load %7[%2, %1, %k] : !tt.tensordesc<1x512x32xf16> -> tensor<512x32xf16, #blocked>
      %10 = tt.descriptor_load %7[%3, %1, %k] : !tt.tensordesc<1x512x32xf16> -> tensor<512x32xf16, #blocked>
      %11 = ttg.convert_layout %9 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %12 = ttg.convert_layout %10 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %13 = tt.dot %11, %B, %acc0 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
      %14 = tt.dot %12, %B, %acc1 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
      %15 = arith.addi %k, %c32_i32 : i32
      scf.yield %15, %13, %14 : i32, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>
    }
    tt.return %acc#1, %acc#2 : tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>
  }
}

// -----

// Each 512x32 M-slice is split across two CTAs. A 1024x32 parent with the
// same CGA layout would hold all of A0 in CTA0 and all of A1 in CTA1, so each
// slice keeps its own TDM copy and buffer.
// CHECK-LABEL: tt.func @cta_split_slices_not_fused
// CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2x512x32xf16
// CHECK-NOT: ttg.memdesc_subslice
// CHECK: tt.return
// LDS_PREFETCH-LABEL: tt.func @cta_split_slices_not_fused
// LDS_PREFETCH-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2x512x32xf16
// LDS_PREFETCH-NOT: ttg.memdesc_subslice
// LDS_PREFETCH: tt.return

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0], [4, 0]]}, CGALayout = [[1, 0]], instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @cta_split_slices_not_fused(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %B: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
    %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32})
    -> (tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>) {
    %c512_i32 = arith.constant 512 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<512x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = arith.addi %1, %c512_i32 : i32
    %3 = arith.extsi %K : i32 to i64
    %4 = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%3, %c1_i64] : <f16>, <512x32xf16>
    %5 = arith.divsi %K, %c32_i32 : i32
    %acc:3 = scf.for %iv = %c0_i32 to %5 step %c1_i32
        iter_args(%k = %c0_i32, %acc0 = %cst, %acc1 = %cst)
        -> (i32, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>) : i32 {
      %6 = tt.descriptor_load %4[%1, %k] : !tt.tensordesc<512x32xf16> -> tensor<512x32xf16, #blocked>
      %7 = tt.descriptor_load %4[%2, %k] : !tt.tensordesc<512x32xf16> -> tensor<512x32xf16, #blocked>
      %8 = ttg.convert_layout %6 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %9 = ttg.convert_layout %7 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %10 = tt.dot %8, %B, %acc0 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
      %11 = tt.dot %9, %B, %acc1 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
      %12 = arith.addi %k, %c32_i32 : i32
      scf.yield %12, %10, %11 : i32, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>
    }
    tt.return %acc#1, %acc#2 : tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>
  }
}

// -----

// A 2x2 M/N-sliced GEMM with the dots in diagonal order: d0 = A0*B0,
// d1 = A1*B1, d2 = A0*B1, d3 = A1*B0. A0 and A1 share one TDM copy, while B0
// and B1 come from separate descriptors and keep their own copies and waits.
// The d0 operands are prefetched in the last two groups and A1 is read in
// group 0, so B1, which d1 consumes, has no free group before d1 and shares
// group 0 with A1: its wait and read come right after d0.
// LDS_PREFETCH-LABEL: tt.func @sliced_diagonal_dots
// LDS_PREFETCH: scf.for
// LDS_PREFETCH: tt.dot
// LDS_PREFETCH-NOT: tt.dot
// LDS_PREFETCH: %[[WAIT:.+]] = amdg.async_tdm_wait %{{[^,]+}} {num = 0 : i32}
// LDS_PREFETCH-NOT: tt.dot
// LDS_PREFETCH: %[[B1:.+]] = ttg.local_load %{{.+}} token %[[WAIT]]
// LDS_PREFETCH-NOT: tt.dot
// LDS_PREFETCH: %[[B1_T:.+]] = tt.trans %[[B1]]
// LDS_PREFETCH: tt.dot %{{.+}}, %[[B1_T]]

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]], warp = [[16, 0], [0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 64]}>
#shared1 = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [128, 64]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @sliced_diagonal_dots(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32})
    -> (tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>) {
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = arith.addi %1, %c128_i32 : i32
    %3 = arith.extsi %K : i32 to i64
    %4 = tt.make_tensor_descriptor %a_ptr, [%M, %K], [%3, %c1_i64] : <f16>, <128x64xf16, #shared>
    %5 = tt.make_tensor_descriptor %b_ptr, [%N, %K], [%3, %c1_i64] : <f16>, <128x64xf16, #shared1>
    %6 = tt.make_tensor_descriptor %b_ptr, [%N, %K], [%3, %c1_i64] : <f16>, <128x64xf16, #shared1>
    %acc:4 = scf.for %k = %c0_i32 to %K step %c64_i32
        iter_args(%acc00 = %cst, %acc11 = %cst, %acc01 = %cst, %acc10 = %cst)
        -> (tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>) : i32 {
      %7 = tt.descriptor_load %4[%1, %k] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %8 = tt.descriptor_load %4[%2, %k] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %9 = tt.descriptor_load %5[%1, %k] : !tt.tensordesc<128x64xf16, #shared1> -> tensor<128x64xf16, #blocked>
      %10 = ttg.convert_layout %9 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #linear>
      %11 = tt.trans %10 {order = array<i32: 1, 0>} : tensor<128x64xf16, #linear> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %12 = tt.descriptor_load %6[%2, %k] : !tt.tensordesc<128x64xf16, #shared1> -> tensor<128x64xf16, #blocked>
      %13 = ttg.convert_layout %12 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #linear>
      %14 = tt.trans %13 {order = array<i32: 1, 0>} : tensor<128x64xf16, #linear> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %15 = ttg.convert_layout %7 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %16 = ttg.convert_layout %8 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %17 = tt.dot %15, %11, %acc00 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %18 = tt.dot %16, %14, %acc11 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %19 = tt.dot %15, %14, %acc01 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %20 = tt.dot %16, %11, %acc10 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      scf.yield %17, %18, %19, %20 : tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>
    }
    tt.return %acc#0, %acc#1, %acc#2, %acc#3 : tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>
  }
}

// -----

// Four dots share a loop-invariant B: d0 = X0*B, d1 = X1*B, d2 = Y0*B and
// d3 = Y1*B, where X0/X1 and Y0/Y1 each tile one fused 1024x32 parent. Y is
// read only at the current K, in the groups before d2 and d3, so its wait
// comes before Y0's read, right after d1.
// LDS_PREFETCH-LABEL: tt.func @sliced_m_two_operands
// LDS_PREFETCH-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2x1024x32xf16
// LDS_PREFETCH: scf.for
// LDS_PREFETCH: tt.dot
// LDS_PREFETCH: tt.dot
// LDS_PREFETCH-NOT: tt.dot
// LDS_PREFETCH: %[[WAIT:.+]] = amdg.async_tdm_wait
// LDS_PREFETCH-NOT: tt.dot
// LDS_PREFETCH: %[[Y0:.+]] = ttg.local_load %{{.+}} token %[[WAIT]]
// LDS_PREFETCH: tt.dot %[[Y0]], %arg2

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0], [4, 0]]}, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @sliced_m_two_operands(%x_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %y_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %B: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>,
    %M: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32})
    -> (tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>) {
    %c512_i32 = arith.constant 512 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i32 = arith.constant 32 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<512x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = arith.addi %1, %c512_i32 : i32
    %3 = arith.extsi %K : i32 to i64
    %4 = tt.make_tensor_descriptor %x_ptr, [%M, %K], [%3, %c1_i64] : <f16>, <512x32xf16>
    %5 = tt.make_tensor_descriptor %y_ptr, [%M, %K], [%3, %c1_i64] : <f16>, <512x32xf16>
    %6 = arith.divsi %K, %c32_i32 : i32
    %acc:5 = scf.for %iv = %c0_i32 to %6 step %c1_i32
        iter_args(%k = %c0_i32, %acc0 = %cst, %acc1 = %cst, %acc2 = %cst, %acc3 = %cst)
        -> (i32, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>) : i32 {
      %7 = tt.descriptor_load %4[%1, %k] : !tt.tensordesc<512x32xf16> -> tensor<512x32xf16, #blocked>
      %8 = tt.descriptor_load %4[%2, %k] : !tt.tensordesc<512x32xf16> -> tensor<512x32xf16, #blocked>
      %9 = tt.descriptor_load %5[%1, %k] : !tt.tensordesc<512x32xf16> -> tensor<512x32xf16, #blocked>
      %10 = tt.descriptor_load %5[%2, %k] : !tt.tensordesc<512x32xf16> -> tensor<512x32xf16, #blocked>
      %11 = ttg.convert_layout %7 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %12 = ttg.convert_layout %8 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %13 = ttg.convert_layout %9 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %14 = ttg.convert_layout %10 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %15 = tt.dot %11, %B, %acc0 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
      %16 = tt.dot %12, %B, %acc1 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
      %17 = tt.dot %13, %B, %acc2 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
      %18 = tt.dot %14, %B, %acc3 : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
      %19 = arith.addi %k, %c32_i32 : i32
      scf.yield %19, %15, %16, %17, %18 : i32, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>
    }
    tt.return %acc#1, %acc#2, %acc#3, %acc#4 : tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>, tensor<512x64xf32, #mma>
  }
}

// -----

// A 2x2 M/N-sliced GEMM: four A and four B quarter loads feed eight dots, two
// per output quadrant chained through the accumulator. Those chains do not
// block interleaving, so each quarter is read from LDS next to the dot that
// first uses it. Every dot keeps its program-order place, including the one
// whose operands were both first read by earlier dots. The quarter loads of
// each operand tile one 256x128 tile, so they share one whole-tile TDM copy
// per operand, padded once per 128-element row of the whole tile.
// LDS_PREFETCH: [[SHARED:#[a-z0-9]+]] = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [256, 128]}>
// LDS_PREFETCH: tt.func public @gemm_tdm_sliceMN_k2_kernel
// LDS_PREFETCH-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2x256x128xf16, [[SHARED]]
// LDS_PREFETCH: scf.for
// LDS_PREFETCH: %[[A_BL:.+]] = ttg.local_load
// LDS_PREFETCH: tt.dot %[[A_BL]]
// LDS_PREFETCH: %[[B_TR_LOAD:.+]] = ttg.local_load
// LDS_PREFETCH: %[[B_TR:.+]] = tt.trans %[[B_TR_LOAD]]
// LDS_PREFETCH: tt.dot %{{.+}}, %[[B_TR]]
// LDS_PREFETCH-NOT: tt.dot
// LDS_PREFETCH: tt.dot %[[A_BL]], %[[B_TR]]
// LDS_PREFETCH: amdg.async_tdm_copy_global_to_local {{.*}} -> !ttg.memdesc<256x128xf16
// LDS_PREFETCH: amdg.async_tdm_copy_global_to_local {{.*}} -> !ttg.memdesc<256x128xf16
// LDS_PREFETCH: scf.yield
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 8]], warp = [[16, 0], [0, 0]], block = []}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 64]}>
#shared1 = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [128, 64]}>
#shared2 = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [128, 128]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gemm_tdm_sliceMN_k2_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c8_i32 = arith.constant 8 : i32
    %c256_i32 = arith.constant 256 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c255_i32 = arith.constant 255 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c255_i32 : i32
    %2 = arith.divsi %1, %c256_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %5 : i32
    %11 = arith.remsi %10, %9 : i32
    %12 = arith.addi %7, %11 : i32
    %13 = arith.divsi %10, %9 : i32
    %14 = arith.muli %12, %c256_i32 : i32
    %15 = arith.muli %13, %c256_i32 : i32
    %16 = arith.extsi %arg5 : i32 to i64
    %17 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%16, %c1_i64] : <f16>, <128x64xf16, #shared>
    %18 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%16, %c1_i64] : <f16>, <128x64xf16, #shared1>
    %19 = arith.extsi %arg4 : i32 to i64
    %20 = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%19, %c1_i64] : <f16>, <128x128xf16, #shared2>
    %21 = arith.addi %14, %c128_i32 : i32
    %22 = arith.addi %15, %c128_i32 : i32
    %23:4 = scf.for %arg6 = %c0_i32 to %arg5 step %c128_i32 iter_args(%arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst) -> (tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>)  : i32 {
      %34 = tt.descriptor_load %17[%14, %arg6] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %35 = arith.addi %arg6, %c64_i32 : i32
      %36 = tt.descriptor_load %17[%14, %35] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %37 = tt.descriptor_load %17[%21, %arg6] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %38 = tt.descriptor_load %17[%21, %35] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
      %39 = tt.descriptor_load %18[%15, %arg6] : !tt.tensordesc<128x64xf16, #shared1> -> tensor<128x64xf16, #blocked>
      %40 = ttg.convert_layout %39 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #linear>
      %41 = tt.trans %40 {order = array<i32: 1, 0>} : tensor<128x64xf16, #linear> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %42 = tt.descriptor_load %18[%22, %arg6] : !tt.tensordesc<128x64xf16, #shared1> -> tensor<128x64xf16, #blocked>
      %43 = ttg.convert_layout %42 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #linear>
      %44 = tt.trans %43 {order = array<i32: 1, 0>} : tensor<128x64xf16, #linear> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %45 = tt.descriptor_load %18[%15, %35] : !tt.tensordesc<128x64xf16, #shared1> -> tensor<128x64xf16, #blocked>
      %46 = ttg.convert_layout %45 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #linear>
      %47 = tt.trans %46 {order = array<i32: 1, 0>} : tensor<128x64xf16, #linear> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %48 = tt.descriptor_load %18[%22, %35] : !tt.tensordesc<128x64xf16, #shared1> -> tensor<128x64xf16, #blocked>
      %49 = ttg.convert_layout %48 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #linear>
      %50 = tt.trans %49 {order = array<i32: 1, 0>} : tensor<128x64xf16, #linear> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %51 = ttg.convert_layout %34 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %52 = tt.dot %51, %41, %arg7 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %53 = ttg.convert_layout %37 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %54 = tt.dot %53, %41, %arg8 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %55 = tt.dot %51, %44, %arg9 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %56 = tt.dot %53, %44, %arg10 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %57 = ttg.convert_layout %36 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %58 = tt.dot %57, %47, %52 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %59 = ttg.convert_layout %38 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %60 = tt.dot %59, %47, %54 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %61 = tt.dot %57, %50, %55 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %62 = tt.dot %59, %50, %56 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      scf.yield %58, %60, %61, %62 : tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>, tensor<128x128xf32, #mma>
    }
    %24 = arith.truncf %23#0 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %25 = ttg.convert_layout %24 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked1>
    tt.descriptor_store %20[%14, %15], %25 : !tt.tensordesc<128x128xf16, #shared2>, tensor<128x128xf16, #blocked1>
    %26 = arith.addi %14, %c128_i32 : i32
    %27 = arith.truncf %23#1 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %28 = ttg.convert_layout %27 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked1>
    tt.descriptor_store %20[%26, %15], %28 : !tt.tensordesc<128x128xf16, #shared2>, tensor<128x128xf16, #blocked1>
    %29 = arith.addi %15, %c128_i32 : i32
    %30 = arith.truncf %23#2 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %31 = ttg.convert_layout %30 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked1>
    tt.descriptor_store %20[%14, %29], %31 : !tt.tensordesc<128x128xf16, #shared2>, tensor<128x128xf16, #blocked1>
    %32 = arith.truncf %23#3 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %33 = ttg.convert_layout %32 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked1>
    tt.descriptor_store %20[%26, %29], %33 : !tt.tensordesc<128x128xf16, #shared2>, tensor<128x128xf16, #blocked1>
    tt.return
  }
}
