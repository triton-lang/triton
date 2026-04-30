// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1250" | FileCheck %s

// CHECK-LABEL: subslice_one_cta

// This testing is just to make sure no assertion is triggered in
// MemDescReinterpretOp::verify. No need to inspect resulting IR.

#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#shared = #ttg.padded_shared<[128:+8] {order = [1, 0], shape = [16, 128]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared2 = #ttg.padded_shared<[128:+8] {order = [0, 1], shape = [128, 16]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 17376 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @subslice_one_cta(%a_ptr: !tt.ptr<f16>, %b_ptr: !tt.ptr<f16>, %c_ptr: !tt.ptr<bf16>,
        %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) {
    %c32768_i32 = arith.constant 32768 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1_i64 = arith.constant 1 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c16_i32 = arith.constant 16 : i32
    %c15_i32 = arith.constant 15 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %M, %c15_i32 : i32
    %2 = arith.divsi %1, %c16_i32 : i32
    %3 = arith.remsi %0, %2 : i32
    %4 = arith.divsi %0, %2 : i32
    %5 = arith.muli %3, %c16_i32 : i32
    %6 = arith.muli %3, %c32768_i32 : i32
    %7 = arith.muli %4, %c16_i32 : i32
    %8 = arith.muli %4, %c32768_i32 : i32
    %9 = tt.addptr %a_ptr, %6 : !tt.ptr<f16>, i32
    %10 = tt.make_tensor_descriptor %9, [%M, %K], [%c2048_i64, %c1_i64] : <f16>, <16x128xf16, #shared>
    %11 = tt.addptr %b_ptr, %8 : !tt.ptr<f16>, i32
    %12 = tt.make_tensor_descriptor %11, [%N, %K], [%c2048_i64, %c1_i64] : <f16>, <16x128xf16, #shared>
    %13 = tt.make_tensor_descriptor %c_ptr, [%M, %N], [%c2048_i64, %c1_i64] : <bf16>, <16x16xbf16, #shared1>
    %14 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable>
    %15 = ttg.local_alloc {allocation.offset = 8688 : i32} : () -> !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable>
    %16 = ttg.memdesc_index %14[%c0_i32] : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %17 = amdg.async_tdm_copy_global_to_local %10[%c0_i32, %c0_i32] into %16, pred = %c1_i32 : !tt.tensordesc<16x128xf16, #shared> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %18 = ttg.memdesc_index %15[%c0_i32] : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %19 = amdg.async_tdm_copy_global_to_local %12[%c0_i32, %c0_i32] into %18, pred = %c1_i32 : !tt.tensordesc<16x128xf16, #shared> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %20 = amdg.async_tdm_intrinsic_wait {count = 0 : i32, ttg.num_tdm_ops = 0 : i32}
    %21 = ttg.memdesc_subslice %16[0, 0] : !ttg.memdesc<16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128>
    %22 = ttg.local_load %21 : !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %23 = ttg.memdesc_subslice %18[0, 0] : !ttg.memdesc<16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128>
    %24 = ttg.memdesc_trans %23 {order = array<i32: 1, 0>} : !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128> -> !ttg.memdesc<32x16xf16, #shared2, #smem, mutable, 128x16>
    %25 = ttg.local_load %24 : !ttg.memdesc<32x16xf16, #shared2, #smem, mutable, 128x16> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %26 = ttg.memdesc_index %14[%c1_i32] : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %27 = amdg.async_tdm_copy_global_to_local %10[%c0_i32, %c128_i32] into %26, pred = %c0_i32 : !tt.tensordesc<16x128xf16, #shared> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %28 = ttg.memdesc_index %15[%c1_i32] : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %29 = amdg.async_tdm_copy_global_to_local %12[%c0_i32, %c128_i32] into %28, pred = %c0_i32 : !tt.tensordesc<16x128xf16, #shared> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %30 = ttg.memdesc_subslice %16[0, 32] : !ttg.memdesc<16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128>
    %31 = ttg.local_load %30 : !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %32 = ttg.memdesc_subslice %18[0, 32] : !ttg.memdesc<16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128>
    %33 = ttg.memdesc_trans %32 {order = array<i32: 1, 0>} : !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128> -> !ttg.memdesc<32x16xf16, #shared2, #smem, mutable, 128x16>
    %34 = ttg.local_load %33 : !ttg.memdesc<32x16xf16, #shared2, #smem, mutable, 128x16> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %35 = tt.dot %22, %25, %cst : tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    %36 = ttg.memdesc_subslice %16[0, 64] : !ttg.memdesc<16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128>
    %37 = ttg.local_load %36 : !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %38 = ttg.memdesc_subslice %18[0, 64] : !ttg.memdesc<16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128>
    %39 = ttg.memdesc_trans %38 {order = array<i32: 1, 0>} : !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128> -> !ttg.memdesc<32x16xf16, #shared2, #smem, mutable, 128x16>
    %40 = ttg.local_load %39 : !ttg.memdesc<32x16xf16, #shared2, #smem, mutable, 128x16> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %41 = tt.dot %31, %34, %35 : tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    %42 = ttg.memdesc_subslice %16[0, 96] : !ttg.memdesc<16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128>
    %43 = ttg.local_load %42 : !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %44 = ttg.memdesc_subslice %18[0, 96] : !ttg.memdesc<16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128>
    %45 = ttg.memdesc_trans %44 {order = array<i32: 1, 0>} : !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128> -> !ttg.memdesc<32x16xf16, #shared2, #smem, mutable, 128x16>
    %46 = ttg.local_load %45 : !ttg.memdesc<32x16xf16, #shared2, #smem, mutable, 128x16> -> tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %47 = tt.dot %37, %40, %41 : tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    %48 = amdg.async_tdm_intrinsic_wait {count = 0 : i32, ttg.num_tdm_ops = 0 : i32}
    %49 = amdg.async_tdm_copy_global_to_local %10[%c0_i32, %c256_i32] into %16, pred = %c0_i32 : !tt.tensordesc<16x128xf16, #shared> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %50 = amdg.async_tdm_copy_global_to_local %12[%c0_i32, %c256_i32] into %18, pred = %c0_i32 : !tt.tensordesc<16x128xf16, #shared> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %51 = tt.dot %43, %46, %47 : tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x16xf32, #mma>
    %52 = ttg.memdesc_reinterpret %14 : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x16xbf16, #shared1, #smem, mutable>
    %53 = arith.truncf %51 : tensor<16x16xf32, #mma> to tensor<16x16xbf16, #mma>
    ttg.local_store %53, %52 : tensor<16x16xbf16, #mma> -> !ttg.memdesc<16x16xbf16, #shared1, #smem, mutable>
    amdg.async_tdm_copy_local_to_global %13[%5, %7] from %52 : !ttg.memdesc<16x16xbf16, #shared1, #smem, mutable> -> !tt.tensordesc<16x16xbf16, #shared1>
    %54 = amdg.async_tdm_intrinsic_wait {count = 0 : i32, ttg.num_tdm_ops = 0 : i32}
    tt.return
  }
}
