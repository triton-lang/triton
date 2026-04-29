// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-buffer-ops="gfx-arch=gfx1250" | FileCheck %s

// CHECK-LABEL: subslice_assert

#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, CGALayout = [[0, 1]], instrShape = [16, 16, 32]}>
#shared = #ttg.padded_shared<[128:+8] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0], [4, 0], [8, 0]], block = [[0, 0]]}>
#shared1 = #ttg.padded_shared<[128:+8] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [1, 0], [2, 0], [4, 0], [8, 0]], block = [[16, 0]]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1]]}>
#shared3 = #ttg.padded_shared<[128:+8] {offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 1], [0, 2], [0, 4], [0, 8]], block = [[0, 16]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 17376 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @subslice_assert(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                  %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                  %c_ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %c65536_i32 = arith.constant 65536 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x32xf32, #mma>
    %c1_i64 = arith.constant 1 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c1024_i32 = arith.constant 1024 : i32
    %c32_i32 = arith.constant 32 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %c16_i32 = arith.constant 16 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.remsi %0, %c64_i32 : i32
    %2 = arith.divsi %0, %c64_i32 : i32
    %3 = arith.muli %1, %c16_i32 : i32
    %4 = arith.muli %1, %c32768_i32 : i32
    %5 = arith.muli %2, %c32_i32 : i32
    %6 = arith.muli %2, %c65536_i32 : i32
    %7 = tt.addptr %a_ptr, %4 : !tt.ptr<f16>, i32
    %8 = tt.make_tensor_descriptor %7, [%c1024_i32, %c2048_i32], [%c2048_i64, %c1_i64] : <f16>, <16x128xf16, #shared>
    %9 = tt.addptr %b_ptr, %6 : !tt.ptr<f16>, i32
    %10 = tt.make_tensor_descriptor %9, [%c2048_i32, %c2048_i32], [%c2048_i64, %c1_i64] : <f16>, <32x128xf16, #shared1>
    %11 = tt.make_tensor_descriptor %c_ptr, [%c1024_i32, %c2048_i32], [%c2048_i64, %c1_i64] : <bf16>, <16x32xbf16, #shared2>
    %12 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable>
    %13 = ttg.local_alloc {allocation.offset = 8688 : i32} : () -> !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable>
    %14 = ttg.memdesc_index %12[%c0_i32] : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %15 = amdg.async_tdm_copy_global_to_local %8[%c0_i32, %c0_i32] into %14, pred = %c1_i32 : !tt.tensordesc<16x128xf16, #shared> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %16 = ttg.memdesc_index %13[%c0_i32] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    %17 = amdg.async_tdm_copy_global_to_local %10[%c0_i32, %c0_i32] into %16, pred = %c1_i32 : !tt.tensordesc<32x128xf16, #shared1> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    %18 = amdg.async_tdm_intrinsic_wait {count = 0 : i32, ttg.num_tdm_ops = 0 : i32}
    %19 = ttg.memdesc_subslice %14[0, 0] : !ttg.memdesc<16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128>
    %20 = ttg.local_load %19 : !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %21 = ttg.memdesc_subslice %16[0, 0] : !ttg.memdesc<32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf16, #shared1, #smem, mutable, 32x128>
    %22 = ttg.memdesc_trans %21 {order = array<i32: 1, 0>} : !ttg.memdesc<32x32xf16, #shared1, #smem, mutable, 32x128> -> !ttg.memdesc<32x32xf16, #shared3, #smem, mutable, 128x32>
    %23 = ttg.local_load %22 : !ttg.memdesc<32x32xf16, #shared3, #smem, mutable, 128x32> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %24 = ttg.memdesc_index %12[%c1_i32] : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %25 = amdg.async_tdm_copy_global_to_local %8[%c0_i32, %c128_i32] into %24, pred = %c1_i32 : !tt.tensordesc<16x128xf16, #shared> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable>
    %26 = ttg.memdesc_index %13[%c1_i32] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    %27 = amdg.async_tdm_copy_global_to_local %10[%c0_i32, %c128_i32] into %26, pred = %c1_i32 : !tt.tensordesc<32x128xf16, #shared1> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    %28 = ttg.memdesc_subslice %14[0, 32] : !ttg.memdesc<16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128>
    %29 = ttg.local_load %28 : !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %30 = ttg.memdesc_subslice %16[0, 32] : !ttg.memdesc<32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf16, #shared1, #smem, mutable, 32x128>
    %31 = ttg.memdesc_trans %30 {order = array<i32: 1, 0>} : !ttg.memdesc<32x32xf16, #shared1, #smem, mutable, 32x128> -> !ttg.memdesc<32x32xf16, #shared3, #smem, mutable, 128x32>
    %32 = ttg.local_load %31 : !ttg.memdesc<32x32xf16, #shared3, #smem, mutable, 128x32> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %33 = tt.dot %20, %23, %cst : tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x32xf32, #mma>
    %34 = ttg.memdesc_subslice %14[0, 64] : !ttg.memdesc<16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128>
    %35 = ttg.local_load %34 : !ttg.memdesc<16x32xf16, #shared, #smem, mutable, 16x128> -> tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %36 = ttg.memdesc_subslice %16[0, 64] : !ttg.memdesc<32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf16, #shared1, #smem, mutable, 32x128>
    %37 = ttg.memdesc_trans %36 {order = array<i32: 1, 0>} : !ttg.memdesc<32x32xf16, #shared1, #smem, mutable, 32x128> -> !ttg.memdesc<32x32xf16, #shared3, #smem, mutable, 128x32>
    %38 = ttg.local_load %37 : !ttg.memdesc<32x32xf16, #shared3, #smem, mutable, 128x32> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %39 = tt.dot %29, %32, %33 : tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x32xf32, #mma>
    %40 = tt.dot %35, %38, %39 : tensor<16x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<16x32xf32, #mma>
    %41 = amdg.async_tdm_intrinsic_wait {count = 0 : i32, ttg.num_tdm_ops = 0 : i32}
    %42 = ttg.memdesc_reinterpret %12 : !ttg.memdesc<2x16x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x32xbf16, #shared2, #smem, mutable>
    %43 = arith.truncf %40 : tensor<16x32xf32, #mma> to tensor<16x32xbf16, #mma>
    ttg.local_store %43, %42 : tensor<16x32xbf16, #mma> -> !ttg.memdesc<16x32xbf16, #shared2, #smem, mutable>
    amdg.async_tdm_copy_local_to_global %11[%3, %5] from %42 : !ttg.memdesc<16x32xbf16, #shared2, #smem, mutable> -> !tt.tensordesc<16x32xbf16, #shared2>
    %44 = amdg.async_tdm_intrinsic_wait {count = 0 : i32, ttg.num_tdm_ops = 0 : i32}
    tt.return
  }
}
