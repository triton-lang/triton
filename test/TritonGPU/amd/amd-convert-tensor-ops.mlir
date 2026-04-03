// RUN: triton-opt %s -split-input-file --tritonamdgpu-convert-tensor-ops | FileCheck %s

// CHECK-LABEL: test_cvt1
// CHECK: amdg.async_tdm_copy_global_to_local {{.*}}: !tt.tensordesc<tensor<128x16xf16, #shared>> -> !ttg.memdesc<128x16xf16, #shared, #smem, mutable>
// CHECK: amdg.async_tdm_wait  {num = 0 : i32}
// CHECK: amdg.async_tdm_copy_local_to_global {{.*}} : !ttg.memdesc<128x128xf16, #shared2, #smem, mutable> -> !tt.tensordesc<tensor<128x128xf16, #shared2>>
// CHECK: amdg.async_tdm_wait  {num = 0 : i32}

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = [[0, 0], [1, 0]]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0], CGALayout = [[0, 1], [0, 0]]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0], CGALayout = [[0, 1], [1, 0]]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [0, 2], [1, 0]]}, CGALayout = [[0, 1], [1, 0]], instrShape = [16, 16, 32]}>
#shared = #ttg.padded_shared<[128:+8] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], block = [[0, 0], [64, 0]]}>
#shared1 = #ttg.padded_shared<[128:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0], [8, 0]], block = [[0, 64], [0, 0]]}>
#shared2 = #ttg.padded_shared<[64:+8] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], block = [[0, 64], [64, 0]]}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @test_cvt1(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                            %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                            %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c512_i64 = arith.constant 512 : i64
    %c512_i32 = arith.constant 512 : i32
    %c1_i64 = arith.constant 1 : i64
    %c256_i64 = arith.constant 256 : i64
    %c256_i32 = arith.constant 256 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c128_i32 : i32
    %3 = arith.muli %1, %c128_i32 : i32
    %4 = tt.make_tensor_descriptor %a_ptr, [%c1024_i32, %c256_i32], [%c256_i64, %c1_i64] : <f16>, <tensor<128x16xf16, #shared>>
    %5 = tt.make_tensor_descriptor %b_ptr, [%c256_i32, %c512_i32], [%c512_i64, %c1_i64] : <f16>, <tensor<16x128xf16, #shared1>>
    %6 = tt.make_tensor_descriptor %c_ptr, [%c1024_i32, %c512_i32], [%c512_i64, %c1_i64] : <f16>, <tensor<128x128xf16, #shared2>>
    %7 = tt.descriptor_load %4[%2, %c0_i32] : !tt.tensordesc<tensor<128x16xf16, #shared>> -> tensor<128x16xf16, #blocked>
    %8 = tt.descriptor_load %5[%c0_i32, %3] : !tt.tensordesc<tensor<16x128xf16, #shared1>> -> tensor<16x128xf16, #blocked1>
    %9 = ttg.convert_layout %7 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    %10 = ttg.convert_layout %8 : tensor<16x128xf16, #blocked1> -> tensor<16x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    %11 = tt.dot %9, %10, %cst : tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<16x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
    %12 = arith.truncf %11 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %13 = ttg.convert_layout %12 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked2>
    tt.descriptor_store %6[%2, %3], %13 : !tt.tensordesc<tensor<128x128xf16, #shared2>>, tensor<128x128xf16, #blocked2>
    tt.return
  }
}
