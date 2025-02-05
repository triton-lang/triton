// RUN: triton-opt %s -split-input-file -tritonamdgpu-in-thread-transpose | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#smem = #ttg.shared_memory
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [32, 32], isTransposed = true}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-DAG: [[threadrake_layout:#.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
// CHECK-DAG: [[transposed_in_regs_layout:#.*]] = #ttg.linear<{offset = {{.*}}}>
// CHECK-DAG: [[special_shared:#.*]] = #ttg.swizzled_blocks_shared
// CHECK: [[load_ptr:%.*]] = ttg.convert_layout {{.*}} -> tensor<32x128x!tt.ptr<f16>, [[threadrake_layout]]>
// CHECK: [[load_val:%.*]] = tt.load [[load_ptr]] : tensor<32x128x!tt.ptr<f16>, [[threadrake_layout]]>
// CHECK: [[transposed_in_reg:%.*]] = ttg.convert_layout [[load_val]]{{.*}} -> tensor<32x128xf16, [[transposed_in_regs_layout]]>
// CHECK: [[shared_value:%.*]] = ttg.local_alloc [[transposed_in_reg]] : (tensor<32x128xf16, #transposed_in_reg>) -> !ttg.memdesc<32x128xf16, [[special_shared]], #smem>
// CHECK: {{.*}} = ttg.local_load [[shared_value]] : !ttg.memdesc<32x128xf16, [[special_shared]], #smem> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
  tt.func public @threadRake_transpose_b_no_change(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x32x!tt.ptr<f16>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %2 = tt.load %0 : tensor<256x32x!tt.ptr<f16>, #blocked>
    %3 = tt.load %1 : tensor<32x128x!tt.ptr<f16>, #blocked1>
    %4 = ttg.convert_layout %2 : tensor<256x32xf16, #blocked> -> tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>

    %5 = ttg.local_alloc %3 : (tensor<32x128xf16, #blocked1>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
    %6 = ttg.local_load %5 : !ttg.memdesc<32x128xf16, #shared, #smem> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>

    %7 = tt.dot %4, %6, %cst_0 : tensor<256x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
    tt.return
  }
}
