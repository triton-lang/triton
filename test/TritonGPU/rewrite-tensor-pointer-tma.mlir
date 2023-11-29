// RUN: ENABLE_TMA=1 triton-opt %s -split-input-file -tritongpu-rewrite-tensor-pointer=compute-capability=90 | FileCheck %s
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_0d1d2d3de4de5de6de7c8c9de10de11c(%arg0: !tt.ptr<f8E5M2, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) attributes {noinline = false} {
    %c127_i32 = arith.constant 127 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg4, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg3, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.muli %2, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %4, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %9 : i32
    %11 = arith.addi %7, %10 : i32
    %12 = arith.remsi %0, %5 : i32
    %13 = arith.divsi %12, %9 : i32
    %14 = arith.muli %11, %c128_i32 : i32
    %15 = arith.muli %13, %c128_i32 : i32
    %16 = arith.extsi %arg3 : i32 to i64
    %17 = arith.extsi %arg5 : i32 to i64
    %18 = arith.extsi %arg6 : i32 to i64
    // CHECK: tt.make_tensor_ptr
    %19 = tt.make_tensor_ptr %arg0, [%16, %17], [%18, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf8E5M2, #blocked>, 1>
    %20 = arith.extsi %arg4 : i32 to i64
    %21 = arith.extsi %arg7 : i32 to i64
    // CHECK: tt.make_tensor_ptr
    %22 = tt.make_tensor_ptr %arg1, [%17, %20], [%c1_i64, %21], [%c0_i32, %15] {order = array<i32: 0, 1>} : <tensor<64x128xf8E5M2, #blocked1>, 1>
    tt.return
  }
}
