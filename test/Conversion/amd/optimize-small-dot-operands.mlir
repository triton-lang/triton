// RUN: triton-opt %s --tritonamdgpu-optimize-small-dot-operands -split-input-file | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: small_dot
  tt.func @small_dot(%a: tensor<1x512xi8, #blocked>, %bPtr: tensor<512x32x!tt.ptr<i8>, #blocked>, %outPtr: tensor<1x32x!tt.ptr<i32>, #blocked>) {
    %aOp = triton_gpu.convert_layout %a : tensor<1x512xi8, #blocked> -> tensor<1x512xi8, #dot_operand_a>
    %b = tt.load %bPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<512x32x!tt.ptr<i8>, #blocked>
    %bOp = triton_gpu.convert_layout %b : tensor<512x32xi8, #blocked> -> tensor<512x32xi8, #dot_operand_b>
    %c = arith.constant dense<0> : tensor<1x32xi32, #blocked>
    // CHECK: tt.dot
    %0 = tt.dot %aOp, %bOp, %c, inputPrecision = tf32 : tensor<1x512xi8, #dot_operand_a> * tensor<512x32xi8, #dot_operand_b> -> tensor<1x32xi32, #blocked>
    tt.store %outPtr, %0 : tensor<1x32x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: small_dot_m2
  tt.func @small_dot_m2(%a: tensor<2x512xi8, #blocked>, %bPtr: tensor<512x32x!tt.ptr<i8>, #blocked>, %outPtr: tensor<2x32x!tt.ptr<i32>, #blocked>) {
    %aOp = triton_gpu.convert_layout %a : tensor<2x512xi8, #blocked> -> tensor<2x512xi8, #dot_operand_a>
    %b = tt.load %bPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<512x32x!tt.ptr<i8>, #blocked>
    %bOp = triton_gpu.convert_layout %b : tensor<512x32xi8, #blocked> -> tensor<512x32xi8, #dot_operand_b>
    %c = arith.constant dense<0> : tensor<2x32xi32, #blocked>
    // CHECK: tt.dot
    %0 = tt.dot %aOp, %bOp, %c, inputPrecision = tf32 : tensor<2x512xi8, #dot_operand_a> * tensor<512x32xi8, #dot_operand_b> -> tensor<2x32xi32, #blocked>
    tt.store %outPtr, %0 : tensor<2x32x!tt.ptr<i32>, #blocked>
    tt.return
  }
}
