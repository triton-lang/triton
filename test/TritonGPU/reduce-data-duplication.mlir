// RUN: triton-opt %s -split-input-file -tritongpu-reduce-data-duplication | FileCheck %s

//       CHECK:   #[[SHARED:.*]] = #triton_gpu.shared<{vec = 8, perPhase = 8, maxPhase = 2, order = [0, 1], hasLeadingOffset = false}
//       CHECK:   apply_swizzle
//       CHECK:   %[[CVTS:.*]] = triton_gpu.local_alloc %{{.*}} : (tensor<16x256xf16, #{{.*}}>) -> !tt.memdesc<16x256xf16, #[[SHARED]]>
//       CHECK:   %[[OPERAND1:.*]] = triton_gpu.local_load %[[CVTS]]
//       CHECK:   tt.dot %[[OPERAND1]]

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @apply_swizzle(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8, 1> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<16x32xf32, #mma> 
    %11 = tt.make_tensor_ptr %arg0, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x256xf16, #blocked>, 1> 
    %14 = tt.make_tensor_ptr %arg1, [%c0_i64, %c0_i64], [%c0_i64, %c0_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x32xi8, #blocked1>, 1> 
    %21 = tt.load %11 {boundaryCheck = array<i32: 0>, cache = 1 : i32, evict = 1 : i32, isVolatile = false, padding = 1 : i32} : !tt.ptr<tensor<16x256xf16, #blocked>, 1> -> tensor<16x256xf16, #blocked2> 
    %23 = tt.load %14 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<256x32xi8, #blocked1>, 1> -> tensor<256x32xi8, #blocked3> 
    %25 = triton_gpu.convert_layout %21 : tensor<16x256xf16, #blocked2> -> tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> 
    %26 = triton_gpu.convert_layout %23 : tensor<256x32xi8, #blocked3> -> tensor<256x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
    %27 = arith.sitofp %26 : tensor<256x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> to tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> 
    %28 = tt.dot %25, %27, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<16x256xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<16x32xf32, #mma> 
    tt.return 
  } 
} 

// -----