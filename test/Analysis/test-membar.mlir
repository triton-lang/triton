// RUN: triton-opt %s -split-input-file --mlir-disable-threading -test-print-membar 2>&1 | FileCheck %s

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#sliceAd0 = #triton_gpu.slice<{dim = 0, parent = #AL}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#A_SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#A_SHARED_T = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#B_SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C = #triton_gpu.mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A_DOT = #triton_gpu.dot_op<{opIdx = 0, parent = #C}>
#B_DOT = #triton_gpu.dot_op<{opIdx = 1, parent = #C}>

module attributes {"triton_gpu.num-warps" = 4 : i32} {

// If branch inserted a barrier for %cst0 and %cst1, but else didn't, then the barrier should be inserted in the parent region
// CHECK-LABEL: multi_blocks
func @multi_blocks(%i1 : i1) {
  %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  %cst1 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
  scf.if %i1 {
    // CHECK: Membar 2
    %a = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
    scf.yield
  } else {
    %cst2 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
    %cst3 = arith.constant dense<0.000000e+00> : tensor<16x16xf16, #A_SHARED>
    // CHECK-NEXT: Membar 7
    %b = tt.cat %cst2, %cst3 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
    scf.yield
  }
  // CHECK-NEXT: Membar 10
  %c = tt.cat %cst0, %cst1 {axis = 0} : (tensor<16x16xf16, #A_SHARED>, tensor<16x16xf16, #A_SHARED>) -> tensor<32x16xf16, #A_SHARED>
  return
}

}
