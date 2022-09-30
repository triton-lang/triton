// RUN: triton-opt %s -tritongpu-combine 2>&1 | FileCheck %s

#layout0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#layout1 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

func @remat(%arg0: i32) -> tensor<1024xi32, #layout1> {
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout0>
  %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #layout0>
  %2 = arith.muli %0, %1 : tensor<1024xi32, #layout0>
  %3 = arith.muli %2, %1 : tensor<1024xi32, #layout0>
  %4 = arith.muli %2, %3 : tensor<1024xi32, #layout0>
  %5 = triton_gpu.convert_layout %4 : (tensor<1024xi32, #layout0>) -> tensor<1024xi32, #layout1>
  return %5: tensor<1024xi32, #layout1>
  // CHECK: %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[target_layout]]>
  // CHECK: %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[target_layout]]>
  // CHECK: %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[target_layout]]>
  // CHECK: %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, [[target_layout]]>
  // CHECK: %4 = arith.muli %2, %3 : tensor<1024xi32, [[target_layout]]>
  // CHECK: %5 = arith.muli %0, %1 : tensor<1024xi32, [[target_layout]]>
  // CHECK: %6 = arith.addi %4, %5 : tensor<1024xi32, [[target_layout]]>
  // CHECK: return %6 : tensor<1024xi32, [[target_layout]]>
}