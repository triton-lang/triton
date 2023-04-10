// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu=num-warps=2 | FileCheck %s

tt.func @ops() {
  // CHECK: module attributes {"triton_gpu.num-warps" = 2 : i32} {{.*}}
  %a = arith.constant dense<1.00e+00> : tensor<128x32xf16>
  %b = arith.constant dense<2.00e+00> : tensor<32x128xf16>
  %c = arith.constant dense<3.00e+00> : tensor<128x128xf32>
  %0 = tt.dot %a, %b, %c {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
  tt.return
}

// -----

tt.func @load_ops(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // Test if LoadOp is lowered properly (see #771)
  %ptrs = tt.splat %ptr : (!tt.ptr<f32>) -> tensor<128x!tt.ptr<f32>>
  %mask = arith.constant dense<true> : tensor<128xi1>
  %other = arith.constant dense<0.0e+0> : tensor<128xf32>
  // CHECK: %{{.*}} = tt.load %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = true} : {{.*}}
  %a = tt.load %ptrs {cache = 1 : i32, evict = 1 : i32, isVolatile = true} : tensor<128xf32>
  // CHECK: %{{.*}} = tt.load %{{.*}}, %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = true} : {{.*}}
  %b = tt.load %ptrs, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = true} : tensor<128xf32>
  // CHECK: %{{.*}} = tt.load %{{.*}}, %{{.*}}, %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = true} : {{.*}}
  %c = tt.load %ptrs, %mask, %other {cache = 1 : i32, evict = 1 : i32, isVolatile = true} : tensor<128xf32>
  tt.store %ptrs, %a : tensor<128xf32>
  tt.store %ptrs, %b : tensor<128xf32>
  tt.store %ptrs, %c : tensor<128xf32>
  tt.return
}

// -----

tt.func @reduce_ops(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  // Test if the total number of threadsPerWarp is 32
  // Test if the total number of warps is 2
  // CHECK: #[[blocked0:.*]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 2], order = [0, 1]}>
  // CHECK: #[[blocked1:.*]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 2], order = [0, 1]}>
  // CHECK: #[[blocked2:.*]] = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 2], order = [0, 1]}>
  // CHECK: module attributes {"triton_gpu.num-warps" = 2 : i32} {{.*}}
  %c0 = arith.constant dense<1.00e+00> : tensor<4x4xf32>
  %c1 = arith.constant dense<2.00e+00> : tensor<8x2xf32>
  %c2 = arith.constant dense<3.00e+00> : tensor<16x16xf32>
  // CHECK: tensor<4x4xf32, #[[blocked0]]> -> tensor<4xf32, #triton_gpu.slice<{dim = 0, parent = #[[blocked0]]}>>
  %c0_ = tt.reduce %c0 {redOp = 1 : i32, axis = 0 : i32} : tensor<4x4xf32> -> tensor<4xf32>
  // CHECK: tensor<8x2xf32, #[[blocked1]]> -> tensor<2xf32, #triton_gpu.slice<{dim = 0, parent = #[[blocked1]]}>
  %c1_ = tt.reduce %c1 {redOp = 1 : i32, axis = 0 : i32} : tensor<8x2xf32> -> tensor<2xf32>
  // CHECK: tensor<8x2xf32, #[[blocked1]]> -> tensor<8xf32, #triton_gpu.slice<{dim = 1, parent = #[[blocked1]]}>>
  %c2_ = tt.reduce %c1 {redOp = 1 : i32, axis = 1 : i32} : tensor<8x2xf32> -> tensor<8xf32>
  // CHECK: tensor<16x16xf32, #[[blocked2]]> -> tensor<16xf32, #triton_gpu.slice<{dim = 0, parent = #[[blocked2]]}>>
  %c3_ = tt.reduce %c2 {redOp = 1 : i32, axis = 0 : i32} : tensor<16x16xf32> -> tensor<16xf32>

  tt.return
}
