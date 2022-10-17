// RUN: triton-opt %s -convert-triton-to-tritongpu=num-warps=2 | FileCheck %s

func @ops() {
  // CHECK: module attributes {"triton_gpu.num-warps" = 2 : i32} {{.*}}
  %a = arith.constant dense<1.00e+00> : tensor<128x32xf16>
  %b = arith.constant dense<2.00e+00> : tensor<32x128xf16>
  %c = arith.constant dense<3.00e+00> : tensor<128x128xf32>
  %0 = tt.dot %a, %b, %c {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
  return
}

func @load_ops(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
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
  return
}
