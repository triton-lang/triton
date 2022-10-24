// RUN: triton-opt %s | FileCheck %s

func @cast_ops(%scalar_ptr: !tt.ptr<f32>, %scalar_f32: f32, %scalar_i64: i64) {
  // scalar -> scalar
  // CHECK:  i64 -> !tt.ptr<f32>
  %0 = tt.int_to_ptr %scalar_i64 : i64 -> !tt.ptr<f32>
  // CHECK: !tt.ptr<f32> -> i64
  %1 = tt.ptr_to_int %scalar_ptr : !tt.ptr<f32> -> i64
  // CHECK: f32 -> f16
  %2 = tt.fp_to_fp %scalar_f32 : f32 -> f16

  // 0D tensor -> 0D tensor
  %tensor_ptr_0d = tt.splat %scalar_ptr : (!tt.ptr<f32>) -> tensor<!tt.ptr<f32>>
  %tensor_f32_0d = tt.splat %scalar_f32 : (f32) -> tensor<f32>
  %tensor_i64_0d = tt.splat %scalar_i64 : (i64) -> tensor<i64>

  // CHECK: tensor<i64> -> tensor<!tt.ptr<f32>>
  %3 = tt.int_to_ptr %tensor_i64_0d : tensor<i64> -> tensor<!tt.ptr<f32>>
  // CHECK: tensor<!tt.ptr<f32>> -> tensor<i64>
  %4 = tt.ptr_to_int %tensor_ptr_0d : tensor<!tt.ptr<f32>> -> tensor<i64>
  // CHECK: tensor<f32> -> tensor<f16>
  %5 = tt.fp_to_fp %tensor_f32_0d : tensor<f32> -> tensor<f16>

  // 1D tensor -> 1D tensor
  %tensor_ptr_1d = tt.splat %scalar_ptr : (!tt.ptr<f32>) -> tensor<16x!tt.ptr<f32>>
  %tensor_f32_1d = tt.splat %scalar_f32 : (f32) -> tensor<16xf32>
  %tensor_i64_1d = tt.splat %scalar_i64 : (i64) -> tensor<16xi64>

  // CHECK: tensor<16xi64> -> tensor<16x!tt.ptr<f32>>
  %6 = tt.int_to_ptr %tensor_i64_1d : tensor<16xi64> -> tensor<16x!tt.ptr<f32>>
  // CHECK: tensor<16x!tt.ptr<f32>> -> tensor<16xi64>
  %7 = tt.ptr_to_int %tensor_ptr_1d : tensor<16x!tt.ptr<f32>> -> tensor<16xi64>
  // CHECK: tensor<16xf32> -> tensor<16xf16>
  %8 = tt.fp_to_fp %tensor_f32_1d : tensor<16xf32> -> tensor<16xf16>
  return
}

func @addptr_ops(%scalar_ptr: !tt.ptr<f32>, %scalar_i32: i32) {
  // scalar -> scalar
  // CHECK: !tt.ptr<f32>
  %0 = tt.addptr %scalar_ptr, %scalar_i32 : !tt.ptr<f32>

  // 0D tensor -> 0D tensor
  %tensor_ptr_0d = tt.splat %scalar_ptr : (!tt.ptr<f32>) -> tensor<!tt.ptr<f32>>
  %tensor_i32_0d = tt.splat %scalar_i32 : (i32) -> tensor<i32>
  // CHECK: tensor<!tt.ptr<f32>>
  %1 = tt.addptr %tensor_ptr_0d, %tensor_i32_0d : tensor<!tt.ptr<f32>>

  // 1D tensor -> 1D tensor
  %tensor_ptr_1d = tt.splat %scalar_ptr : (!tt.ptr<f32>) -> tensor<16x!tt.ptr<f32>>
  %tensor_i32_1d = tt.splat %scalar_i32 : (i32) -> tensor<16xi32>
  // CHECK: tensor<16x!tt.ptr<f32>>
  %2 = tt.addptr %tensor_ptr_1d, %tensor_i32_1d : tensor<16x!tt.ptr<f32>>
  return
}

func @load_store_ops_scalar(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %mask : i1) {
  // Test if Load/Store ops can handle scalar values
  %other = arith.constant 0.0e+0 : f32

  // load scalar
  // CHECK: %[[L0:.*]] = tt.load %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = true} : f32
  %a = tt.load %ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = true} : f32
  // CHECK: %[[L1:.*]] = tt.load %{{.*}}, %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = true} : f32
  %b = tt.load %ptr, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = true} : f32
  // CHECK: %[[L2:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = true} : f32
  %c = tt.load %ptr, %mask, %other {cache = 1 : i32, evict = 1 : i32, isVolatile = true} : f32

  // store scalar
  // CHECK: tt.store %{{.*}}, %[[L0]] : f32
  tt.store %ptr, %a : f32
  // CHECK: tt.store %{{.*}}, %[[L1]], %{{.*}} : f32
  tt.store %ptr, %b, %mask : f32
  // CHECK: tt.store %{{.*}}, %[[L2]], %{{.*}} : f32
  tt.store %ptr, %c, %mask : f32
  return
}

func @reduce_ops_infer(%ptr: !tt.ptr<f32>, %v : tensor<1x2x4xf32>) {
  // Test if reduce ops infer types correctly

  // CHECK: %{{.*}} = tt.reduce %{{.*}} -> tensor<2x4xf32>
  %a = tt.reduce %v {redOp = 1 : i32, axis = 0 : i32} : tensor<1x2x4xf32> -> tensor<2x4xf32>
  // CHECK: %{{.*}} = tt.reduce %{{.*}} -> tensor<1x4xf32>
  %b = tt.reduce %v {redOp = 1 : i32, axis = 1 : i32} : tensor<1x2x4xf32> -> tensor<1x4xf32>
  // CHECK: %{{.*}} = tt.reduce %{{.*}} -> tensor<1x2xf32>
  %c = tt.reduce %v {redOp = 1 : i32, axis = 2 : i32} : tensor<1x2x4xf32> -> tensor<1x2xf32>
  // CHECK: %{{.*}} = tt.reduce %{{.*}} -> tensor<1xf32>
  %e = tt.reduce %b {redOp = 1 : i32, axis = 1 : i32} : tensor<1x4xf32> -> tensor<1xf32>
  // CHECK: %{{.*}} = tt.reduce %{{.*}} -> tensor<4xf32>
  %f = tt.reduce %a {redOp = 1 : i32, axis = 0 : i32} : tensor<2x4xf32> -> tensor<4xf32>
  // CHECK: %{{.*}} = tt.reduce %{{.*}} -> f32
  %g = tt.reduce %f {redOp = 1 : i32, axis = 0 : i32} : tensor<4xf32> -> f32

  // Avoid optimizations for c, e, and g
  %ptr1x2 = tt.splat %ptr : (!tt.ptr<f32>) -> tensor<1x2x!tt.ptr<f32>>
  %ptr1 = tt.splat %ptr : (!tt.ptr<f32>) -> tensor<1x!tt.ptr<f32>>
  tt.store %ptr1x2, %c : tensor<1x2xf32>
  tt.store %ptr1, %e : tensor<1xf32>
  tt.store %ptr, %g : f32
  return
}

func @dot_ops_infer(%ptr: !tt.ptr<f32>, %v : f32) {
  // Test if reduce ops infer types correctly
  %v128x32 = tt.splat %v : (f32) -> tensor<128x32xf32>
  %v32x128 = tt.splat %v : (f32) -> tensor<32x128xf32>
  %v128x1 = tt.splat %v : (f32) -> tensor<128x1xf32>
  %v1x128 = tt.splat %v : (f32) -> tensor<1x128xf32>

  %zero128x128 = arith.constant dense<0.00e+00> : tensor<128x128xf32>
  %zero32x32 = arith.constant dense<0.00e+00> : tensor<32x32xf32>
  %zero1x1 = arith.constant dense<0.00e+00> : tensor<1x1xf32>

  // CHECK: %{{.*}} = tt.dot %{{.*}} -> tensor<128x128xf32>
  %r1 = tt.dot %v128x32, %v32x128, %zero128x128 {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf32> * tensor<32x128xf32> -> tensor<128x128xf32>
  // CHECK: %{{.*}} = tt.dot %{{.*}} -> tensor<32x32xf32>
  %r2 = tt.dot %v32x128, %v128x32, %zero32x32 {allowTF32 = true, transA = false, transB = false} : tensor<32x128xf32> * tensor<128x32xf32> -> tensor<32x32xf32>
  // CHECK: %{{.*}} = tt.dot %{{.*}} -> tensor<128x128xf32>
  %r3 = tt.dot %v128x1, %v1x128, %zero128x128 {allowTF32 = true, transA = false, transB = false} : tensor<128x1xf32> * tensor<1x128xf32> -> tensor<128x128xf32>
  // CHECK: %{{.*}} = tt.dot %{{.*}} -> tensor<1x1xf32>
  %r4 = tt.dot %v1x128, %v128x1, %zero1x1 {allowTF32 = true, transA = false, transB = false} : tensor<1x128xf32> * tensor<128x1xf32> -> tensor<1x1xf32>

  %ptr128x128 = tt.splat %ptr : (!tt.ptr<f32>) -> tensor<128x128x!tt.ptr<f32>>
  %ptr32x32 = tt.splat %ptr : (!tt.ptr<f32>) -> tensor<32x32x!tt.ptr<f32>>
  %ptr1x1 = tt.splat %ptr : (!tt.ptr<f32>) -> tensor<1x1x!tt.ptr<f32>>
  tt.store %ptr128x128, %r1 : tensor<128x128xf32>
  tt.store %ptr32x32, %r2 : tensor<32x32xf32>
  tt.store %ptr128x128, %r3 : tensor<128x128xf32>
  tt.store %ptr1x1, %r4 : tensor<1x1xf32>
  return
}
