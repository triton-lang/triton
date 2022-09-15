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
