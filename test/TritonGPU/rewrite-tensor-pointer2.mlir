// RUN: triton-opt %s -tritongpu-rewrite-tensor-pointer | FileCheck %s

#blocked = #triton_gpu.blocked<{
    sizePerThread = [1, 1],
    threadsPerWarp = [32, 1],
    warpsPerCTA = [4, 1],
    order = [0, 1],
    CTAsPerCGA = [1, 1],
    CTASplitNum = [1, 1],
CTAOrder = [0, 1]}>

#blocked1 = #triton_gpu.blocked<{
    sizePerThread = [1, 1, 1, 1],
    threadsPerWarp = [32, 1, 1, 1],
    warpsPerCTA = [4, 1, 1, 1],
    order = [0, 1, 2, 3],
    CTAsPerCGA = [1, 1, 1, 1],
    CTASplitNum = [1, 1, 1, 1],
CTAOrder = [0, 1, 2, 3]}>

module attributes {
    "triton_gpu.compute-capability" = 80 : i32,
    "triton_gpu.num-ctas" = 1 : i32,
    "triton_gpu.num-warps" = 4 : i32
} {

// CHECK-LABEL: @test2D
tt.func public @test2D(%base_ptr: !tt.ptr<f32, 1>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c32_i64 = arith.constant 32 : i64
    // CHECK-DAG: tt.splat {{.*}} : (i64) -> tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    // CHECK-DAG: tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    // CHECK: tt.expand_dims {{.*}} {axis = 1 : i32} : (tensor<16xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<16x1xi64, #blocked>
    // CHECK: tt.broadcast {{.*}} : (tensor<16x1xi64, #blocked>) -> tensor<16x32xi64, #blocked>

    // CHECK-DAG: tt.splat {{.*}} : (i64) -> tensor<32xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-DAG: tt.make_range {{.*}} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK: tt.expand_dims {{.*}} {axis = 0 : i32} : (tensor<32xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x32xi64, #blocked>
    // CHECK: tt.broadcast {{.*}} : (tensor<1x32xi64, #blocked>) -> tensor<16x32xi64, #blocked>
    %tensor_ptr = tt.make_tensor_ptr %base_ptr,
      [%c16_i64, %c32_i64], // shape
      [%c1_i64, %c1_i64], // strides
      [%c0_i32, %c0_i32] // offsets
      {order = array<i32: 1, 0>}:
      !tt.ptr<tensor<16x32xf32, #blocked>, 1>
    %tensor = tt.load %tensor_ptr
      {cache = 1 : i32, evict = 1 : i32, isVolatile = false}:
      !tt.ptr<tensor<16x32xf32, #blocked>, 1> -> tensor<16x32xf32, #blocked>
    tt.return
}

// CHECK-LABEL: @test4D
tt.func public @test4D(%base_ptr: !tt.ptr<f32, 1>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c4_i64 = arith.constant 4 : i64
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64

    // One run of the dim-expansion algorithm.  This pattern repeats four times,
    // once for each dimension, but we only check one of them.
    // CHECK-DAG: tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 3, parent = #blocked1}>}>}>>
    // CHECK-DAG: tt.splat {{.*}} : (i64) -> tensor<2xi64, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 3, parent = #blocked1}>}>}>>
    // CHECK: tt.expand_dims {{.*}} {axis = 1 : i32} : (tensor<2xi64, #triton_gpu.slice<{dim = 1, parent = #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 3, parent = #blocked1}>}>}>>) -> tensor<2x1xi64, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 3, parent = #blocked1}>}>>
    // CHECK: tt.broadcast {{.*}} : (tensor<2x1xi64, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 3, parent = #blocked1}>}>>) -> tensor<2x4xi64, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 3, parent = #blocked1}>}>>
    // CHECK: tt.expand_dims {{.*}} {axis = 2 : i32} : (tensor<2x4xi64, #triton_gpu.slice<{dim = 2, parent = #triton_gpu.slice<{dim = 3, parent = #blocked1}>}>>) -> tensor<2x4x1xi64, #triton_gpu.slice<{dim = 3, parent = #blocked1}>>
    // CHECK: tt.broadcast {{.*}} : (tensor<2x4x1xi64, #triton_gpu.slice<{dim = 3, parent = #blocked1}>>) -> tensor<2x4x8xi64, #triton_gpu.slice<{dim = 3, parent = #blocked1}>>
    // CHECK: tt.expand_dims {{.*}} {axis = 3 : i32} : (tensor<2x4x8xi64, #triton_gpu.slice<{dim = 3, parent = #blocked1}>>) -> tensor<2x4x8x1xi64, #blocked1>
    // CHECK: tt.broadcast {{.*}} : (tensor<2x4x8x1xi64, #blocked1>) -> tensor<2x4x8x16xi64, #blocked1>

    %tensor_ptr = tt.make_tensor_ptr %base_ptr,
      [%c2_i64, %c4_i64, %c8_i64, %c16_i64], // shape
      [%c1_i64, %c1_i64, %c1_i64, %c1_i64], // strides
      [%c0_i32, %c0_i32, %c0_i32, %c0_i32] // offsets
      {order = array<i32: 3, 2, 1, 0>}:
      !tt.ptr<tensor<2x4x8x16xf32, #blocked1>, 1>
    %tensor = tt.load %tensor_ptr
      {cache = 1 : i32, evict = 1 : i32, isVolatile = false}:
      !tt.ptr<tensor<2x4x8x16xf32, #blocked1>, 1> -> tensor<2x4x8x16xf32, #blocked1>
    tt.return
}

} // end module
