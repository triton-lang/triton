// RUN: triton-opt %s -split-input-file -canonicalize -triton-combine
// RUN: triton-opt %s -split-input-file -canonicalize -triton-combine | FileCheck %s

// CHECK-LABEL: @test_combine_dot_add_pattern
func.func @test_combine_dot_add_pattern() -> (tensor<128x128xf32>, tensor<128x128xf32>) {
    // CHECK-DAG: %[[d:.*]] = arith.constant dense<3.000000e+00> : tensor<128x128xf32>
    // CHECK-DAG: %[[b:.*]] = arith.constant dense<2.000000e+00> : tensor<128x128xf32>
    // CHECK-DAG: %[[a:.*]] = arith.constant dense<1.000000e+00> : tensor<128x128xf32>
    %a = arith.constant dense<1.0> : tensor<128x128xf32>
    %b = arith.constant dense<2.0> : tensor<128x128xf32>
    %zero = arith.constant dense<0.0> : tensor<128x128xf32>
    %d = arith.constant dense<3.0> : tensor<128x128xf32>

    %dot_out = tt.dot %a, %b, %zero {allowTF32 = true, transA = false, transB = false} : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>

    // CHECK-NEXT: %[[res0:.*]] = tt.dot %[[a]], %[[b]], %[[d]] {allowTF32 = true} : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>
    %res0 = arith.addf %dot_out, %d : tensor<128x128xf32>

    // CHECK-NEXT: %[[res1:.*]] = tt.dot %[[a]], %[[b]], %[[d]] {allowTF32 = true} : tensor<128x128xf32> * tensor<128x128xf32> -> tensor<128x128xf32>
    %res1 = arith.addf %d, %dot_out : tensor<128x128xf32>

    return %res0, %res1 : tensor<128x128xf32>, tensor<128x128xf32>
}


// COM: CHECK-LABEL: @test_combine_addptr_pattern
func.func @test_combine_addptr_pattern(%base: !tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>> {
    %off0 = arith.constant 10 : i32
    %off1 = arith.constant 15 : i32

    // 10 + 15 = 25
    // COM: CHECK-NEXT: %[[cst:.*]] = arith.constant dense<25> : tensor<8xi32>

    %base_ = tt.broadcast %base : (!tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>>

    // COM: CHECK-NEXT: %[[tmp0:.*]] = tt.broadcast %{{.*}} : (!tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>>

    %idx0 = tt.broadcast %off0 : (i32) -> tensor<8xi32>
    %idx1 = tt.broadcast %off1 : (i32) -> tensor<8xi32>

    // COM: CHECK-NEXT: %1 = tt.addptr %[[tmp0]], %[[cst]] : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %ptr0 = tt.addptr %base_, %idx0 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %ptr1 = tt.addptr %ptr0, %idx1 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>

    return %ptr1 : tensor<8x!tt.ptr<f32>>
}


// CHECK-LABEL: @test_combine_select_masked_load_pattern
func.func @test_combine_select_masked_load_pattern(%ptr: tensor<8x!tt.ptr<f32>>, %cond: i1) -> (tensor<8xf32>, tensor<8xf32>) {
    %mask = tt.broadcast %cond : (i1) -> tensor<8xi1>
    %false_val = arith.constant dense<0.0> : tensor<8xf32>

    // CHECK: %[[res1:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<8xf32>
    %x = tt.load %ptr, %mask, %false_val {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<8xf32>
    %0 = arith.select %cond, %x, %false_val : tensor<8xf32>

    // CHECK: %[[res2:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<8xf32>
    %y = tt.load %ptr, %mask, %false_val {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<8xf32>
    %1 = arith.select %cond, %y, %false_val : tensor<8xf32>

    // CHECK: return %[[res1]], %[[res2]] : tensor<8xf32>, tensor<8xf32>
    return %0, %1 : tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: @test_combine_select_masked_load_fail_pattern
func.func @test_combine_select_masked_load_fail_pattern(%ptr: tensor<8x!tt.ptr<f32>>, %dummy_load: tensor<8xf32>, %dummy_broadcast: tensor<8xi1>, %cond0: i1, %cond1: i1) -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
    %false_val = arith.constant dense<0.0> : tensor<8xf32>

    // Case 1: value at the "load" position is not an "op".  Select should not be canonicalized.
    // CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<8xf32>
    %0 = arith.select %cond0, %dummy_load, %false_val : tensor<8xf32>

    // Case 2: value at the "broadcast" position is not an "op".  Select should not be canonicalized.
    %real_load0 = tt.load %ptr, %dummy_broadcast, %false_val {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<8xf32>
    // CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<8xf32>
    %1 = arith.select %cond0, %real_load0, %false_val : tensor<8xf32>

    // Case 3: condition of "broadcast" is not the same as the condition of "select".  Select should not be canonicalized.
    %cond0_ = tt.broadcast %cond0 : (i1) -> tensor<8xi1>
    %real_load1 = tt.load %ptr, %cond0_, %false_val {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<8xf32>
    // CHECK: %{{.*}} = arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<8xf32>
    %2 = arith.select %cond1, %real_load1, %false_val : tensor<8xf32>

    return %0, %1, %2 : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: @test_combine_broadcast_constant_pattern
func.func @test_combine_broadcast_constant_pattern(%cst : f32) -> tensor<8x2xf32> {
    // CHECK: %[[cst:.*]] = arith.constant dense<1.000000e+00> : tensor<8x2xf32>
    %const = arith.constant dense<1.0> : tensor<8xf32>
    %bst_out = tt.broadcast %const : (tensor<8xf32>) -> tensor<8x2xf32>

    // CHECK-NEXT: return %[[cst]] : tensor<8x2xf32>
    return %bst_out : tensor<8x2xf32>
}

// CHECK-LABEL: @test_canonicalize_masked_load_pattern
func.func @test_canonicalize_masked_load_pattern(%ptr: tensor<8x!tt.ptr<f32>>) -> (tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) {
    %true_mask = arith.constant dense<true> : tensor<8xi1>
    %false_mask = arith.constant dense<false> : tensor<8xi1>
    %other_val = arith.constant dense<0.0> : tensor<8xf32>

    // true_mask with other
    // CHECK: %[[res1:.*]] = tt.load %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<8xf32>
    %x = tt.load %ptr, %true_mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<8xf32>

    // true_mask without other
    // CHECK: %[[res2:.*]] = tt.load %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<8xf32>
    %y = tt.load %ptr, %true_mask, %other_val {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<8xf32>

    // false_mask with other. It should become "other" (i.e., %y)
    %z = tt.load %ptr, %false_mask, %y {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<8xf32>

    // CHECK: return %[[res1]], %[[res2]], %[[res2]] : tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
    return %x, %y, %z: tensor<8xf32>, tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: @test_canonicalize_masked_load_fail_pattern
func.func @test_canonicalize_masked_load_fail_pattern(%ptr: tensor<8x!tt.ptr<f32>>, %mask: tensor<8xi1>) -> (tensor<8xf32>, tensor<8xf32>) {
    %other_val = arith.constant dense<0.0> : tensor<8xf32>

    // Case: value at the "mask" position is not an "op".  Load should not be canonicalized.
    // CHECK: %[[res1:.*]] = tt.load %{{.*}}, %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<8xf32>
    %x = tt.load %ptr, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<8xf32>
    // CHECK: %[[res1:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<8xf32>
    %y = tt.load %ptr, %mask, %other_val {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<8xf32>

    return %x, %y: tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: @test_canonicalize_masked_store_pattern
func.func @test_canonicalize_masked_store_pattern(%ptr: tensor<8x!tt.ptr<f32>>, %val: tensor<8xf32>) {
    %true_mask = arith.constant dense<true> : tensor<8xi1>
    %false_mask = arith.constant dense<false> : tensor<8xi1>

    // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<8xf32>
    tt.store %ptr, %val, %true_mask : tensor<8xf32>

    // The following store should disappear.
    // CHECK-NEXT: return
    tt.store %ptr, %val, %false_mask : tensor<8xf32>
    return
}

// CHECK-LABEL: @test_canonicalize_masked_store_fail_pattern
func.func @test_canonicalize_masked_store_fail_pattern(%ptr: tensor<8x!tt.ptr<f32>>, %val: tensor<8xf32>, %mask: tensor<8xi1>) {
    // Case: value at the "mask" position is not an "op".  Store should not be canonicalized.
    // CHECK: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<8xf32>
    tt.store %ptr, %val, %mask : tensor<8xf32>
    return
}

// CHECK-LABEL: @test_canonicalize_expand_dims
func.func @test_canonicalize_expand_dims(%arg0: tensor<f32>) -> (tensor<1x8xf32>) {
    %splat = tt.splat %arg0 : (tensor<f32>) -> tensor<8xf32>
    // CHECK: %{{.*}} = tt.splat %arg0 : (tensor<f32>) -> tensor<1x8xf32>
    %ed = tt.expand_dims %splat {axis = 0 : i32} : (tensor<8xf32>) -> tensor<1x8xf32>

    return %ed : tensor<1x8xf32>
}


// CHECK-LABEL: @test_canonicalize_view
func.func @test_canonicalize_view(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> (tensor<4x2xf32>, tensor<2x2x2xf32>, tensor<8xf32>) {
    %view0 = tt.view %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
    // CHECK: %{{.*}} = tt.view %arg0 : (tensor<8xf32>) -> tensor<4x2xf32>
    %view1 = tt.view %view0 : (tensor<2x4xf32>) -> tensor<4x2xf32>

    %splat = tt.splat %arg1 : (tensor<f32>) -> tensor<8xf32>
    // CHECK: %{{.*}} = tt.splat %arg1 : (tensor<f32>) -> tensor<2x2x2xf32>
    %view2 = tt.view %splat : (tensor<8xf32>) -> tensor<2x2x2xf32>

    %view3 = tt.view %arg0 : (tensor<8xf32>) -> tensor<8xf32>
    // CHECK: %{{.*}} = arith.addf %arg0, %arg0 : tensor<8xf32>
    %add = arith.addf %view3, %arg0 : tensor<8xf32>

    return %view1, %view2, %add : tensor<4x2xf32>, tensor<2x2x2xf32>, tensor<8xf32>
}

// CHECK-LABEL: @test_canonicalize_broadcast
func.func @test_canonicalize_broadcast(%arg0: tensor<1x1x8xf32>, %arg1: tensor<f32>) -> (tensor<4x2x8xf32>, tensor<8x8xf32>, tensor<1x1x8xf32>) {
    %broadcast0 = tt.broadcast %arg0 : (tensor<1x1x8xf32>) -> tensor<1x2x8xf32>
    // CHECK: %{{.*}} = tt.broadcast %arg0 : (tensor<1x1x8xf32>) -> tensor<4x2x8xf32>
    %broadcast1 = tt.broadcast %broadcast0 : (tensor<1x2x8xf32>) -> tensor<4x2x8xf32>

    %splat = tt.splat %arg1 : (tensor<f32>) -> tensor<1x8xf32>
    // CHECK: %{{.*}} = tt.splat %arg1 : (tensor<f32>) -> tensor<8x8xf32>
    %broadcast2 = tt.broadcast %splat : (tensor<1x8xf32>) -> tensor<8x8xf32>

    %broadcast3 = tt.broadcast %arg0 : (tensor<1x1x8xf32>) -> tensor<1x1x8xf32>
    // CHECK: %{{.*}} = arith.addf %arg0, %arg0 : tensor<1x1x8xf32>
    %add = arith.addf %broadcast3, %arg0 : tensor<1x1x8xf32>

    return %broadcast1, %broadcast2, %add : tensor<4x2x8xf32>, tensor<8x8xf32>, tensor<1x1x8xf32>
}

// CHECK-LABEL: @test_fold_views
func.func @test_fold_views() -> (tensor<16x8xf32>, tensor<16x128xf32>, tensor<1x1x128xf32>) {
    %a = arith.constant dense<1.0> : tensor<1x128xf32>

    // CHECK-DAG: %{{.*}} = arith.constant dense<1.{{.*}}> : tensor<16x8xf32>
    %b = tt.view %a : (tensor<1x128xf32>) -> tensor<16x8xf32>

    // CHECK-DAG: %{{.*}} = arith.constant dense<1.{{.*}}> : tensor<16x128xf32>
    %c = tt.broadcast %a : (tensor<1x128xf32>) -> tensor<16x128xf32>

    // CHECK-DAG: %{{.*}} = arith.constant dense<1.{{.*}}> : tensor<1x1x128xf32>
    %d = tt.expand_dims %a {axis = 0: i32} : (tensor<1x128xf32>) -> tensor<1x1x128xf32>

    return %b, %c, %d : tensor<16x8xf32>, tensor<16x128xf32>, tensor<1x1x128xf32>
}
