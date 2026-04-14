// RUN: %triton-opt %s --triton-rewrite-tensor-descriptor-to-pointer -split-input-file | %FileCheck %s

// Tests for in-bounds optimization in RewriteTensorDescriptorToPointerPass.
// When tensor descriptor accesses are statically provable to be in-bounds
// (offset[i] + blockShape[i] <= shape[i] for all dims), the pass skips mask
// generation to avoid unnecessary masked memory operations.

// Test 1: In-bounds 2D load with zero offset
// Tensor shape is 256x256 and block shape is 256x256, offset is (0,0).
// Since 0+256 <= 256 for both dims, the access is fully in-bounds.
// The pass should emit an unmasked tt.load (no arith.cmpi bounds checks).

// CHECK-LABEL: in_bounds_load_static_zero_offset
// CHECK-NOT:   arith.cmpi
// CHECK:       tt.load {{%.+}} : tensor<256x256x!tt.ptr<f32>>
// CHECK-NOT:   arith.cmpi
tt.func public @in_bounds_load_static_zero_offset(
    %arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}) {
  %c256_i32 = arith.constant 256 : i32
  %c256_i64 = arith.constant 256 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <tensor<256x256xf32>>
  %result = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<256x256xf32>> -> tensor<256x256xf32>
  tt.return
}

// -----

// Test 2: In-bounds 2D store with zero offset
// Same as Test 1 but for stores. The pass should emit an unmasked tt.store.

// CHECK-LABEL: in_bounds_store_static_zero_offset
// CHECK-NOT:   arith.cmpi
// CHECK:       tt.store {{%.+}}, {{%.+}} : tensor<256x256x!tt.ptr<f32>>
// CHECK-NOT:   arith.cmpi
tt.func public @in_bounds_store_static_zero_offset(
    %arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32},
    %data: tensor<256x256xf32>) {
  %c256_i32 = arith.constant 256 : i32
  %c256_i64 = arith.constant 256 : i64
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <tensor<256x256xf32>>
  tt.descriptor_store %desc[%c0_i32, %c0_i32], %data : !tt.tensordesc<tensor<256x256xf32>>, tensor<256x256xf32>
  tt.return
}

// -----

// Test 3: In-bounds 2D load with nonzero offset
// Block shape is 128x128, tensor shape is 256x256, offset is (64, 0).
// 64+128=192 <= 256 and 0+128=128 <= 256, so the access is in-bounds.
// The pass should emit an unmasked tt.load.

// CHECK-LABEL: in_bounds_load_nonzero_offset
// CHECK-NOT:   arith.cmpi
// CHECK:       tt.load {{%.+}} : tensor<128x128x!tt.ptr<f32>>
// CHECK-NOT:   arith.cmpi
tt.func public @in_bounds_load_nonzero_offset(
    %arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}) {
  %c256_i32 = arith.constant 256 : i32
  %c256_i64 = arith.constant 256 : i64
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c0_i32 = arith.constant 0 : i32
  %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <tensor<128x128xf32>>
  %result = tt.descriptor_load %desc[%c64_i32, %c0_i32] : !tt.tensordesc<tensor<128x128xf32>> -> tensor<128x128xf32>
  tt.return
}

// -----

// Test 4: Out-of-bounds load with dynamic shape
// Shape is a dynamic function argument (%arg1), so we cannot prove in-bounds
// at compile time. The pass must conservatively generate a mask.

// CHECK-LABEL: out_of_bounds_load_dynamic_shape
// CHECK:       arith.cmpi
// CHECK:       tt.load {{%.+}}, {{%.+}}, {{%.+}} : tensor<32x32x!tt.ptr<f32>>
tt.func public @out_of_bounds_load_dynamic_shape(
    %arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32},
    %arg1: i32) {
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %stride = arith.extsi %arg1 : i32 to i64
  %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg1], [%stride, %c1_i64] : <f32>, <tensor<32x32xf32>>
  %result = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<32x32xf32>> -> tensor<32x32xf32>
  tt.return
}

// -----

// Test 5: Out-of-bounds load with dynamic offset
// Offset is a dynamic function argument (%arg1), so we cannot prove in-bounds.
// The pass must conservatively generate a mask.

// CHECK-LABEL: out_of_bounds_load_dynamic_offset
// CHECK:       arith.cmpi
// CHECK:       tt.load {{%.+}}, {{%.+}}, {{%.+}} : tensor<128x128x!tt.ptr<f32>>
tt.func public @out_of_bounds_load_dynamic_offset(
    %arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32},
    %arg1: i32) {
  %c256_i32 = arith.constant 256 : i32
  %c256_i64 = arith.constant 256 : i64
  %c1_i64 = arith.constant 1 : i64
  %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <tensor<128x128xf32>>
  %result = tt.descriptor_load %desc[%arg1, %arg1] : !tt.tensordesc<tensor<128x128xf32>> -> tensor<128x128xf32>
  tt.return
}

// -----

// Test 6: Out-of-bounds load where offset+block exceeds shape
// Block shape is 256x256, offset is (64, 64), shape is 256x256.
// 64+256=320 > 256, so the access is out-of-bounds. Must generate mask.

// CHECK-LABEL: out_of_bounds_load_block_exceeds_shape
// CHECK:       arith.cmpi
// CHECK:       tt.load {{%.+}}, {{%.+}}, {{%.+}} : tensor<256x256x!tt.ptr<f32>>
tt.func public @out_of_bounds_load_block_exceeds_shape(
    %arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}) {
  %c256_i32 = arith.constant 256 : i32
  %c256_i64 = arith.constant 256 : i64
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %desc = tt.make_tensor_descriptor %arg0, [%c256_i32, %c256_i32], [%c256_i64, %c1_i64] : <f32>, <tensor<256x256xf32>>
  %result = tt.descriptor_load %desc[%c64_i32, %c64_i32] : !tt.tensordesc<tensor<256x256xf32>> -> tensor<256x256xf32>
  tt.return
}

// -----

// Test 7: In-bounds 1D load
// Block shape is 64, offset is 0, shape is 64. 0+64 <= 64 -> in-bounds.
// The pass should emit an unmasked tt.load.

// CHECK-LABEL: in_bounds_1d_load
// CHECK-NOT:   arith.cmpi
// CHECK:       tt.load {{%.+}} : tensor<64x!tt.ptr<f32>>
// CHECK-NOT:   arith.cmpi
tt.func public @in_bounds_1d_load(
    %arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32}) {
  %c64_i32 = arith.constant 64 : i32
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %desc = tt.make_tensor_descriptor %arg0, [%c64_i32], [%c1_i64] : <f32>, <tensor<64xf32>>
  %result = tt.descriptor_load %desc[%c0_i32] : !tt.tensordesc<tensor<64xf32>> -> tensor<64xf32>
  tt.return
}

// -----

// Test 8: Out-of-bounds store with dynamic shape
// Shape is dynamic, so the pass must conservatively generate a mask.

// CHECK-LABEL: out_of_bounds_store_dynamic_shape
// CHECK:       arith.cmpi
// CHECK:       tt.store {{%.+}}, {{%.+}}, {{%.+}} : tensor<32x32x!tt.ptr<f32>>
tt.func public @out_of_bounds_store_dynamic_shape(
    %arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32},
    %arg1: i32,
    %data: tensor<32x32xf32>) {
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %stride = arith.extsi %arg1 : i32 to i64
  %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg1], [%stride, %c1_i64] : <f32>, <tensor<32x32xf32>>
  tt.descriptor_store %desc[%c0_i32, %c0_i32], %data : !tt.tensordesc<tensor<32x32xf32>>, tensor<32x32xf32>
  tt.return
}

// -----

// Test 9: skip_boundary_check attribute on load with dynamic shape
// Even though shapes are dynamic, skip_boundary_check=true tells the pass
// to skip mask generation (the caller guarantees the access is in-bounds).

// CHECK-LABEL: skip_boundary_check_load_dynamic
// CHECK-NOT:   arith.cmpi
// CHECK:       tt.load {{%.+}} : tensor<32x32x!tt.ptr<f32>>
// CHECK-NOT:   arith.cmpi
tt.func public @skip_boundary_check_load_dynamic(
    %arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32},
    %arg1: i32) {
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %stride = arith.extsi %arg1 : i32 to i64
  %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg1], [%stride, %c1_i64] : <f32>, <tensor<32x32xf32>>
  %result = tt.descriptor_load %desc[%c0_i32, %c0_i32] {skip_boundary_check} : !tt.tensordesc<tensor<32x32xf32>> -> tensor<32x32xf32>
  tt.return
}

// -----

// Test 10: skip_boundary_check attribute on store with dynamic shape
// Same as Test 9 but for stores.

// CHECK-LABEL: skip_boundary_check_store_dynamic
// CHECK-NOT:   arith.cmpi
// CHECK:       tt.store {{%.+}}, {{%.+}} {skip_boundary_check} : tensor<32x32x!tt.ptr<f32>>
// CHECK-NOT:   arith.cmpi
tt.func public @skip_boundary_check_store_dynamic(
    %arg0: !tt.ptr<f32> {tt.divisibility = 32 : i32},
    %arg1: i32,
    %data: tensor<32x32xf32>) {
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %stride = arith.extsi %arg1 : i32 to i64
  %desc = tt.make_tensor_descriptor %arg0, [%arg1, %arg1], [%stride, %c1_i64] : <f32>, <tensor<32x32xf32>>
  tt.descriptor_store %desc[%c0_i32, %c0_i32], %data {skip_boundary_check} : !tt.tensordesc<tensor<32x32xf32>>, tensor<32x32xf32>
  tt.return
}
