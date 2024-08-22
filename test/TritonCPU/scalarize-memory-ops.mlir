// RUN: triton-opt %s -split-input-file -triton-cpu-convert-memory-ops=use-scalar-loops=true -cse -canonicalize | FileCheck %s

// Convert strided masked load and store to loops. Pointer and mask should be scalarized.
// TODO: There is an optimization opportunity to fuse loops.
// TODO: There is an optimization opportunity to reuse temp buffers.

// CHECK-LABEL: @strided_masked_load_store
// CHECK:       %[[ALLOCA1:.*]] = memref.alloca() {alignment = 64 : i64} : memref<128xf32>
// CHECK-NEXT:  scf.for %[[IV1:.*]] = %c0 to %c128 step %c1 {
// CHECK-NEXT:    %[[IV1_I32:.*]] = arith.index_castui %[[IV1]] : index to i32
// CHECK-NEXT:    %[[IDX1:.*]] = arith.muli %[[IV1_I32]], %c3_i32 : i32
// CHECK-NEXT:    %[[PTR1:.*]] = tt.addptr %arg0, %[[IDX1]] : !tt.ptr<f32>, i32
// CHECK-NEXT:    %[[MASK1:.*]] = arith.cmpi slt, %[[IDX1]], %arg2 : i32
// CHECK-NEXT:    scf.if %[[MASK1]] {
// CHECK-NEXT:      %[[VAL1:.*]] = tt.load %[[PTR1]] : !tt.ptr<f32>
// CHECK-NEXT:      memref.store %[[VAL1]], %[[ALLOCA1]][%[[IV1]]] : memref<128xf32>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      memref.store %{{.*}}, %[[ALLOCA1]][%[[IV1]]] : memref<128xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[VEC_VAL:.*]] = vector.transfer_read %[[ALLOCA1]][%c0], %{{.*}} {in_bounds = [true]} : memref<128xf32>, vector<128xf32>
// CHECK-NEXT:  %[[ALLOCA2:.*]] = memref.alloca() {alignment = 64 : i64} : memref<128xf32>
// CHECK-NEXT:  vector.transfer_write %[[VEC_VAL]], %[[ALLOCA2]][%c0] {in_bounds = [true]} : vector<128xf32>, memref<128xf32>
// CHECK-NEXT:  scf.for %[[IV2:.*]] = %c0 to %c128 step %c1 {
// CHECK-NEXT:    %[[IV2_I32:.*]] = arith.index_castui %[[IV2]] : index to i32
// CHECK-NEXT:    %[[IDX2:.*]] = arith.muli %[[IV2_I32]], %c3_i32 : i32
// CHECK-NEXT:    %[[PTR2:.*]] = tt.addptr %arg1, %[[IDX2]] : !tt.ptr<f32>, i32
// CHECK-NEXT:    %[[MASK2:.*]] = arith.cmpi slt, %[[IDX2]], %arg2 : i32
// CHECK-NEXT:    %[[VAL2:.*]] = memref.load %[[ALLOCA2]][%[[IV2]]] : memref<128xf32>
// CHECK-NEXT:    scf.if %[[MASK2]] {
// CHECK-NEXT:      tt.store %[[PTR2]], %[[VAL2]] : !tt.ptr<f32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

module {
  tt.func public @strided_masked_load_store(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32>
    %cst_0 = arith.constant dense<3> : tensor<128xi32>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = arith.muli %0, %cst_0 : tensor<128xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %3 = arith.cmpi slt, %1, %2 : tensor<128xi32>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %5 = tt.addptr %4, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %6 = tt.load %5, %3, %cst : tensor<128x!tt.ptr<f32>>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %8 = tt.addptr %7, %1 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %8, %6, %3 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// Convert indirect masked load and store. Pointer and mask are bufferized.
// TODO: There is an optimization opportunity to fuse loops.
// TODO: There is an optimization opportunity to reuse temp buffers.

// CHECK-LABEL: @indirect_masked_load_store
// CHECK:       %[[ALLOCA_VALS1:.*]] = memref.alloca() {alignment = 64 : i64} : memref<128xf32>
// CHECK-NEXT:  %[[ALLOCA_PTRS1:.*]] = memref.alloca() {alignment = 64 : i64} : memref<128xi64>
// CHECK-NEXT:  vector.transfer_write %{{.*}}, %[[ALLOCA_PTRS1]][%c0] {in_bounds = [true]} : vector<128xi64>, memref<128xi64>
// CHECK-NEXT:  %[[EXT_MASK:.*]] = arith.extui %{{.*}} : vector<128xi1> to vector<128xi8>
// CHECK-NEXT:  %[[ALLOCA_MASK1:.*]] = memref.alloca() {alignment = 64 : i64} : memref<128xi8>
// CHECK-NEXT:  vector.transfer_write %[[EXT_MASK]], %[[ALLOCA_MASK1]][%c0] {in_bounds = [true]} : vector<128xi8>, memref<128xi8>
// CHECK-NEXT:  scf.for %[[IV1:.*]] = %c0 to %c128 step %c1 {
// CHECK-NEXT:    %[[PTR1_INT:.*]] = memref.load %[[ALLOCA_PTRS1]][%[[IV1]]] : memref<128xi64>
// CHECK-NEXT:    %[[PTR1:.*]] = tt.int_to_ptr %[[PTR1_INT]] : i64 -> !tt.ptr<f32>
// CHECK-NEXT:    %[[MASK1_I8:.*]] = memref.load %[[ALLOCA_MASK1]][%[[IV1]]] : memref<128xi8>
// CHECK-NEXT:    %[[MASK1:.*]] = arith.trunci %[[MASK1_I8]] : i8 to i1
// CHECK-NEXT:    scf.if %[[MASK1]] {
// CHECK-NEXT:      %[[VAL1:.*]] = tt.load %[[PTR1]] : !tt.ptr<f32>
// CHECK-NEXT:      memref.store %[[VAL1]], %[[ALLOCA_VALS1]][%[[IV1]]] : memref<128xf32>
// CHECK-NEXT:    } else {
// CHECK-NEXT:      memref.store %{{.*}}, %[[ALLOCA_VALS1]][%[[IV1]]] : memref<128xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[VEC_VAL:.*]] = vector.transfer_read %[[ALLOCA_VALS1]][%c0], %{{.*}} {in_bounds = [true]} : memref<128xf32>, vector<128xf32>
// CHECK:       %[[ALLOCA_PTRS2:.*]] = memref.alloca() {alignment = 64 : i64} : memref<128xi64>
// CHECK-NEXT:  vector.transfer_write %{{.*}}, %[[ALLOCA_PTRS2]][%c0] {in_bounds = [true]} : vector<128xi64>, memref<128xi64>
// CHECK-NEXT:  %[[ALLOCA_MASK2:.*]] = memref.alloca() {alignment = 64 : i64} : memref<128xi8>
// CHECK-NEXT:  vector.transfer_write %[[EXT_MASK]], %[[ALLOCA_MASK2]][%c0] {in_bounds = [true]} : vector<128xi8>, memref<128xi8>
// CHECK-NEXT:  %[[ALLOCA_VALS2:.*]] = memref.alloca() {alignment = 64 : i64} : memref<128xf32>
// CHECK-NEXT:  vector.transfer_write %[[VEC_VAL]], %[[ALLOCA_VALS2]][%c0] {in_bounds = [true]} : vector<128xf32>, memref<128xf32>
// CHECK-NEXT:  scf.for %[[IV2:.*]] = %c0 to %c128 step %c1 {
// CHECK-NEXT:    %[[PTR2_INT:.*]] = memref.load %[[ALLOCA_PTRS2]][%[[IV2]]] : memref<128xi64>
// CHECK-NEXT:    %[[PTR2:.*]] = tt.int_to_ptr %[[PTR1_INT]] : i64 -> !tt.ptr<f32>
// CHECK-NEXT:    %[[MASK2_I8:.*]] = memref.load %[[ALLOCA_MASK2]][%[[IV2]]] : memref<128xi8>
// CHECK-NEXT:    %[[MASK2:.*]] = arith.trunci %[[MASK2_I8]] : i8 to i1
// CHECK-NEXT:    %[[VAL2:.*]] = memref.load %[[ALLOCA_VALS2]][%[[IV2]]] : memref<128xf32>
// CHECK-NEXT:    scf.if %[[MASK2]] {
// CHECK-NEXT:      tt.store %[[PTR2]], %[[VAL2]] : !tt.ptr<f32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

module {
  tt.func public @indirect_masked_load_store(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<i32>, %arg3: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128xf32>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %3 = tt.load %2 : tensor<128x!tt.ptr<i32>>
    %4 = tt.splat %arg3 : i32 -> tensor<128xi32>
    %5 = arith.cmpi slt, %3, %4 : tensor<128xi32>
    %6 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %7 = tt.addptr %6, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %8 = tt.load %7, %5, %cst : tensor<128x!tt.ptr<f32>>
    %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %10 = tt.addptr %9, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %10, %8, %5 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
