// RUN: triton-opt %s -split-input-file -triton-cpu-convert-memory-ops=use-scalar-loops=false | FileCheck %s

// Convert strided masked loads to scalar loads.

// CHECK-LABEL: @strided_masked_loads
// CHECK:       %[[COND:.+]] = vector.extract %[[MASK:.+]][[[#IDX:]]] : i1
// CHECK-NEXT:  scf.if %[[COND]] -> (vector<32xi32>) {
// CHECK-NEXT:    %[[PTR:.+]] = vector.extract %[[IN:.+]][[[#IDX]]] : i64 from vector<32xi64>
// CHECK-NEXT:    %[[PTR_:.+]] = tt.int_to_ptr %[[PTR]] : i64 -> !tt.ptr<i32>
// CHECK-NEXT:    %[[VAL:.+]] = tt.load %[[PTR_]] : !tt.ptr<i32>
// CHECK-NEXT:    %[[NEW_OUT:.+]] = vector.insert %[[VAL]], %[[OUT:.+]] [[[#IDX]]] : i32 into vector<32xi32>
// CHECK-NEXT:    scf.yield %[[NEW_OUT]] : vector<32xi32>
// CHECK-NEXT:  } else {
// CHECK-NEXT:    scf.yield %[[OUT]] : vector<32xi32>
// CHECK-NEXT:  }

module {
  tt.func public @strided_masked_loads(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) {
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<2> : tensor<32xi32>
    %cst_0 = arith.constant dense<16> : tensor<32xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = arith.cmpi slt, %0, %cst_0 : tensor<32xi32>
    %2 = arith.muli %0, %cst : tensor<32xi32>
    %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
    %4 = tt.addptr %3, %2 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    scf.for %arg1 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %5 = tt.load %4, %1 : tensor<32x!tt.ptr<i32>>
      tt.store %4, %5 : tensor<32x!tt.ptr<i32>>
    }
    tt.return
  }
}

// -----

// Convert strided masked stores to scalar stores.

// CHECK-LABEL: @strided_masked_stores
// CHECK:       %[[COND:.+]] = vector.extract %[[MASK:.+]][[[#IDX:]]] : i1 from vector<32xi1>
// CHECK-NEXT:  scf.if %[[COND]] {
// CHECK-NEXT:    %[[PTR:.+]] = vector.extract %[[OUT:.+]][[[#IDX]]] : i64 from vector<32xi64>
// CHECK-NEXT:    %[[PTR_:.+]] = tt.int_to_ptr %[[PTR]] : i64 -> !tt.ptr<i32>
// CHECK-NEXT:    %[[VAL:.+]] = vector.extract %[[IN:.+]][[[#IDX]]] : i32 from vector<32xi32>
// CHECK-NEXT:    tt.store %[[PTR_]], %[[VAL]] : !tt.ptr<i32>
// CHECK-NEXT:  }

module {
  tt.func public @strided_masked_stores(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32} ) {
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<64> : tensor<32xi32>
    %cst_0 = arith.constant dense<2> : tensor<32xi32>
    %cst_1 = arith.constant dense<16> : tensor<32xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = arith.cmpi slt, %0, %cst_1 : tensor<32xi32>
    %2 = arith.muli %0, %cst_0 : tensor<32xi32>
    %3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
    %4 = tt.addptr %3, %2 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    %5 = arith.subi %cst, %2 : tensor<32xi32>
    %6 = tt.addptr %3, %5 : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    scf.for %arg1 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %7 = tt.load %4 : tensor<32x!tt.ptr<i32>>
      tt.store %6, %7, %1 : tensor<32x!tt.ptr<i32>>
    }
    tt.return
  }
}

// -----

// Check that pointer for vector load/store is not extracted from a vector

// CHECK-LABEL: @scalar_ptrs
// CHECK-NOT:   vector.extract {{.+}} : i64 from vector<128xi64>
// CHECK:       {{.+}} = vector.load {{.+}} : memref<128xf32>, vector<128xf32>
// CHECK-NOT:   vector.extract {{.+}} : i64 from vector<128xi64>
// CHECK:       vector.store {{.+}}, {{.+}} : memref<128xf32>, vector<128xf32>

module {
  tt.func public @scalar_ptrs(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %3 = tt.load %2 : tensor<128x!tt.ptr<f32>>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %5, %3 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
