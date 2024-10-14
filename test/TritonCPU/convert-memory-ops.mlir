// RUN: triton-opt %s -split-input-file -triton-cpu-convert-memory-ops=use-gather-scatter=true -cse | FileCheck %s

// Convert strided masked loads to gather.

// CHECK-LABEL: @strided_masked_loads
// CHECK:       %[[PTR:.+]] = triton_cpu.ptr_to_memref %[[BASE:.+]] : <i32> -> memref<i32>
// CHECK:       %[[VAL:.+]] = vector.gather %[[PTR]][] [%[[INDEX_VEC:.+]]], %[[MASK:.+]], %[[OTHER:.+]] : memref<i32>, vector<32xi32>, vector<32xi1>, vector<32xi32> into vector<32xi32>

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

// Convert strided masked stores to scatter.

// CHECK-LABEL: @strided_masked_stores
// CHECK:       %[[PTR:.+]] = triton_cpu.ptr_to_memref %[[BASE:.+]] : <i32> -> memref<i32>
// CHECK:       vector.scatter %[[PTR]][] [%[[INDEX_VEC:.+]]], %[[MASK:.+]], %[[VALS:.+]] : memref<i32>, vector<32xi32>, vector<32xi1>, vector<32xi32>

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
