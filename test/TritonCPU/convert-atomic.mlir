// RUN: triton-opt %s -split-input-file -triton-cpu-convert-atomic-ops | FileCheck %s

// Convert atomic ops with non-constant masks into scf.if + maskless atomic op.
// Check that the final tt.atomic_rmw only has 5 parameters (the 6th would be the mask).

// CHECK-LABEL: @atomic_mask
// CHECK:       %[[COND:.+]] = vector.extract %{{.+}}[[[#IDX:]]] : i1 from vector<16xi1>
// CHECK-NEXT:  scf.if %[[COND]] -> (f32) {
// CHECK-NEXT:    %[[OLD:.+]] = tt.atomic_rmw fadd, acq_rel, gpu, %{{[^%]+}} %{{[^%]+}} : (!tt.ptr<f32>, f32) -> f32
// CHECK-NEXT:    scf.yield %[[OLD]] : f32
// CHECK-NEXT:  } else {
// CHECK-NEXT:    %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    scf.yield %[[CST]] : f32
// CHECK-NEXT:  }

module {
  tt.func public @atomic_mask(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<[0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]> : vector<16xi64>
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant dense<5.000000e-01> : vector<16xf32>
    %cst_1 = arith.constant dense<3.000000e+00> : vector<16xf32>
    %0 = builtin.unrealized_conversion_cast %cst_1 : vector<16xf32> to tensor<16xf32>
    %1 = tt.ptr_to_int %arg0 : !tt.ptr<f32> -> i64
    %2 = vector.splat %1 : vector<16xi64>
    %3 = arith.addi %2, %cst : vector<16xi64>
    %4 = builtin.unrealized_conversion_cast %3 : vector<16xi64> to tensor<16x!tt.ptr<f32>>
    %5 = vector.extract %3[0] : i64 from vector<16xi64>
    %6 = tt.int_to_ptr %5 : i64 -> !tt.ptr<f32>
    %7 = triton_cpu.ptr_to_memref %6 : <f32> -> memref<16xf32>
    %8 = vector.load %7[%c0] : memref<16xf32>, vector<16xf32>
    %9 = arith.cmpf olt, %8, %cst_0 : vector<16xf32>
    %10 = builtin.unrealized_conversion_cast %9 : vector<16xi1> to tensor<16xi1>
    %11 = tt.atomic_rmw fadd, acq_rel, gpu, %4, %0, %10 : (tensor<16x!tt.ptr<f32>>, tensor<16xf32>, tensor<16xi1>) -> tensor<16xf32>
    tt.return
  }
}
