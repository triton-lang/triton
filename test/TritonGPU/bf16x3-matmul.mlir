// RUN: triton-opt %s -tritongpu-F32DotTC="emu-tf32=0"  -canonicalize | FileCheck %s --check-prefixes=CHECK

module {
  tt.func @dot_test_BF16x3(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>, %arg2: tensor<16x16xf32>) -> tensor<16x16xf32> {
    // CHECK-LABEL: dot_test_BF16x3

    // CHECK: %[[lhs_hi:.*]] = arith.truncf %arg0
    // CHECK-NEXT: %[[val1:.*]]    = arith.extf %[[lhs_hi]]
    // CHECK-NEXT: %[[val2:.*]]    = arith.subf %arg0, %[[val1]]
    // CHECK-NEXT: %[[lhs_mid:.*]] = arith.truncf %[[val2]]

    // CHECK: %[[rhs_hi:.*]] = arith.truncf %arg1
    // CHECK-NEXT: %[[val8:.*]]    = arith.extf %[[rhs_hi]]
    // CHECK-NEXT: %[[val9:.*]]    = arith.subf %arg1, %[[val8]]
    // CHECK-NEXT: %[[rhs_mid:.*]] = arith.truncf %[[val9]]

    // CHECK-NEXT: %[[val20:.*]] = tt.dot %[[lhs_mid]], %[[rhs_hi]]
    // CHECK-NEXT: %[[val21:.*]] = tt.dot %[[lhs_hi]],  %[[rhs_mid]], %[[val20]]

    // CHECK: %[[val22:.*]] = arith.cmpf uno, %[[val21]], %[[val21]]
    // CHECK-NEXT: %[[val23:.*]] = arith.select %[[val22]]

    // CHECK: %[[val24:.*]] = tt.dot %[[lhs_hi]], %[[rhs_hi]], %[[val23]]
    // CHECK-NEXT: %[[val25:.*]] = arith.addf %[[val24]], %arg2

    %4 = tt.dot %arg0, %arg1, %arg2, inputPrecision = bf16x3 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    tt.return %4 : tensor<16x16xf32>
  }

  tt.func @dot_test_BF16x6(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>, %arg2: tensor<16x16xf32>) -> tensor<16x16xf32> {
    // CHECK-LABEL: dot_test_BF16x6

    // CHECK: %[[lhs_hi:.*]] = arith.truncf %arg0
    // CHECK-NEXT: %[[val1:.*]]    = arith.extf %[[lhs_hi]]
    // CHECK-NEXT: %[[val2:.*]]    = arith.subf %arg0, %[[val1]]
    // CHECK-NEXT: %[[lhs_mid:.*]] = arith.truncf %[[val2]]
    // CHECK-NEXT: %[[val4:.*]]    = arith.extf %[[lhs_mid]]
    // CHECK-NEXT: %[[val5:.*]]    = arith.subf %[[val2]], %[[val4]]
    // CHECK-NEXT: %[[lhs_lo:.*]]  = arith.truncf %[[val5]]

    // CHECK: %[[rhs_hi:.*]] = arith.truncf %arg1
    // CHECK-NEXT: %[[val8:.*]]    = arith.extf %[[rhs_hi]]
    // CHECK-NEXT: %[[val9:.*]]    = arith.subf %arg1, %[[val8]]
    // CHECK-NEXT: %[[rhs_mid:.*]] = arith.truncf %[[val9]]
    // CHECK-NEXT: %[[val11:.*]]   = arith.extf %[[rhs_mid]]
    // CHECK-NEXT: %[[val12:.*]]   = arith.subf %[[val9]], %[[val11]]
    // CHECK-NEXT: %[[rhs_lo:.*]]  = arith.truncf %[[val12]]

    // CHECK: %[[val17:.*]] = tt.dot %[[lhs_mid]], %[[rhs_mid]]
    // CHECK-NEXT: %[[val18:.*]] = tt.dot %[[lhs_lo]],  %[[rhs_hi]],  %[[val17]]
    // CHECK-NEXT: %[[val19:.*]] = tt.dot %[[lhs_hi]],  %[[rhs_lo]],  %[[val18]]
    // CHECK-NEXT: %[[val20:.*]] = tt.dot %[[lhs_mid]], %[[rhs_hi]],  %[[val19]]
    // CHECK-NEXT: %[[val21:.*]] = tt.dot %[[lhs_hi]],  %[[rhs_mid]], %[[val20]]

    // CHECK: %[[val22:.*]] = arith.cmpf uno, %[[val21]], %[[val21]]
    // CHECK-NEXT: %[[val23:.*]] = arith.select %[[val22]]

    // CHECK: %[[val24:.*]] = tt.dot %[[lhs_hi]], %[[rhs_hi]], %[[val23]]
    // CHECK-NEXT: %[[val25:.*]] = arith.addf %[[val24]], %arg2

    %4 = tt.dot %arg0, %arg1, %arg2, inputPrecision = bf16x6 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    tt.return %4 : tensor<16x16xf32>
  }
}
