// RUN: triton-opt %s -int-range-optimizations | FileCheck %s

// CHECK-LABEL: tt.func @signed_rem
// CHECK: %[[R:.*]] = tt.make_range
// CHECK-NEXT: tt.return %[[R]]
tt.func @signed_rem() -> tensor<128xi32> {
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %c = arith.constant dense<128> : tensor<128xi32>
  %rem = arith.remsi %r, %c : tensor<128xi32>
  tt.return %rem : tensor<128xi32>
}

// CHECK-LABEL: tt.func @unsigned_rem
// CHECK: %[[R:.*]] = tt.make_range
// CHECK-NEXT: tt.return %[[R]]
tt.func @unsigned_rem() -> tensor<128xi32> {
  %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
  %c = arith.constant dense<128> : tensor<128xi32>
  %rem = arith.remui %r, %c : tensor<128xi32>
  tt.return %rem : tensor<128xi32>
}

// Including the modulus makes the remainder nontrivial.
// CHECK-LABEL: tt.func @includes_modulus
// CHECK: arith.remsi
// CHECK: arith.remui
tt.func @includes_modulus() -> (tensor<128xi32>, tensor<128xi32>) {
  %r = tt.make_range {start = 1 : i32, end = 129 : i32} : tensor<128xi32>
  %c = arith.constant dense<128> : tensor<128xi32>
  %s = arith.remsi %r, %c : tensor<128xi32>
  %u = arith.remui %r, %c : tensor<128xi32>
  tt.return %s, %u : tensor<128xi32>, tensor<128xi32>
}

// Negative i32 bit patterns can be below a high-bit unsigned modulus.
// CHECK-LABEL: tt.func @high_bit_modulus
// CHECK: %[[R:.*]] = tt.make_range
// CHECK-NEXT: tt.return %[[R]]
tt.func @high_bit_modulus() -> tensor<2xi32> {
  %r = tt.make_range {start = -4 : i32, end = -2 : i32} : tensor<2xi32>
  %c = arith.constant dense<-1> : tensor<2xi32>
  %u = arith.remui %r, %c : tensor<2xi32>
  tt.return %u : tensor<2xi32>
}

// Signed negative inputs and a disjoint unsigned range cannot be treated as
// [0, 128). In particular, the signed interval must not wrap in unsigned order.
// CHECK-LABEL: tt.func @cross_zero
// CHECK: arith.remsi
// CHECK: arith.remui
tt.func @cross_zero() -> (tensor<128xi32>, tensor<128xi32>) {
  %r = tt.make_range {start = -64 : i32, end = 64 : i32} : tensor<128xi32>
  %c = arith.constant dense<128> : tensor<128xi32>
  %s = arith.remsi %r, %c : tensor<128xi32>
  %u = arith.remui %r, %c : tensor<128xi32>
  tt.return %s, %u : tensor<128xi32>, tensor<128xi32>
}
