// RUN: triton-opt %s --triton-cpu-lower-sve | FileCheck %s

module {
  // CHECK-LABEL: tt.func @transpose_rows_to_columns(
  // CHECK-NOT: vector.transpose
  // CHECK: vector.vscale
  // CHECK: scf.for
  // CHECK: vector.maskedload {{.*}} : memref<74xf32>, vector<[8]xi1>, vector<[8]xf32>
  // CHECK-NEXT: %[[EVEN:.*]], %[[ODD:.*]] = vector.deinterleave {{.*}} : vector<[8]xf32> -> vector<[4]xf32>
  // CHECK: vector.maskedstore {{.*}}, %[[EVEN]] : memref<74xf32>, vector<[4]xi1>, vector<[4]xf32>
  // CHECK: vector.maskedstore {{.*}}, %[[ODD]] : memref<74xf32>, vector<[4]xi1>, vector<[4]xf32>
  // CHECK: vector.shape_cast {{.*}} : vector<74xf32> to vector<2x37xf32>
  tt.func @transpose_rows_to_columns(%arg: vector<37x2xf32>) -> vector<2x37xf32> {
    %result = vector.transpose %arg, [1, 0] : vector<37x2xf32> to vector<2x37xf32>
    tt.return %result : vector<2x37xf32>
  }

  // CHECK-LABEL: tt.func @transpose_columns_to_rows(
  // CHECK-NOT: vector.transpose
  // CHECK: vector.vscale
  // CHECK: scf.for
  // CHECK: %[[LHS:.*]] = vector.maskedload {{.*}} : memref<74xf32>, vector<[4]xi1>, vector<[4]xf32>
  // CHECK: %[[RHS:.*]] = vector.maskedload {{.*}} : memref<74xf32>, vector<[4]xi1>, vector<[4]xf32>
  // CHECK: %[[JOINED:.*]] = vector.interleave %[[LHS]], %[[RHS]] : vector<[4]xf32> -> vector<[8]xf32>
  // CHECK: vector.maskedstore {{.*}}, %[[JOINED]] : memref<74xf32>, vector<[8]xi1>, vector<[8]xf32>
  tt.func @transpose_columns_to_rows(%arg: vector<2x37xf32>) -> vector<37x2xf32> {
    %result = vector.transpose %arg, [1, 0] : vector<2x37xf32> to vector<37x2xf32>
    tt.return %result : vector<37x2xf32>
  }

  // CHECK-LABEL: tt.func @unsupported_shape(
  // CHECK: vector.transpose
  tt.func @unsupported_shape(%arg: vector<8x3xf32>) -> vector<3x8xf32> {
    %result = vector.transpose %arg, [1, 0] : vector<8x3xf32> to vector<3x8xf32>
    tt.return %result : vector<3x8xf32>
  }
}
