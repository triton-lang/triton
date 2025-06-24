// RUN: triton-opt %s -split-input-file -verify-diagnostics

// CHECK-LABEL: @test_preserve_basic
tt.func @test_preserve_basic(%arg0: tensor<256xf32>) {
  tt.preserve %arg0 : tensor<256xf32>
  tt.return
}

// -----

// CHECK-LABEL: @test_preserve_different_types
tt.func @test_preserve_different_types(%arg0: tensor<128xi32>, %arg1: tensor<64xf16>) {
  tt.preserve %arg0 : tensor<128xi32>
  tt.preserve %arg1 : tensor<64xf16>
  tt.return
}
