// RUN: triton-opt %s -split-input-file -triton-cpu-convert-reduction -canonicalize

// Regression test: Check that we handle consecutive calls to tt.reduce with
// different types & number of arguments.

module {
  tt.func public @triton_(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xi32>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      tt.reduce.return %arg3 : f32
    }) : (tensor<1x4xf32>) -> tensor<1xf32>
    %1:2 = "tt.reduce"(%arg0, %arg1) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: i32, %arg5: f32, %arg6: i32):
      tt.reduce.return %arg3, %arg4 : f32, i32
    }) : (tensor<1x4xf32>, tensor<1x4xi32>) -> (tensor<1xf32>, tensor<1xi32>)
    tt.return
  }
}
