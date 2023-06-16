// RUN: triton-opt --triton-to-linalg %s
// XFAIL: *
// Disable this test since we do not support scalar loads at the moment.

module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32
  )
  {
    %0 = tt.addptr %arg0, %arg2 : !tt.ptr<bf16>, i32
    %1 = tt.addptr %arg1, %arg2 : !tt.ptr<bf16>, i32

    // expected-error @below {{Scalar load is currently not supported}}
    %10 = tt.load %0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: bf16
    tt.store %1, %10 : bf16
  tt.return
  }
}
