// RUN: triton-opt --split-input-file %s --verify-diagnostics

tt.func public @i64_tensor() {
    // expected-error @+1 {{i32 elements}}
    %a = tt.make_range { start = 0 : i32, end = 16 : i32 } : tensor<16xi64>
    tt.return
}

// -----
tt.func public @i32_scalar() {
    // expected-error @+1 {{invalid kind of type}}
    %a = tt.make_range { start = 0 : i32, end = 16 : i32 } : i32
    tt.return
}

// -----
tt.func public @_2d_tensor() {
    // expected-error @+1 {{must be a 1D tensor}}
    %a = tt.make_range { start = 0 : i32, end = 16 : i32 } : tensor<16x1xi32>
    tt.return
}

// -----
tt.func public @bad_start_end() {
    // expected-error @+1 {{start must be less than end}}
    %a = tt.make_range { start = 0 : i32, end = -16 : i32 } : tensor<16xi32>
    tt.return
}

// -----
tt.func public @bad_num_elems() {
    // expected-error @+1 {{number of elements}}
    %a = tt.make_range { start = 0 : i32, end = 32 : i32 } : tensor<16xi32>
    tt.return
}

// -----

tt.func @same_start_end() {
  // expected-error @+1 {{'tt.make_range' op start must be less than end}}
  %0 = tt.make_range{end = 1 : i32, start = 1 : i32} : tensor<0xi32>
  tt.return
}
