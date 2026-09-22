// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -test-tritonamdgpu-range-analysis -verify-diagnostics=only-expected | FileCheck %s

// Test that user-specified PID bounds override the default range.
// Module attrs "test.pid-bound-x"=127, "test.pid-bound-y"=63 tell the test
// pass to call setPidBound(0, 127) and setPidBound(1, 63).

// CHECK-LABEL: tt.func @pid_bounds
module attributes {"ttg.num-warps" = 4 : i32, "test.pid-bound-x" = 127 : i64, "test.pid-bound-y" = 63 : i64} {
  tt.func @pid_bounds() {
    // expected-remark@+2 {{unsigned : [0, 127] signed : [0, 127]}}
    // expected-remark@+1 {{non-neg}}
    %pid_x = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 63] signed : [0, 63]}}
    // expected-remark@+1 {{non-neg}}
    %pid_y = tt.get_program_id y : i32
    tt.return
  }
}

// -----

// Test that axis without a bound still uses the default range.
// With pid-bounds=0=127,1=63, axis z should use the default.

// CHECK-LABEL: tt.func @pid_bounds_default_axis
module attributes {"ttg.num-warps" = 4 : i32, "test.pid-bound-x" = 127 : i64, "test.pid-bound-y" = 63 : i64} {
  tt.func @pid_bounds_default_axis() {
    // expected-remark@+2 {{unsigned : [0, 2147483647] signed : [0, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %pid_z = tt.get_program_id z : i32
    tt.return
  }
}

// -----

// Test that PID bounds enable folding boundary masks.
// With pid-bounds=0=127: pid_x is in [0, 127], so pid_x * 32 is in
// [0, 4064]. Adding range(0,32) gives [0, 4095], which is < 4096.

// CHECK-LABEL: tt.func @pid_bounds_fold_mask
module attributes {"ttg.num-warps" = 4 : i32, "test.pid-bound-x" = 127 : i64} {
  tt.func @pid_bounds_fold_mask() {
    // expected-remark@+2 {{unsigned : [0, 127] signed : [0, 127]}}
    // expected-remark@+1 {{non-neg}}
    %pid = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [32, 32] signed : [32, 32]}}
    // expected-remark@+1 {{non-neg}}
    %c32 = arith.constant 32 : i32
    // expected-remark@+2 {{unsigned : [0, 4064] signed : [0, 4064]}}
    // expected-remark@+1 {{non-neg}}
    %offset = arith.muli %pid, %c32 : i32
    // expected-remark@+2 {{unsigned : [0, 31] signed : [0, 31]}}
    // expected-remark@+1 {{non-neg}}
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    // expected-remark@+2 {{unsigned : [0, 4064] signed : [0, 4064]}}
    // expected-remark@+1 {{non-neg}}
    %splat = tt.splat %offset : i32 -> tensor<32xi32>
    // expected-remark@+2 {{unsigned : [0, 4095] signed : [0, 4095]}}
    // expected-remark@+1 {{non-neg}}
    %idx = arith.addi %splat, %range : tensor<32xi32>
    // expected-remark@+2 {{unsigned : [4096, 4096] signed : [4096, 4096]}}
    // expected-remark@+1 {{non-neg}}
    %c4096 = arith.constant dense<4096> : tensor<32xi32>
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %mask = arith.cmpi slt, %idx, %c4096 : tensor<32xi32>
    tt.return
  }
}
