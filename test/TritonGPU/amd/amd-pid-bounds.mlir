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

// -----

// Test per-function PID bounds: two functions in the same module with
// different PID x bounds. Per-function bounds should not leak across functions.

// CHECK-LABEL: tt.func @per_func_bounds_a
// CHECK-LABEL: tt.func @per_func_bounds_b
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @per_func_bounds_a() attributes {"test.pid-bound-x" = 31 : i64} {
    // expected-remark@+2 {{unsigned : [0, 31] signed : [0, 31]}}
    // expected-remark@+1 {{non-neg}}
    %pid_x = tt.get_program_id x : i32
    tt.return
  }

  tt.func @per_func_bounds_b() attributes {"test.pid-bound-x" = 3 : i64} {
    // expected-remark@+2 {{unsigned : [0, 3] signed : [0, 3]}}
    // expected-remark@+1 {{non-neg}}
    %pid_x = tt.get_program_id x : i32
    tt.return
  }
}

// -----

// Test that per-function bounds take precedence over global module bounds.
// Module sets global pid-bound-x=127, but func overrides to 7.

// CHECK-LABEL: tt.func @per_func_overrides_global
// CHECK-LABEL: tt.func @uses_global_bound
module attributes {"ttg.num-warps" = 4 : i32, "test.pid-bound-x" = 127 : i64} {
  tt.func @per_func_overrides_global() attributes {"test.pid-bound-x" = 7 : i64} {
    // expected-remark@+2 {{unsigned : [0, 7] signed : [0, 7]}}
    // expected-remark@+1 {{non-neg}}
    %pid_x = tt.get_program_id x : i32
    tt.return
  }

  tt.func @uses_global_bound() {
    // expected-remark@+2 {{unsigned : [0, 127] signed : [0, 127]}}
    // expected-remark@+1 {{non-neg}}
    %pid_x = tt.get_program_id x : i32
    tt.return
  }
}

// -----

// Test per-function bounds on different axes across functions.

// CHECK-LABEL: tt.func @per_func_x_bound
// CHECK-LABEL: tt.func @per_func_y_bound
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @per_func_x_bound() attributes {"test.pid-bound-x" = 15 : i64} {
    // expected-remark@+2 {{unsigned : [0, 15] signed : [0, 15]}}
    // expected-remark@+1 {{non-neg}}
    %pid_x = tt.get_program_id x : i32
    // y has no bound set — should use default
    // expected-remark@+2 {{unsigned : [0, 2147483647] signed : [0, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %pid_y = tt.get_program_id y : i32
    tt.return
  }

  tt.func @per_func_y_bound() attributes {"test.pid-bound-y" = 7 : i64} {
    // x has no bound set — should use default
    // expected-remark@+2 {{unsigned : [0, 2147483647] signed : [0, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %pid_x = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [0, 7] signed : [0, 7]}}
    // expected-remark@+1 {{non-neg}}
    %pid_y = tt.get_program_id y : i32
    tt.return
  }
}

// -----

// Test per-function bounds with full boundary mask folding.
// Two functions in the same module: func_a tiles M=1024 with N=32 (PID x
// bound 31), func_b tiles M=256 with N=64 (PID x bound 3). Each function's
// mask should fold independently using its own PID bound.

// CHECK-LABEL: tt.func @per_func_fold_mask_a
// CHECK-LABEL: tt.func @per_func_fold_mask_b
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @per_func_fold_mask_a() attributes {"test.pid-bound-x" = 31 : i64} {
    // expected-remark@+2 {{unsigned : [0, 31] signed : [0, 31]}}
    // expected-remark@+1 {{non-neg}}
    %pid = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [32, 32] signed : [32, 32]}}
    // expected-remark@+1 {{non-neg}}
    %c32 = arith.constant 32 : i32
    // expected-remark@+2 {{unsigned : [0, 992] signed : [0, 992]}}
    // expected-remark@+1 {{non-neg}}
    %offset = arith.muli %pid, %c32 : i32
    // expected-remark@+2 {{unsigned : [0, 31] signed : [0, 31]}}
    // expected-remark@+1 {{non-neg}}
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    // expected-remark@+2 {{unsigned : [0, 992] signed : [0, 992]}}
    // expected-remark@+1 {{non-neg}}
    %splat = tt.splat %offset : i32 -> tensor<32xi32>
    // expected-remark@+2 {{unsigned : [0, 1023] signed : [0, 1023]}}
    // expected-remark@+1 {{non-neg}}
    %idx = arith.addi %splat, %range : tensor<32xi32>
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024 = arith.constant dense<1024> : tensor<32xi32>
    // pid*32+[0,31] ∈ [0,1023] < 1024 → always true
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %mask = arith.cmpi slt, %idx, %c1024 : tensor<32xi32>
    tt.return
  }

  tt.func @per_func_fold_mask_b() attributes {"test.pid-bound-x" = 3 : i64} {
    // expected-remark@+2 {{unsigned : [0, 3] signed : [0, 3]}}
    // expected-remark@+1 {{non-neg}}
    %pid = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [64, 64] signed : [64, 64]}}
    // expected-remark@+1 {{non-neg}}
    %c64 = arith.constant 64 : i32
    // expected-remark@+2 {{unsigned : [0, 192] signed : [0, 192]}}
    // expected-remark@+1 {{non-neg}}
    %offset = arith.muli %pid, %c64 : i32
    // expected-remark@+2 {{unsigned : [0, 63] signed : [0, 63]}}
    // expected-remark@+1 {{non-neg}}
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    // expected-remark@+2 {{unsigned : [0, 192] signed : [0, 192]}}
    // expected-remark@+1 {{non-neg}}
    %splat = tt.splat %offset : i32 -> tensor<64xi32>
    // expected-remark@+2 {{unsigned : [0, 255] signed : [0, 255]}}
    // expected-remark@+1 {{non-neg}}
    %idx = arith.addi %splat, %range : tensor<64xi32>
    // expected-remark@+2 {{unsigned : [256, 256] signed : [256, 256]}}
    // expected-remark@+1 {{non-neg}}
    %c256 = arith.constant dense<256> : tensor<64xi32>
    // pid*64+[0,63] ∈ [0,255] < 256 → always true
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %mask = arith.cmpi slt, %idx, %c256 : tensor<64xi32>
    tt.return
  }
}
