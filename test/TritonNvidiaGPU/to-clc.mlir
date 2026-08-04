// RUN: triton-opt %s -split-input-file -triton-nvidia-gpu-to-clc -verify-diagnostics=only-expected | FileCheck %s

// CHECK: #{{.*}} = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK: #{{.*}} = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func public @one_axis
  tt.func public @one_axis(%out: !tt.ptr<i32>) {
    // CHECK-COUNT-1: tt.get_program_id x
    %pid = tt.get_program_id x : i32
    // CHECK: scf.while
    // CHECK: scf.condition
    // CHECK: %[[RESPONSE:.*]] = ttng.clc_try_cancel_sync : tensor<2xi64, #[[REG_1CTA:[A-Za-z0-9_]+]]>
    // CHECK-NEXT: %[[MARKER:.*]] = ttg.local_alloc %[[RESPONSE]] {alignment = 16 : i32} : (tensor<2xi64, #[[REG_1CTA]]>) -> !ttg.memdesc<2xi64, #[[SHARED_1CTA:[A-Za-z0-9_]+]], {{.*}}, mutable>
    // CHECK: tt.get_num_programs x
    %num = tt.get_num_programs x : i32
    %value = arith.addi %pid, %num : i32
    %ptr = tt.addptr %out, %pid : !tt.ptr<i32>, i32
    // CHECK: tt.store
    tt.store %ptr, %value : !tt.ptr<i32>
    // CHECK: %[[RAW:.*]] = ttng.clc_load_result %[[MARKER]] : !ttg.memdesc<2xi64, #[[SHARED_1CTA]], {{.*}}, mutable> -> i128
    // CHECK: ttng.clc_is_canceled %[[RAW]]
    // CHECK: scf.if
    // CHECK: ttng.clc_get_program_id %[[RAW]], x
    // CHECK-COUNT-1: tt.return
    tt.return
  }
}

// -----

// CHECK: #{{.*}} = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = {{\[\[0\]\]}}}>
// CHECK: #{{.*}} = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = {{\[\[0\]\]}}}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func public @repeated_axis
  tt.func public @repeated_axis(%out: !tt.ptr<i32>) {
    // CHECK-COUNT-1: tt.get_program_id x
    %pid0 = tt.get_program_id x : i32
    // CHECK: ^bb0(%[[PID:.*]]: i32, %{{.*}}: i1):
    // CHECK: %[[RESPONSE_2CTA:.*]] = ttng.clc_try_cancel_sync : tensor<2xi64, #[[REG_2CTA:[A-Za-z0-9_]+]]>
    // CHECK-NEXT: %[[MARKER_2CTA:.*]] = ttg.local_alloc %[[RESPONSE_2CTA]] {alignment = 16 : i32} : (tensor<2xi64, #[[REG_2CTA]]>) -> !ttg.memdesc<2xi64, #[[SHARED_2CTA:[A-Za-z0-9_]+]], {{.*}}, mutable>
    %pid1 = tt.get_program_id x : i32
    // CHECK: arith.addi %[[PID]], %[[PID]]
    %sum = arith.addi %pid0, %pid1 : i32
    %ptr = tt.addptr %out, %pid0 : !tt.ptr<i32>, i32
    tt.store %ptr, %sum : !tt.ptr<i32>
    // CHECK: %[[RAW_2CTA:.*]] = ttng.clc_load_result %[[MARKER_2CTA]] : !ttg.memdesc<2xi64, #[[SHARED_2CTA]], {{.*}}, mutable> -> i128
    // CHECK-COUNT-1: ttng.clc_get_program_id {{.*}}, x
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func public @nested_xyz
  tt.func public @nested_xyz(%out: !tt.ptr<i32>, %cond: i1) {
    // CHECK-COUNT-1: tt.get_program_id x
    // CHECK-COUNT-1: tt.get_program_id y
    // CHECK-COUNT-1: tt.get_program_id z
    // CHECK: scf.while
    // CHECK: %[[RESPONSE:.*]] = ttng.clc_try_cancel_sync
    // CHECK-NEXT: %[[MARKER:.*]] = ttg.local_alloc %[[RESPONSE]] {alignment = 16 : i32}
    %pid_x = tt.get_program_id x : i32
    %ptr_x = tt.addptr %out, %pid_x : !tt.ptr<i32>, i32
    tt.store %ptr_x, %pid_x : !tt.ptr<i32>
    // CHECK: scf.if
    scf.if %cond {
      %pid_y = tt.get_program_id y : i32
      %ptr_y = tt.addptr %out, %pid_y : !tt.ptr<i32>, i32
      tt.store %ptr_y, %pid_y : !tt.ptr<i32>
    }
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // CHECK: scf.for
    scf.for %i = %c0 to %c1 step %c1 {
      %pid_z = tt.get_program_id z : i32
      %ptr_z = tt.addptr %out, %pid_z : !tt.ptr<i32>, i32
      tt.store %ptr_z, %pid_z : !tt.ptr<i32>
    }
    // CHECK: %[[RAW:.*]] = ttng.clc_load_result %[[MARKER]]
    // CHECK: ttng.clc_get_program_id %[[RAW]], x
    // CHECK: ttng.clc_get_program_id %[[RAW]], y
    // CHECK: ttng.clc_get_program_id %[[RAW]], z
    // CHECK: scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}} : i32, i32, i32, i1
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // expected-error @+1 {{CLC conversion requires at least one tt.get_program_id}}
  tt.func public @no_pid(%out: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    tt.store %out, %c0 : !tt.ptr<i32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func public @first_kernel
  tt.func public @first_kernel(%out: !tt.ptr<i32>) {
    // CHECK: scf.while
    // CHECK: ttng.clc_try_cancel_sync
    %pid = tt.get_program_id x : i32
    %ptr = tt.addptr %out, %pid : !tt.ptr<i32>, i32
    tt.store %ptr, %pid : !tt.ptr<i32>
    tt.return
  }

  // CHECK-LABEL: tt.func public @second_kernel
  tt.func public @second_kernel(%out: !tt.ptr<i32>) {
    // CHECK: scf.while
    // CHECK: ttng.clc_try_cancel_sync
    %pid = tt.get_program_id y : i32
    %ptr = tt.addptr %out, %pid : !tt.ptr<i32>, i32
    tt.store %ptr, %pid : !tt.ptr<i32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func private @pid_helper() {
    // expected-error @+1 {{CLC conversion requires all tt.get_program_id operations to be in the root function of this kernel after inlining}}
    %pid = tt.get_program_id y : i32
    tt.return
  }

  tt.func public @residual_pid(%out: !tt.ptr<i32>) {
    %pid = tt.get_program_id x : i32
    %ptr = tt.addptr %out, %pid : !tt.ptr<i32>, i32
    tt.store %ptr, %pid : !tt.ptr<i32>
    tt.return
  }
}
