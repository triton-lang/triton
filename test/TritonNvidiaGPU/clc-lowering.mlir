// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -triton-nvidia-gpu-lower-clc -verify-diagnostics=only-expected | FileCheck %s

#regs = #ttg.linear<{register = [[1]], lane = [[0], [0], [0], [0], [0]], warp = [[0], [0]], block = []}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @lower_1cta
  tt.func @lower_1cta(%pid_x0: i32, %pid_y0: i32, %pid_z0: i32, %work0: i1) {
    // CHECK: %[[RESPONSE:.*]] = ttg.local_alloc {alignment = 16 : i32} : () -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
    // CHECK: %[[BARRIER:.*]] = ttg.local_alloc
    // CHECK-NEXT: ttng.init_barrier %[[BARRIER]], 1
    // CHECK: %[[PHASE0:.*]] = arith.constant 0 : i32
    // CHECK: %[[ONE:.*]] = arith.constant 1 : i32
    // CHECK: scf.while ({{.*}} = %[[PHASE0]])
    scf.while (%pid_x = %pid_x0, %pid_y = %pid_y0, %pid_z = %pid_z0, %work = %work0) : (i32, i32, i32, i1) -> (i32, i32, i32, i1) {
      scf.condition(%work) %pid_x, %pid_y, %pid_z, %work : i32, i32, i32, i1
    } do {
    // CHECK: ^bb0({{.*}}, %[[PHASE:[A-Za-z0-9_]+]]: i32):
    ^bb0(%pid_x: i32, %pid_y: i32, %pid_z: i32, %work: i1):
      // CHECK: ttng.barrier_expect %[[BARRIER]], 16 {fromCTA = 0 : i32}, {{.*}}
      // CHECK-NEXT: ttng.clc_try_cancel %[[RESPONSE]], %[[BARRIER]]
      %response = ttg.clc_try_cancel : tensor<2xi64, #regs>
      %marker = ttg.local_alloc %response {alignment = 16 : i32} : (tensor<2xi64, #regs>) -> !ttg.memdesc<2xi64, #shared, #smem, mutable>
      // CHECK: "tile"
      "tile"(%pid_x, %pid_y, %pid_z) : (i32, i32, i32) -> ()
      // CHECK: ttng.wait_barrier %[[BARRIER]], %[[PHASE]]
      // CHECK-NEXT: %[[RAW:.*]] = ttng.clc_load_result %[[RESPONSE]]
      %raw = ttng.clc_load_result %marker : !ttg.memdesc<2xi64, #shared, #smem, mutable> -> i128
      // CHECK: ttng.clc_is_canceled %[[RAW]]
      %next_work = ttng.clc_is_canceled %raw : i128 -> i1
      %next_x = scf.if %next_work -> i32 {
        // CHECK: ttng.clc_get_program_id %[[RAW]], x
        %stolen = ttng.clc_get_program_id %raw, x : i128 -> i32
        scf.yield %stolen : i32
      } else {
        scf.yield %pid_x : i32
      }
      %next_y = scf.if %next_work -> i32 {
        // CHECK: ttng.clc_get_program_id %[[RAW]], y
        %stolen = ttng.clc_get_program_id %raw, y : i128 -> i32
        scf.yield %stolen : i32
      } else {
        scf.yield %pid_y : i32
      }
      %next_z = scf.if %next_work -> i32 {
        // CHECK: ttng.clc_get_program_id %[[RAW]], z
        %stolen = ttng.clc_get_program_id %raw, z : i128 -> i32
        scf.yield %stolen : i32
      } else {
        scf.yield %pid_z : i32
      }
      // CHECK: %[[NEXT_PHASE:.*]] = arith.xori %[[PHASE]], %[[ONE]] : i32
      // CHECK: scf.yield {{.*}}%[[NEXT_PHASE]]
      scf.yield %next_x, %next_y, %next_z, %next_work : i32, i32, i32, i1
    }
    // CHECK: ttng.inval_barrier %[[BARRIER]]
    // CHECK-NEXT: ttg.local_dealloc %[[BARRIER]]
    // CHECK-NEXT: ttg.local_dealloc %[[RESPONSE]]
    // CHECK-NOT: ttg.local_load
    tt.return
  }

}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @wrong_response_shape() {
    // expected-error @+1 {{response must have shape}}
    %response = ttg.clc_try_cancel : tensor<4xi64>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @wrong_response_element() {
    // expected-error @+1 {{response element type must be i64}}
    %response = ttg.clc_try_cancel : tensor<2xf32>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @missing_response_marker(%run: i1) {
    scf.while (%active = %run) : (i1) -> i1 {
      scf.condition(%active) %active : i1
    } do {
    ^bb0(%active: i1):
      // expected-error @+1 {{expected response to have exactly one ttg.local_alloc user}}
      %response = ttg.clc_try_cancel : tensor<2xi64>
      scf.yield %active : i1
    }
    tt.return
  }
}
