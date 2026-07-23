// RUN: triton-opt %s -split-input-file -tritongpu-normalize-ws-while-loops | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @move_ws_attributes_with_clc
  tt.func @move_ws_attributes_with_clc(%run: i1, %lb: i32, %ub: i32,
                                        %step: i32) {
    // CHECK: scf.while
    %loop = scf.while (%active = %run) : (i1) -> i1 {
      scf.condition(%active) %active : i1
    } do {
    ^bb0(%active: i1):
      // CHECK: ttng.clc_try_cancel_sync
      %response = ttng.clc_try_cancel_sync : tensor<2xi64, #blocked>
      // CHECK: scf.for
      // CHECK-NEXT: }{{$}}
      scf.for %i = %lb to %ub step %step : i32 {
        scf.yield
      } {tt.disallow_acc_multi_buffer, tt.warp_specialize}
      // CHECK: scf.for
      // CHECK-NEXT: }{{$}}
      scf.for %i = %lb to %ub step %step : i32 {
        scf.yield
      } {tt.warp_specialize}
      scf.yield %active : i1
    // CHECK: } attributes {tt.disallow_acc_multi_buffer, tt.warp_specialize}
    }
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @keep_ws_attributes_without_clc
  tt.func @keep_ws_attributes_without_clc(%run: i1, %lb: i32, %ub: i32,
                                           %step: i32) {
    // CHECK: scf.while
    %loop = scf.while (%active = %run) : (i1) -> i1 {
      scf.condition(%active) %active : i1
    } do {
    ^bb0(%active: i1):
      // CHECK: scf.for
      // CHECK-NEXT: } {tt.disallow_acc_multi_buffer, tt.warp_specialize}
      scf.for %i = %lb to %ub step %step : i32 {
        scf.yield
      } {tt.disallow_acc_multi_buffer, tt.warp_specialize}
      scf.yield %active : i1
    // CHECK: }{{$}}
    }
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @normalize_while
  tt.func @normalize_while(%initial: i32, %run: i1) {
    %false = arith.constant false
    // CHECK: %[[WHILE:.*]]:2 = scf.while ([[BEFORE_STATE:%.*]] = %{{.*}}, [[BEFORE_ACTIVE:%.*]] = %{{.*}})
    %while = scf.while (%state = %initial, %active = %run) : (i32, i1) -> i1 {
      // CHECK-NEXT: scf.condition([[BEFORE_ACTIVE]]) [[BEFORE_STATE]], [[BEFORE_ACTIVE]] : i32, i1
      scf.condition(%active) %active : i1
    } do {
    // CHECK: ^bb0([[AFTER_STATE:%.*]]: i32, [[AFTER_ACTIVE:%.*]]: i1):
    ^bb0(%active: i1):
      // CHECK: arith.xori [[AFTER_ACTIVE]], %{{.*}}
      %next = arith.xori %active, %false : i1
      // CHECK: scf.yield %{{.*}}, %{{.*}} : i32, i1
      scf.yield %initial, %next : i32, i1
    } attributes {tt.warp_specialize}
    // CHECK: arith.extui %[[WHILE]]#1 : i1 to i32
    %used = arith.extui %while : i1 to i32
    tt.return
  }
}
