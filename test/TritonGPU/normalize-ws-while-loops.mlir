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
  // CHECK-LABEL: tt.func @normalize_computed_result(
  // CHECK-SAME: %[[INITIAL:.*]]: i32, %[[ACTIVE:.*]]: i1, %[[LIMIT:.*]]: i32
  tt.func @normalize_computed_result(%initial: i32, %run: i1,
                                     %limit: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    // CHECK: %[[INITIAL_RESULT:.*]] = arith.addi %[[INITIAL]], %{{.*}} : i32
    // CHECK: %[[LOOP:.*]]:2 = scf.while ([[RESULT:%.*]] = %[[INITIAL_RESULT]], [[COND:%.*]] = %[[ACTIVE]])
    %loop = scf.while (%state = %initial, %active = %run) : (i32, i1) -> i32 {
      %result = arith.addi %state, %c1 : i32
      // CHECK-NEXT: scf.condition([[COND]]) [[RESULT]], [[COND]] : i32, i1
      scf.condition(%active) %result : i32
    } do {
    // CHECK: ^bb0([[BODY_RESULT:%.*]]: i32, %{{.*}}: i1):
    ^bb0(%result: i32):
      // CHECK: %[[NEXT_ACTIVE:.*]] = arith.cmpi slt, [[BODY_RESULT]], %[[LIMIT]] : i32
      %next_active = arith.cmpi slt, %result, %limit : i32
      scf.yield %result, %next_active : i32, i1
    } attributes {tt.warp_specialize}
    // CHECK: %[[NEXT_RESULT:.*]] = arith.addi [[BODY_RESULT]], %{{.*}} : i32
    // CHECK: scf.yield %[[NEXT_RESULT]], %[[NEXT_ACTIVE]] : i32, i1
    // CHECK: tt.return %[[LOOP]]#0 : i32
    tt.return %loop : i32
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @normalize_nested_computed_results(
  tt.func @normalize_nested_computed_results(%initial: i32, %run: i1,
                                              %limit: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    // CHECK: %[[OUTER_INITIAL_RESULT:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
    // CHECK-NEXT: %[[OUTER:.*]]:2 = scf.while ([[OUTER_RESULT:%.*]] = %[[OUTER_INITIAL_RESULT]], [[OUTER_COND:%.*]] = %{{.*}})
    // CHECK-NEXT: scf.condition([[OUTER_COND]]) [[OUTER_RESULT]], [[OUTER_COND]] : i32, i1
    %outer = scf.while (%state = %initial, %active = %run) : (i32, i1) -> i32 {
      %result = arith.addi %state, %c1 : i32
      scf.condition(%active) %result : i32
    } do {
    ^bb0(%result: i32):
      // CHECK: %[[INNER_INITIAL_RESULT:.*]] = arith.addi [[OUTER_RESULT]], %{{.*}} : i32
      // CHECK-NEXT: %[[INNER:.*]]:2 = scf.while ([[INNER_RESULT:%.*]] = %[[INNER_INITIAL_RESULT]], [[INNER_COND:%.*]] = %{{.*}})
      // CHECK-NEXT: scf.condition([[INNER_COND]]) [[INNER_RESULT]], [[INNER_COND]] : i32, i1
      %inner = scf.while (%inner_state = %result, %inner_active = %run) : (i32, i1) -> i32 {
        %inner_result = arith.addi %inner_state, %c1 : i32
        scf.condition(%inner_active) %inner_result : i32
      } do {
      ^bb0(%inner_result: i32):
        // CHECK: %[[INNER_NEXT_ACTIVE:.*]] = arith.cmpi slt, [[INNER_RESULT]], %{{.*}} : i32
        // CHECK-NEXT: %[[INNER_NEXT_RESULT:.*]] = arith.addi [[INNER_RESULT]], %{{.*}} : i32
        // CHECK-NEXT: scf.yield %[[INNER_NEXT_RESULT]], %[[INNER_NEXT_ACTIVE]] : i32, i1
        %inner_next_active = arith.cmpi slt, %inner_result, %limit : i32
        scf.yield %inner_result, %inner_next_active : i32, i1
      }
      // CHECK: %[[OUTER_NEXT_ACTIVE:.*]] = arith.cmpi slt, %[[INNER]]#0, %{{.*}} : i32
      // CHECK-NEXT: %[[OUTER_NEXT_RESULT:.*]] = arith.addi %[[INNER]]#0, %{{.*}} : i32
      // CHECK-NEXT: scf.yield %[[OUTER_NEXT_RESULT]], %[[OUTER_NEXT_ACTIVE]] : i32, i1
      %next_active = arith.cmpi slt, %inner, %limit : i32
      scf.yield %inner, %next_active : i32, i1
    } attributes {tt.warp_specialize}
    // CHECK: tt.return %[[OUTER]]#0 : i32
    tt.return %outer : i32
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @normalize_while
  tt.func @normalize_while(%initial: i32, %run: i1) {
    %false = arith.constant false
    // CHECK: %[[WHILE:.*]]:2 = scf.while ([[BEFORE_RESULT:%.*]] = %{{.*}}, [[BEFORE_COND:%.*]] = %{{.*}})
    %while = scf.while (%state = %initial, %active = %run) : (i32, i1) -> i1 {
      // CHECK-NEXT: scf.condition([[BEFORE_COND]]) [[BEFORE_RESULT]], [[BEFORE_COND]] : i1, i1
      scf.condition(%active) %active : i1
    } do {
    // CHECK: ^bb0([[AFTER_RESULT:%.*]]: i1, %{{.*}}: i1):
    ^bb0(%active: i1):
      // CHECK: %[[NEXT:.*]] = arith.xori [[AFTER_RESULT]], %{{.*}}
      %next = arith.xori %active, %false : i1
      // CHECK: scf.yield %[[NEXT]], %[[NEXT]] : i1, i1
      scf.yield %initial, %next : i32, i1
    } attributes {tt.warp_specialize}
    // CHECK: arith.extui %[[WHILE]]#0 : i1 to i32
    %used = arith.extui %while : i1 to i32
    tt.return
  }
}
