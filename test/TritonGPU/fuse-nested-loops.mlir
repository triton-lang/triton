// RUN: triton-opt %s --allow-unregistered-dialect --tritongpu-fuse-nested-loops | FileCheck %s

// CHECK-LABEL: @empty_function
tt.func @empty_function() {
  tt.return
}

// CHECK-LABEL: @no_fusion
tt.func @no_fusion(%lb: index, %ub: index, %step: index) -> index {
  %c0 = arith.constant 0 : index
  // CHECK: before.loop
  "before.loop"() : () -> ()
  // CHECK-NEXT: scf.for
  %0 = scf.for %i = %lb to %ub step %step iter_args(%k = %c0) -> index {
    // CHECK-NEXT: body
    %1 = "body"(%i, %k) : (index, index) -> index
    // CHECK-NEXT: yield
    scf.yield %1 : index
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: after.loop
  "after.loop"() : () -> ()
  tt.return %0 : index
}

// CHECK-LABEL: @fuse_one_level_simple
// CHECK-SAME: [[LBI:%.*]]: i64, [[UBI:%.*]]: i64, [[STEPI:%.*]]: i64, [[LBJ:%.*]]: i64, [[UBJ:%.*]]: i64, [[STEPJ:%.*]]: i64
tt.func @fuse_one_level_simple(%lbi: i64, %ubi: i64, %stepi: i64, %lbj: i64, %ubj: i64, %stepj: i64) {
  // len_i = len(range(lbi, ubi, stepi))
  //
  // CHECK-NEXT: [[DIFF_I:%.*]] = arith.subi [[UBI]], [[LBI]]
  // CHECK-NEXT: [[LEN_I:%.*]] = arith.ceildivsi [[DIFF_I]], [[STEPI]]

  // len_j = len(range(lbj0, ubj0, stepj0))
  //
  // CHECK-NEXT: [[DIFF_J:%.*]] = arith.subi [[UBJ]], [[LBJ]]
  // CHECK-NEXT: [[LEN_J:%.*]] = arith.ceildivsi [[DIFF_J]], [[STEPJ]]

  // inner_len = len_j0 + len_j1 + ... + len_jN - N
  //
  // CHECK-NEXT: [[PLEN0:%.*]] = arith.constant 0 : i64
  // CHECK-NEXT: [[PLEN1:%.*]] = arith.addi [[PLEN0]], [[LEN_J]]
  // CHECK-NEXT: [[N:%.*]] = arith.constant 0
  // CHECK-NEXT: [[INNER_LEN:%.*]] = arith.subi [[PLEN1]], [[N]]

  // total_iters = len_i * max(1, inner_len)
  //
  // CHECK: [[INNER_LEN_CLAMP:%.*]] = arith.maxsi %c1_i64{{.*}}, [[INNER_LEN]]
  // CHECK: [[TOTAL_ITERS:%.*]] = arith.muli [[LEN_I]], [[INNER_LEN_CLAMP]]

  // T = -1
  // i = lbi
  // j = None
  // for _ in range(total_iters):
  //
  // CHECK: scf.for %{{.*}} = %c0_i64{{.*}} to [[TOTAL_ITERS]] step %c1_i64{{.*}} iter_args(
  // CHECK-SAME: [[T_ARG:%.*]] = %c-1_i64{{.*}}, [[I:%.*]] = [[LBI]], [[J_ARG:%.*]] = %undef_i64) -> (i64, i64, i64) : i64 {
  scf.for %i = %lbi to %ubi step %stepi : i64 {
    // T = (T + 1) % inner_len
    //
    // CHECK:      [[T_PLUS_1:%.*]] = arith.addi [[T_ARG]], %c1_i64
    // CHECK-NEXT: [[T:%.*]] = arith.remsi [[T_PLUS_1]], [[INNER_LEN]]

    // if T == 0:
    //   prologue(i)
    //   j = lbj
    //
    // CHECK:      [[START:%.*]] = arith.subi %c0_i64{{.*}}, %c0_i64{{.*}} : i64
    // CHECK-NEXT: [[PROLOGUE_COND:%.*]] = arith.cmpi eq, [[T]], [[START]]
    // CHECK-NEXT: [[J:%.*]] = scf.if [[PROLOGUE_COND]] -> (i64) {
    // CHECK-NEXT:   "prologue"([[I]]) : (i64) -> ()
    // CHECK-NEXT:   yield [[LBJ]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   yield [[J_ARG]]
    // CHECK-NEXT: }
    "prologue"(%i) : (i64) -> ()

    // if T >= 0 and T < len_j:
    //   body(i, j)
    //   j += stepj
    //
    // CHECK:      [[END:%.*]] = arith.subi [[PLEN1]], %c0_i64
    // CHECK-NEXT: [[GE:%.*]] = arith.cmpi sge, [[T]], [[START]]
    // CHECK-NEXT: [[LT:%.*]] = arith.cmpi slt, [[T]], [[END]]
    // CHECK-NEXT: [[COND:%.*]] = arith.andi [[GE]], [[LT]]
    // CHECK-NEXT: [[J_NEXT:%.*]] = scf.if [[COND]] -> (i64) {
    // CHECK-NEXT:   "body"([[I]], [[J]]) : (i64, i64) -> ()
    // CHECK-NEXT:   [[J_INCR:%.*]] = arith.addi [[J]], [[STEPJ]]
    // CHECK-NEXT:   yield [[J_INCR]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   yield [[J]]
    // CHECK-NEXT: }
    scf.for %j = %lbj to %ubj step %stepj : i64 {
      "body"(%i, %j) : (i64, i64) -> ()
    }

    // if T == len_j - 1:
    //   epilogue(i)
    //   i += stepi
    //
    // CHECK:      [[T_END:%.*]] = arith.subi [[INNER_LEN]], %c1_i64
    // CHECK-NEXT: [[EPILOGUE_COND:%.*]] = arith.cmpi eq, [[T]], [[T_END]]
    // CHECK-NEXT: [[I_NEXT:%.*]] = scf.if [[EPILOGUE_COND]] -> (i64) {
    // CHECK-NEXT:   "epilogue"([[I]]) : (i64) -> ()
    // CHECK-NEXT:   [[I_INCR:%.*]] = arith.addi [[I]], [[STEPI]]
    // CHECK-NEXT:   yield [[I_INCR]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   yield [[I]]
    // CHECK-NEXT: }
    "epilogue"(%i) : (i64) -> ()

    // CHECK-NEXT: yield [[T]], [[I_NEXT]], [[J_NEXT]] : i64, i64, i64
  }
  tt.return
}

// CHECK-LABEL: @fuse_one_level_inouts
// CHECK-SAME: [[LBI:%.*]]: i64, [[UBI:%.*]]: i64, [[STEPI:%.*]]: i64, [[LBJ:%.*]]: i64, [[UBJ:%.*]]: i64, [[STEPJ:%.*]]: i64
// CHECK-SAME: [[INOUT:%.*]]: index
tt.func @fuse_one_level_inouts(%lbi: i64, %ubi: i64, %stepi: i64, %lbj: i64, %ubj: i64, %stepj: i64, %inout: index) -> index {
  // CHECK: [[OUTER_OUTS:%.*]]:7 = scf.for %{{.*}} = %c0_i64{{.*}} to [[TOTAL_ITERS:%.*]] step %c1_i64{{.*}} iter_args(
  // CHECK-SAME: [[T_ARG:%arg[0-9]+]] = %c-1_i64{{[_0-9]*}},
  // CHECK-SAME: [[I:%arg[0-9]+]] = [[LBI]]
  // CHECK-SAME: [[M:%arg[0-9]+]] = [[INOUT]]
  // CHECK-SAME: [[J_ARG:%arg[0-9]+]] = %undef_i64
  // CHECK-SAME: [[K_ARG:%arg[0-9]+]] = %undef
  // CHECK-SAME: [[PROLOGUE_OUT_ARG:%arg[0-9]+]] = %undef
  // CHECK-SAME: [[EPILOGUE_OUT_ARG:%arg[0-9]+]] = %undef
  // CHECK-SAME: ) -> (i64, i64, index, i64, index, index, index) : i64 {
  %outer_out = scf.for %i = %lbi to %ubi step %stepi iter_args(%m = %inout) -> index : i64 {
    // if T == 0:
    //   prologue(i)
    //   j = lbj
    //
    // CHECK:      [[PROLOGUE_OUTS:%.*]]:3 = scf.if %{{[0-9]+}} -> (i64, index, index) {
    // CHECK-NEXT:   [[PROLOGUE_RES:%.*]] = "prologue"([[I]], [[INOUT]], [[M]]) : (i64, index, index) -> index
    // CHECK-NEXT:   yield [[LBJ]], [[PROLOGUE_RES]], [[M]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   yield [[J_ARG]], [[PROLOGUE_OUT_ARG]], [[K_ARG]]
    // CHECK-NEXT: }
    //
    // J := [[PROLOGUE_OUTS]]#0
    // PROLOGUE_OUT := [[PROLOGUE_OUTS]]#1
    // K := [[PROLOGUE_OUTS]]#2
    %prologue_out = "prologue"(%i, %inout, %m) : (i64, index, index) -> index

    // if T >= 0 and T < len_j:
    //   body(i, j)
    //   j += stepj
    //
    // CHECK:      [[BODY_OUTS:%.*]]:2 = scf.if {{.*}} -> (i64, index) {
    // CHECK-NEXT:   [[BODY_OUT:%.*]] = "body"([[I]], [[PROLOGUE_OUTS]]#0, [[PROLOGUE_OUTS]]#2, [[PROLOGUE_OUTS]]#1, [[M]]) : (i64, i64, index, index, index) -> index
    // CHECK-NEXT:   [[J_INCR:%.*]] = arith.addi [[PROLOGUE_OUTS]]#0, [[STEPJ]]
    // CHECK-NEXT:   yield [[J_INCR]], [[BODY_OUT]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   yield [[PROLOGUE_OUTS]]#0, [[K_ARG]]
    // CHECK-NEXT: }
    %inner_out = scf.for %j = %lbj to %ubj step %stepj iter_args(%k = %m) -> index : i64 {
      %body_out = "body"(%i, %j, %k, %prologue_out, %m) : (i64, i64, index, index, index) -> index
      scf.yield %body_out : index
    }

    // if T == len_j - 1:
    //   epilogue(i)
    //   i += stepi
    //
    // CHECK:      [[EPILOGUE_OUTS:%.*]]:2 = scf.if {{.*}} -> (i64, index) {
    // CHECK-NEXT:   [[EPILOGUE_OUT:%.*]] = "epilogue"([[I]], [[PROLOGUE_OUTS]]#1, [[BODY_OUTS]]#1, [[M]]) : (i64, index, index, index) -> index
    // CHECK-NEXT:   [[I_INCR:%.*]] = arith.addi [[I]], [[STEPI]]
    // CHECK-NEXT:   yield [[I_INCR]], [[EPILOGUE_OUT]]
    // CHECK-NEXT: } else {
    // CHECK-NEXT:   yield [[I]], [[EPILOGUE_OUT_ARG]]
    // CHECK-NEXT: }
    %epilogue_out = "epilogue"(%i, %prologue_out, %inner_out, %m) : (i64, index, index, index) -> index

    // CHECK-NEXT: yield [[T]], [[EPILOGUE_OUTS]]#0, [[EPILOGUE_OUTS]]#1, [[BODY_OUTS]]#0, [[BODY_OUTS]]#1, [[PROLOGUE_OUTS]]#1, [[EPILOGUE_OUTS]]#1 : i64, i64, index, i64, index, index, index
    scf.yield %epilogue_out : index
  }
  // CHECK: return [[OUTER_OUTS]]#2
  tt.return %outer_out : index
}
