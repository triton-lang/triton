// RUN: triton-opt %s -split-input-file -nv-per-lane-loop-retirement -verify-diagnostics | FileCheck %s

// A lock-step while-loop latch: per-lane predicate, zext, warp redux.max,
// compare-with-zero, conditional branch. The pass must verify safety, branch
// on the per-lane predicate, delete the redux, capture activemask in the
// preheader, and reconverge at the exit.
//   %acc is live-out and frozen (add of select(pred, x, 0));
//   %j feeds the predicate and is a monotone +1 induction.

// CHECK-LABEL: @retire_simple
llvm.func @retire_simple(%trip: i32, %out: !llvm.ptr<1>) {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  // CHECK: [[AM:%.*]] = llvm.inline_asm has_side_effects {{.*}}"activemask.b32 $0;", "=r"
  llvm.br ^header(%c0, %c0 : i32, i32)
^header(%j: i32, %acc: i32):
  // CHECK: [[PRED:%.*]] = llvm.icmp "slt"
  %pred = llvm.icmp "slt" %j, %trip : i32
  %z = llvm.zext %pred : i1 to i32
  %mask = llvm.mlir.constant(-1 : i32) : i32
  // CHECK-NOT: nvvm.redux.sync
  %r = nvvm.redux.sync max %z, %mask : i32 -> i32
  %any = llvm.icmp "sgt" %r, %c0 : i32
  // CHECK: llvm.cond_br [[PRED]], ^bb2, ^bb3
  // expected-remark @below {{applying per-lane loop retirement (verified: 1 live-out value(s) frozen on inactive iterations, predicate monotone)}}
  llvm.cond_br %any, ^body, ^exit
^body:
  %inc = llvm.select %pred, %j, %c0 : i1, i32
  %accn = llvm.add %acc, %inc : i32
  %jn = llvm.add %j, %c1 : i32
  llvm.br ^header(%jn, %accn : i32, i32)
^exit:
  // CHECK: ^bb3:
  // CHECK-NEXT: nvvm.bar.warp.sync [[AM]]
  llvm.store %acc, %out : i32, !llvm.ptr<1>
  llvm.return
}

// -----

// An add-reduction latch (`tl.sum(pred) > 0`) is also "any lane active"
// over the {0,1} range and must be rewritten too.

// CHECK-LABEL: @retire_sum_latch
llvm.func @retire_sum_latch(%trip: i32, %out: !llvm.ptr<1>) {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  llvm.br ^header(%c0 : i32)
^header(%j: i32):
  %pred = llvm.icmp "slt" %j, %trip : i32
  %z = llvm.zext %pred : i1 to i32
  %mask = llvm.mlir.constant(-1 : i32) : i32
  // CHECK-NOT: nvvm.redux.sync
  %r = nvvm.redux.sync add %z, %mask : i32 -> i32
  %any = llvm.icmp "sgt" %r, %c0 : i32
  // CHECK: llvm.cond_br %{{[0-9]+}}, ^bb2, ^bb3
  // expected-remark @below {{applying per-lane loop retirement (verified: 0 live-out value(s) frozen on inactive iterations, predicate monotone)}}
  llvm.cond_br %any, ^body, ^exit
^body:
  %jn = llvm.add %j, %c1 : i32
  llvm.br ^header(%jn : i32)
^exit:
  // CHECK: nvvm.bar.warp.sync
  llvm.store %c1, %out : i32, !llvm.ptr<1>
  llvm.return
}

// -----

// Negative: a warp-collective (shfl) in the body -- a retired lane could no
// longer participate. The loop must be left untouched.

// CHECK-LABEL: @keep_collective_body
llvm.func @keep_collective_body(%trip: i32, %out: !llvm.ptr<1>) {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c31 = llvm.mlir.constant(31 : i32) : i32
  llvm.br ^header(%c0, %c0 : i32, i32)
^header(%j: i32, %acc: i32):
  %pred = llvm.icmp "slt" %j, %trip : i32
  %z = llvm.zext %pred : i1 to i32
  %mask = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: nvvm.redux.sync
  %r = nvvm.redux.sync max %z, %mask : i32 -> i32
  %any = llvm.icmp "sgt" %r, %c0 : i32
  llvm.cond_br %any, ^body, ^exit
^body:
  %shfl = nvvm.shfl.sync bfly %mask, %acc, %c1, %c31 : i32 -> i32
  %accn = llvm.add %acc, %shfl : i32
  %jn = llvm.add %j, %c1 : i32
  llvm.br ^header(%jn, %accn : i32, i32)
^exit:
  // CHECK-NOT: nvvm.bar.warp.sync
  llvm.store %acc, %out : i32, !llvm.ptr<1>
  llvm.return
}

// -----

// Negative (soundness verifier, live-out): %acc is live-out but its update
// is NOT frozen on inactive iterations (unmasked accumulation) -- under
// lock-step an inactive lane would keep accumulating; a retired lane would
// not. Must be refused.

// CHECK-LABEL: @keep_unfrozen_liveout
llvm.func @keep_unfrozen_liveout(%trip: i32, %out: !llvm.ptr<1>) {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  llvm.br ^header(%c0, %c0 : i32, i32)
^header(%j: i32, %acc: i32):
  %pred = llvm.icmp "slt" %j, %trip : i32
  %z = llvm.zext %pred : i1 to i32
  %mask = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: nvvm.redux.sync
  %r = nvvm.redux.sync max %z, %mask : i32 -> i32
  %any = llvm.icmp "sgt" %r, %c0 : i32
  llvm.cond_br %any, ^body, ^exit
^body:
  %accn = llvm.add %acc, %j : i32
  %jn = llvm.add %j, %c1 : i32
  llvm.br ^header(%jn, %accn : i32, i32)
^exit:
  // CHECK-NOT: nvvm.bar.warp.sync
  llvm.store %acc, %out : i32, !llvm.ptr<1>
  llvm.return
}

// -----

// Negative (soundness verifier, monotonicity): the predicate compares two
// carried values that BOTH move (%j grows, %k shrinks toward %j from
// above... here %k also updated unmasked). Once false it could become true
// again under lock-step; a retired lane cannot come back. Must be refused.

// CHECK-LABEL: @keep_nonmonotone_pred
llvm.func @keep_nonmonotone_pred(%trip: i32, %out: !llvm.ptr<1>) {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %cm1 = llvm.mlir.constant(-1 : i32) : i32
  llvm.br ^header(%c0, %trip : i32, i32)
^header(%j: i32, %k: i32):
  %pred = llvm.icmp "slt" %j, %k : i32
  %z = llvm.zext %pred : i1 to i32
  %mask = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: nvvm.redux.sync
  %r = nvvm.redux.sync max %z, %mask : i32 -> i32
  %any = llvm.icmp "sgt" %r, %c0 : i32
  llvm.cond_br %any, ^body, ^exit
^body:
  %jn = llvm.add %j, %c1 : i32
  %kn = llvm.add %k, %c1 : i32
  llvm.br ^header(%jn, %kn : i32, i32)
^exit:
  // CHECK-NOT: nvvm.bar.warp.sync
  llvm.store %j, %out : i32, !llvm.ptr<1>
  llvm.return
}

// -----

// Negative: reduction kind min is not an "any lane active" test. Left
// untouched.

// CHECK-LABEL: @keep_min_redux
llvm.func @keep_min_redux(%trip: i32, %out: !llvm.ptr<1>) {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  llvm.br ^header(%c0 : i32)
^header(%j: i32):
  %pred = llvm.icmp "slt" %j, %trip : i32
  %z = llvm.zext %pred : i1 to i32
  %mask = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: nvvm.redux.sync
  %r = nvvm.redux.sync min %z, %mask : i32 -> i32
  %any = llvm.icmp "sgt" %r, %c0 : i32
  llvm.cond_br %any, ^body, ^exit
^body:
  %jn = llvm.add %j, %c1 : i32
  llvm.br ^header(%jn : i32)
^exit:
  // CHECK-NOT: nvvm.bar.warp.sync
  llvm.store %j, %out : i32, !llvm.ptr<1>
  llvm.return
}

// -----

// Negative: partial member mask -- the loop already executes under known
// divergence this rewrite does not model. Left untouched.

// CHECK-LABEL: @keep_partial_mask
llvm.func @keep_partial_mask(%trip: i32, %out: !llvm.ptr<1>) {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  llvm.br ^header(%c0 : i32)
^header(%j: i32):
  %pred = llvm.icmp "slt" %j, %trip : i32
  %z = llvm.zext %pred : i1 to i32
  %mask = llvm.mlir.constant(65535 : i32) : i32
  // CHECK: nvvm.redux.sync
  %r = nvvm.redux.sync max %z, %mask : i32 -> i32
  %any = llvm.icmp "sgt" %r, %c0 : i32
  llvm.cond_br %any, ^body, ^exit
^body:
  %jn = llvm.add %j, %c1 : i32
  llvm.br ^header(%jn : i32)
^exit:
  // CHECK-NOT: nvvm.bar.warp.sync
  llvm.store %j, %out : i32, !llvm.ptr<1>
  llvm.return
}

// -----

// Negative: a second (side) exit from the loop body -- retired lanes could
// bypass the reconvergence point. Left untouched.

// CHECK-LABEL: @keep_side_exit
llvm.func @keep_side_exit(%trip: i32, %flag: i1, %out: !llvm.ptr<1>) {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  llvm.br ^header(%c0, %c0 : i32, i32)
^header(%j: i32, %acc: i32):
  %pred = llvm.icmp "slt" %j, %trip : i32
  %z = llvm.zext %pred : i1 to i32
  %mask = llvm.mlir.constant(-1 : i32) : i32
  // CHECK: nvvm.redux.sync
  %r = nvvm.redux.sync max %z, %mask : i32 -> i32
  %any = llvm.icmp "sgt" %r, %c0 : i32
  llvm.cond_br %any, ^body, ^exit
^body:
  %jn = llvm.add %j, %c1 : i32
  llvm.cond_br %flag, ^exit, ^back
^back:
  llvm.br ^header(%jn, %acc : i32, i32)
^exit:
  // CHECK-NOT: nvvm.bar.warp.sync
  llvm.store %acc, %out : i32, !llvm.ptr<1>
  llvm.return
}

// -----

// The shape Triton's tile lowering actually produces: loop-carried scalars
// packed in struct-typed block arguments, projected with extractvalue and
// rebuilt with insertvalue on the back edge. The verifier must see through
// the projections: %acc's field is frozen (add of select(pred, x, 0)),
// %j's field is a monotone +1 induction feeding the predicate.

// CHECK-LABEL: @retire_struct_carried
llvm.func @retire_struct_carried(%trip: i32, %out: !llvm.ptr<1>) {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %u = llvm.mlir.undef : !llvm.struct<(i32)>
  %init = llvm.insertvalue %c0, %u[0] : !llvm.struct<(i32)>
  // CHECK: llvm.inline_asm has_side_effects {{.*}}"activemask.b32 $0;", "=r"
  llvm.br ^header(%init, %init : !llvm.struct<(i32)>, !llvm.struct<(i32)>)
^header(%acc: !llvm.struct<(i32)>, %j: !llvm.struct<(i32)>):
  %jv = llvm.extractvalue %j[0] : !llvm.struct<(i32)>
  // CHECK: [[PRED:%.*]] = llvm.icmp "slt"
  %pred = llvm.icmp "slt" %jv, %trip : i32
  %z = llvm.zext %pred : i1 to i32
  %mask = llvm.mlir.constant(-1 : i32) : i32
  // CHECK-NOT: nvvm.redux.sync
  %r = nvvm.redux.sync max %z, %mask : i32 -> i32
  %any = llvm.icmp "sgt" %r, %c0 : i32
  // CHECK: llvm.cond_br [[PRED]], ^bb2, ^bb3
  // expected-remark @below {{applying per-lane loop retirement (verified: 1 live-out value(s) frozen on inactive iterations, predicate monotone)}}
  llvm.cond_br %any, ^body, ^exit
^body:
  %inc = llvm.select %pred, %jv, %c0 : i1, i32
  %av = llvm.extractvalue %acc[0] : !llvm.struct<(i32)>
  %an = llvm.add %av, %inc : i32
  %accn = llvm.insertvalue %an, %u[0] : !llvm.struct<(i32)>
  %jn = llvm.add %jv, %c1 : i32
  %jpack = llvm.insertvalue %jn, %u[0] : !llvm.struct<(i32)>
  llvm.br ^header(%accn, %jpack : !llvm.struct<(i32)>, !llvm.struct<(i32)>)
^exit:
  // CHECK: ^bb3:
  // CHECK-NEXT: nvvm.bar.warp.sync
  %res = llvm.extractvalue %acc[0] : !llvm.struct<(i32)>
  llvm.store %res, %out : i32, !llvm.ptr<1>
  llvm.return
}
