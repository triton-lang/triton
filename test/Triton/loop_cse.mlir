// RUN: triton-opt %s -triton-loop-aware-cse -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @loop_buffer_phase_args
tt.func @loop_buffer_phase_args(%arg0: i32) {
  %c2_i32 = arith.constant 2 : i32
  %c128_i32 = arith.constant 128 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK: [[LOOP_RES:%.*]]:3 = scf.for {{.*}} iter_args
  // CHECK-SAME: [[M2_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[M2_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[M1_PHASE:%arg[0-9]+]] = %c0_i32
  %0:10 = scf.for %arg1 = %c0_i32 to %arg0 step %c128_i32 iter_args(%arg2 = %c0_i32, %arg3 = %c0_i32, %arg4 = %c0_i32, %arg5 = %c0_i32, %arg6 = %c0_i32, %arg7 = %c0_i32, %arg8 = %c0_i32, %arg9 = %c0_i32, %arg10 = %c0_i32, %arg11 = %c0_i32) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)  : i32 {
    %1 = arith.subi %arg0, %c128_i32 : i32
    %2 = arith.cmpi slt, %arg1, %1 : i32
    // CHECK: [[M1_PHASE_INCR:%.*]] = arith.xori [[M1_PHASE]], %c1_i32
    %3 = arith.xori %arg7, %c1_i32 : i32
    // CHECK: "index_phase_use"([[M2_INDEX]], [[M2_PHASE]], [[M1_PHASE_INCR]], [[M1_PHASE]])
    "index_phase_use"(%arg4, %arg5, %3, %arg8) : (i32, i32, i32, i32) -> ()
    %4 = arith.addi %arg4, %c1_i32 : i32
    %5 = arith.xori %arg5, %c1_i32 : i32
    %6 = arith.cmpi eq, %4, %c2_i32 : i32
    // CHECK: [[M2_INDEX_INCR:%.*]] = arith.select %{{.*}}, %c0_i32
    // CHECK-NEXT: [[M2_PHASE_INCR:%.*]] = arith.select %{{.*}}, %{{.*}}, [[M2_PHASE]]
    // CHECK-NOT: arith.select
    %7 = arith.select %6, %c0_i32, %4 : i32
    %8 = arith.select %6, %5, %arg5 : i32
    %9 = arith.xori %arg8, %c1_i32 : i32
    %10 = arith.xori %arg11, %c1_i32 : i32
    %11 = arith.xori %arg6, %c1_i32 : i32
    %12 = arith.addi %arg2, %c1_i32 : i32
    %13 = arith.xori %arg3, %c1_i32 : i32
    %14 = arith.cmpi eq, %12, %c2_i32 : i32
    %15 = arith.select %14, %c0_i32, %12 : i32
    %16 = arith.select %14, %13, %arg3 : i32
    // CHECK: "index_phase_use"([[M2_INDEX_INCR]], [[M2_PHASE_INCR]], [[M1_PHASE_INCR]],
    "index_phase_use"(%15, %16, %11, %2) : (i32, i32, i32, i1) -> ()
    %17 = arith.xori %arg10, %c1_i32 : i32
    // CHECK: "index_phase_use"([[M1_PHASE_INCR]], [[M1_PHASE]])
    "index_phase_use"(%17, %arg11) : (i32, i32) -> ()
    %18 = arith.xori %arg9, %c1_i32 : i32
    // CHECK: "index_phase_use"([[M1_PHASE_INCR]], [[M1_PHASE]])
    "index_phase_use"(%17, %arg11) : (i32, i32) -> ()
    scf.yield %15, %16, %7, %8, %11, %3, %9, %18, %17, %10 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
  }
  tt.return
}

// CHECK-LABEL: @invalid_cache_test
tt.func public @invalid_cache_test(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK: %0:4 = scf.for
  %0:4 = scf.for %arg2 = %c0_i32 to %arg0 step %arg1 iter_args(%arg3 = %c0_i32, %arg4 = %c0_i32, %arg5 = %c0_i32, %arg6 = %c0_i32) -> (i32, i32, i32, i32)  : i32 {

    %1 = arith.addi %arg5, %c1_i32 : i32
    %2 = arith.xori %arg6, %c1_i32 : i32
    %3 = arith.cmpi eq, %1, %c3_i32 : i32
    %4 = arith.select %3, %2, %arg6 : i32
    %5 = arith.select %3, %c1_i32, %1 : i32

    %6 = arith.addi %arg3, %c1_i32 : i32
    %7 = arith.xori %arg4, %c1_i32 : i32
    %8 = arith.cmpi eq, %6, %c3_i32 : i32
    %9 = arith.select %8, %c0_i32, %6 : i32
    %10 = arith.select %8, %7, %arg4 : i32

    scf.yield %9, %10, %5, %4 : i32, i32, i32, i32
  }
  tt.return %0#1, %0#3 : i32, i32
}

// CHECK-LABEL: @multiple_op_results
tt.func @multiple_op_results(%arg0: i32) -> (i32, i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK: %0:2 = scf.for
  %0:2 = scf.for %i = %c0_i32 to %arg0 step %c1_i32 iter_args(%a = %c0_i32, %b = %c0_i32) -> (i32, i32) : i32 {
    // CHECK-NEXT: %1:2 = {{.*}} %arg2, %arg3
    %1:2 = tt.elementwise_inline_asm "asm" {constraints = "=r,=r,r,r", pure = true, packed_element = 1 : i32} %a, %b : i32, i32 -> i32, i32
    // CHECK-NEXT: yield %1#0, %1#1 : i32, i32
    scf.yield %1#0, %1#1 : i32, i32
  }
  tt.return %0#0, %0#1 : i32, i32
}
