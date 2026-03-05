// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=100 | FileCheck %s --dump-input-context=50

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_try_cancel
  tt.func @clc_try_cancel(%result: !ttg.memdesc<2xi64, #shared0, #smem>, %mbar: !ttg.memdesc<1xi64, #shared0, #smem>) {
    // CHECK: clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128
    ttng.clc_try_cancel %result, %mbar {multicast = false} : !ttg.memdesc<2xi64, #shared0, #smem>, !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_load_result
  tt.func @clc_load_result(%result: !ttg.memdesc<2xi64, #shared0, #smem>) {
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i128
    %res = ttng.clc_load_result %result : !ttg.memdesc<2xi64, #shared0, #smem> -> i128
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_is_canceled
  tt.func @clc_is_canceled(%clcRes: i128) {
    // CHECK: clusterlaunchcontrol.query_cancel.is_canceled.pred.b128
    %is_canceled = ttng.clc_is_canceled %clcRes : i128 -> i1
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_get_program_id_x
  tt.func @clc_get_program_id_x(%clcResult: i128) {
    // CHECK: clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128
    // CHECK-NOT: sdiv
    %ctaid = ttng.clc_get_program_id %clcResult, x : i128 -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_get_program_id_x_multicta
  tt.func @clc_get_program_id_x_multicta(%clcResult: i128) {
    // CHECK: %[[ctaid:[^ ]*]] = {{.*}}clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128
    // CHECK-NEXT: %[[four:.*]] = llvm.mlir.constant(4 : i32)
    // CHECK-NEXT: llvm.sdiv %[[ctaid]], %[[four]] : i32
    %ctaid = ttng.clc_get_program_id %clcResult, x : i128 -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_get_program_id_y
  tt.func @clc_get_program_id_y(%clcResult: i128) {
    // CHECK: clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128
    %ctaid = ttng.clc_get_program_id %clcResult, y : i128 -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_get_program_id_z
  tt.func @clc_get_program_id_z(%clcResult: i128) {
    // CHECK: clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128
    %ctaid = ttng.clc_get_program_id %clcResult, z : i128 -> i32
    tt.return
  }
}
