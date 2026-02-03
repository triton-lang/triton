// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=100 | FileCheck %s

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
  // CHECK-LABEL: clc_is_canceled
  tt.func @clc_is_canceled(%result: !ttg.memdesc<2xi64, #shared0, #smem>) -> i32 {
    // CHECK: clusterlaunchcontrol.query_cancel.is_canceled.pred.b128
    %is_canceled = ttng.clc_is_canceled %result : !ttg.memdesc<2xi64, #shared0, #smem> -> i32
    tt.return %is_canceled : i32
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_get_first_ctaid_x
  tt.func @clc_get_first_ctaid_x(%result: !ttg.memdesc<2xi64, #shared0, #smem>) -> i32 {
    // CHECK: clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128
    %ctaid = ttng.clc_get_first_ctaid %result {dim = 0 : i32} : !ttg.memdesc<2xi64, #shared0, #smem> -> i32
    tt.return %ctaid : i32
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_get_first_ctaid_y
  tt.func @clc_get_first_ctaid_y(%result: !ttg.memdesc<2xi64, #shared0, #smem>) -> i32 {
    // CHECK: clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128
    %ctaid = ttng.clc_get_first_ctaid %result {dim = 1 : i32} : !ttg.memdesc<2xi64, #shared0, #smem> -> i32
    tt.return %ctaid : i32
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_get_first_ctaid_z
  tt.func @clc_get_first_ctaid_z(%result: !ttg.memdesc<2xi64, #shared0, #smem>) -> i32 {
    // CHECK: clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128
    %ctaid = ttng.clc_get_first_ctaid %result {dim = 2 : i32} : !ttg.memdesc<2xi64, #shared0, #smem> -> i32
    tt.return %ctaid : i32
  }
}
