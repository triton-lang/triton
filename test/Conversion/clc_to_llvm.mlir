// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=100 | FileCheck %s --dump-input-context=50

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_try_cancel
  tt.func @clc_try_cancel(%result: !ttg.memdesc<2xi64, #shared0, #smem>, %mbar: !ttg.memdesc<1xi64, #shared0, #smem>) {
    // CHECK: clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128
    ttng.clc_try_cancel %result, %mbar : !ttg.memdesc<2xi64, #shared0, #smem>, !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }
}

// -----

#shared_clc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_try_cancel_multicast
  tt.func @clc_try_cancel_multicast(%result: !ttg.memdesc<2xi64, #shared_clc, #smem>, %mbar: !ttg.memdesc<1xi64, #barrier, #smem>) {
    // CHECK: clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128
    ttng.clc_try_cancel %result, %mbar : !ttg.memdesc<2xi64, #shared_clc, #smem>, !ttg.memdesc<1xi64, #barrier, #smem>
    tt.return
  }
}

// -----

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_load_result
  tt.func private @clc_load_result(%result: !ttg.memdesc<2xi64, #shared0, #smem>) -> i128 {
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<3> -> i128
    %res = ttng.clc_load_result %result : !ttg.memdesc<2xi64, #shared0, #smem> -> i128
    tt.return %res : i128
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_is_canceled
  tt.func private @clc_is_canceled(%clcRes: i128) -> i1 {
    // CHECK: nvvm.clusterlaunchcontrol.query.cancel query = is_canceled, %arg0 : i1
    %is_canceled = ttng.clc_is_canceled %clcRes : i128 -> i1
    tt.return %is_canceled : i1
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_get_program_id_x
  tt.func private @clc_get_program_id_x(%clcResult: i128) -> i32 {
    // CHECK: nvvm.clusterlaunchcontrol.query.cancel query = get_first_cta_id_x, %arg0 : i32
    // CHECK-NOT: sdiv
    %ctaid = ttng.clc_get_program_id %clcResult, x : i128 -> i32
    tt.return %ctaid : i32
  }
}

// -----

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_get_program_id_x_multicta
  // CHECK-DAG: %[[four:.*]] = llvm.mlir.constant(4 : i32)
  tt.func private @clc_get_program_id_x_multicta(%clcResult: i128) -> i32 {
    // CHECK: %[[ctaid:[^ ]*]] = nvvm.clusterlaunchcontrol.query.cancel query = get_first_cta_id_x, %arg0 : i32
    // CHECK-NEXT: llvm.sdiv %[[ctaid]], %[[four]] : i32
    %ctaid = ttng.clc_get_program_id %clcResult, x : i128 -> i32
    tt.return %ctaid : i32
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_get_program_id_y
  tt.func private @clc_get_program_id_y(%clcResult: i128) -> i32 {
    // CHECK: nvvm.clusterlaunchcontrol.query.cancel query = get_first_cta_id_y, %arg0 : i32
    %ctaid = ttng.clc_get_program_id %clcResult, y : i128 -> i32
    tt.return %ctaid : i32
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: clc_get_program_id_z
  tt.func private @clc_get_program_id_z(%clcResult: i128) -> i32 {
    // CHECK: nvvm.clusterlaunchcontrol.query.cancel query = get_first_cta_id_z, %arg0 : i32
    %ctaid = ttng.clc_get_program_id %clcResult, z : i128 -> i32
    tt.return %ctaid : i32
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: llvm.func @grid_dependencies
  tt.func @grid_dependencies() {
    // CHECK: nvvm.griddepcontrol wait
    tt.grid_dependency_wait
    // CHECK-NEXT: nvvm.griddepcontrol launch_dependents
    tt.grid_dependency_launch_dependents
    tt.return
  }
}
