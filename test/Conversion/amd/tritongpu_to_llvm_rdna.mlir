// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx1100 --convert-builtin-func-to-llvm | FileCheck %s

#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: reduce_dpp_max
  tt.func @reduce_dpp_max(%arg0: tensor<32xf32, #blocked3>) {
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 280, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 276, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 274, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 273, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK: rocdl.permlanex16
    // CHECK: llvm.intr.maxnum
    // CHECK: rocdl.readlane
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<32xf32, #blocked3>) -> f32
    tt.return
  }
}

#linear = #ttg.linear<{register = [[16, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @reduce_linear_layout
tt.func private @reduce_linear_layout(%arg0: tensor<32x2xi32, #linear>) -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>> {
  // This tensor has 64 elements with the last dimension across the lower and upper 16 lanes.
  // Therefore, we can reduce it with a 16 element butterfly shuffle.

  // CHECK-DAG: [[result0:%.*]] = llvm.mlir.undef
  // CHECK-DAG: [[select_lo:%.*]] = llvm.mlir.constant(1985229328 : i32)
  // CHECK-DAG: [[select_hi:%.*]] = llvm.mlir.constant(-19088744 : i32)
  // CHECK-DAG: [[reg0:%.*]] = llvm.extractvalue %arg0[0]
  // CHECK-DAG: [[reg1:%.*]] = llvm.extractvalue %arg0[1]
  // CHECK: [[permlane0:%.*]] = rocdl.permlanex16 [[reg0]], [[reg0]], [[select_lo]], [[select_hi]], true, false
  // CHECK: [[sum0:%.*]] = llvm.add [[reg0]], [[permlane0]]
  // CHECK: [[permlane1:%.*]] = rocdl.permlanex16 [[reg1]], [[reg1]], [[select_lo]], [[select_hi]], true, false
  // CHECK: [[sum1:%.*]] = llvm.add [[reg1]], [[permlane1]]
  // CHECK: [[result1:%.*]] = llvm.insertvalue [[sum0]], [[result0]][0]
  // CHECK: [[result2:%.*]] = llvm.insertvalue [[sum1]], [[result1]][1]

  %0 = "tt.reduce"(%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.reduce.return %1 : i32
  }) {axis = 1 : i32} : (tensor<32x2xi32, #linear>) -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>>

  // CHECK: llvm.return [[result2]]
  tt.return %0 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>>
}
}
