// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s

// Check the optimized LLVMIR, since InstCombine makes the linear layout
// logic understandable enough (in simple cases) to check correctness by eye.

#crazy_2d_src = #ttg.linear<{register = [[0, 2], [2, 0]], lane = [[0, 8], [8, 0], [1, 0], [4, 0], [16, 0]], warp = [[0, 1], [0, 4]], block = []}>
#crazy_2d_idx = #ttg.linear<{register = [[2, 0], [0, 2]], lane = [[0, 8], [16, 0], [1, 0], [8, 0], [4, 0]], warp = [[0, 1], [0, 4]], block = []}>
#broadcasted_lane_1d = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#broadcasted_warp_2d = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @gather_2d_crazy
tt.func private @gather_2d_crazy(%arg0: tensor<32x16xi32, #crazy_2d_idx>, %arg1: tensor<32x16xf32, #crazy_2d_src>) -> tensor<32x16xf32, #crazy_2d_idx> {
  // The specific logic becomes hard to grasp here. Just check the shuffles.

  // CHECK-NEXT: [[SRC0:%.*]] = extractvalue { float, float, float, float } %1, 0
  // CHECK-NEXT: [[SRC1:%.*]] = extractvalue { float, float, float, float } %1, 1
  // CHECK-NEXT: [[SRC2:%.*]] = extractvalue { float, float, float, float } %1, 2
  // CHECK-NEXT: [[SRC3:%.*]] = extractvalue { float, float, float, float } %1, 3

  // CHECK: [[VALUE0:%.*]] = bitcast float [[SRC0]] to i32
  // CHECK-NEXT: tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[VALUE0]],
  // CHECK-NEXT: {{%.*}} = bitcast i32 {{%.*}} to float
  // CHECK-NEXT: {{%.*}} = icmp eq i32
  // CHECK-NEXT: {{%.*}} = select i1
  // CHECK-NEXT: [[VALUE2:%.*]] = bitcast float [[SRC2]] to i32
  // CHECK-NEXT: tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[VALUE2]],

  // CHECK: tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[VALUE0]],
  // CHECK-NEXT: {{%.*}} = bitcast i32 {{%.*}} to float
  // CHECK-NEXT: {{%.*}} = icmp eq i32
  // CHECK-NEXT: {{%.*}} = select i1
  // CHECK-NEXT: tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[VALUE2]],

  // CHECK: [[VALUE1:%.*]] = bitcast float [[SRC1]] to i32
  // CHECK-NEXT: tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[VALUE1]],
  // CHECK-NEXT: [[VALUE3:%.*]] = bitcast float [[SRC3]] to i32
  // CHECK-NEXT: tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[VALUE3]],
  // CHECK-NEXT: {{%.*}} = icmp eq i32
  // CHECK-NEXT: {{%.*}} = select i1
  // CHECK-NEXT: {{%.*}} = bitcast i32 {{%.*}} to float

  // CHECK: tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[VALUE1]],
  // CHECK-NEXT: tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[VALUE3]],
  // CHECK-NEXT: {{%.*}} = icmp eq i32
  // CHECK-NEXT: {{%.*}} = select i1
  // CHECK-NEXT: {{%.*}} = bitcast i32 {{%.*}} to float

  %0 = tt.gather %arg1[%arg0] {axis = 0 : i32} : (tensor<32x16xf32, #crazy_2d_src>, tensor<32x16xi32, #crazy_2d_idx>) -> tensor<32x16xf32, #crazy_2d_idx>
  tt.return %0 : tensor<32x16xf32, #crazy_2d_idx>
}

// There are 16 elements in the tensor. For each warp, each half-warp is mapped
// to the 16 elements, so it doesn't matter if the second half [16, 32) indexes
// into [0, 16), since they contain the same data.
// CHECK-LABEL: @gather_broadcasted_lane_1d
tt.func private @gather_broadcasted_lane_1d(%arg0: tensor<16xi32, #broadcasted_lane_1d>, %arg1: tensor<16xf32, #broadcasted_lane_1d>) -> tensor<16xf32, #broadcasted_lane_1d> {
  // CHECK-NEXT: [[SRC:%.*]] = extractvalue { float } %1, 0
  // CHECK-NEXT: [[IDX:%.*]] = extractvalue { i32 } %0, 0

  // CHECK-NEXT: [[LANEID:%.*]] = and i32 [[IDX]], 15
  // CHECK-NEXT: [[VALUE:%.*]] = bitcast float [[SRC]] to i32
  // CHECK-NEXT: [[RES_i32:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[VALUE]], i32 [[LANEID]], i32 31)
  %0 = tt.gather %arg1[%arg0] {axis = 0 : i32} : (tensor<16xf32, #broadcasted_lane_1d>, tensor<16xi32, #broadcasted_lane_1d>) -> tensor<16xf32, #broadcasted_lane_1d>

  // CHECK-NEXT: [[RES:%.*]] = bitcast i32 [[RES_i32]] to float
  // CHECK-NEXT: ret float [[RES]]
  tt.return %0 : tensor<16xf32, #broadcasted_lane_1d>
}

// Single gather column with 64 elements, all of which have to fit into a single
// warp, so the whole column is broadcasted across the 4 warps. Each process the
// same data so the warp doesn't matter.
// CHECK-LABEL: @gather_broadcasted_warp_2d
tt.func private @gather_broadcasted_warp_2d(%arg0: tensor<64x1xi32, #broadcasted_warp_2d>, %arg1: tensor<64x1xf32, #broadcasted_warp_2d>) -> tensor<64x1xf32, #broadcasted_warp_2d> {
  // CHECK-NEXT: [[SRC0:%.*]] = extractvalue { float, float } %1, 0
  // CHECK-NEXT: [[SRC1:%.*]] = extractvalue { float, float } %1, 1
  // CHECK-NEXT: [[IDX0:%.*]] = extractvalue { i32, i32 } %0, 0
  // CHECK-NEXT: [[IDX1:%.*]] = extractvalue { i32, i32 } %0, 1

  // CHECK-NEXT: [[REGID0:%.*]] = and i32 [[IDX0]], 1
  // CHECK-NEXT: [[TMP:%.*]] = lshr i32 [[IDX0]], 1
  // CHECK-NEXT: [[LANEID0:%.*]] = and i32 [[TMP]], 31

  // CHECK-NEXT: [[VALUE0:%.*]] = bitcast float [[SRC0]] to i32
  // CHECK-NEXT: [[RES0_i32:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[VALUE0]], i32 [[LANEID0]], i32 31)
  // CHECK-NEXT: [[VALUE1:%.*]] = bitcast float [[SRC1]] to i32
  // CHECK-NEXT: [[RES1_i32:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[VALUE1]], i32 [[LANEID0]], i32 31)

  // CHECK-NEXT: [[PICK0:%.*]] = icmp eq i32 [[REGID0]], 0
  // CHECK-NEXT: select i1 [[PICK0]], i32 [[RES0_i32]], i32 [[RES1_i32]]

  // CHECK: [[REGID1:%.*]] = and i32 [[IDX1]], 1
  // CHECK-NEXT: [[TMP:%.*]] = lshr i32 [[IDX1]], 1
  // CHECK-NEXT: [[LANEID1:%.*]] = and i32 [[TMP]], 31

  // CHECK-NEXT: [[RES0_i32:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[VALUE0]], i32 [[LANEID1]], i32 31)
  // CHECK-NEXT: [[RES1_i32:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 [[VALUE1]], i32 [[LANEID1]], i32 31)

  // CHECK-NEXT: [[PICK1:%.*]] = icmp eq i32 [[REGID1]], 0
  // CHECK-NEXT: select i1 [[PICK1]], i32 [[RES0_i32]], i32 [[RES1_i32]]
  %0 = tt.gather %arg1[%arg0] {axis = 0 : i32} : (tensor<64x1xf32, #broadcasted_warp_2d>, tensor<64x1xi32, #broadcasted_warp_2d>) -> tensor<64x1xf32, #broadcasted_warp_2d>
  tt.return %0 : tensor<64x1xf32, #broadcasted_warp_2d>
}

// Keep LLVM from DCE'ing the above functions. Use volatile stores to stop LLVM
// from removing unused function results.
tt.func @anchor_warp4(%ptr: !llvm.ptr,
    %arg9: tensor<32x16xi32, #crazy_2d_idx>,
    %arg10: tensor<32x16xf32, #crazy_2d_src>,
    %arg11: tensor<16xi32, #broadcasted_lane_1d>,
    %arg12: tensor<16xf32, #broadcasted_lane_1d>,
    %arg13: tensor<64x1xi32, #broadcasted_warp_2d>,
    %arg14: tensor<64x1xf32, #broadcasted_warp_2d>) {

  %12 = tt.call @gather_2d_crazy(%arg9, %arg10) : (tensor<32x16xi32, #crazy_2d_idx>, tensor<32x16xf32, #crazy_2d_src>) -> tensor<32x16xf32, #crazy_2d_idx>
  %13 = builtin.unrealized_conversion_cast %12 : tensor<32x16xf32, #crazy_2d_idx> to !llvm.struct<(f32, f32, f32, f32)>
  llvm.store volatile %13, %ptr : !llvm.struct<(f32, f32, f32, f32)>, !llvm.ptr

  %14 = tt.call @gather_broadcasted_lane_1d(%arg11, %arg12) : (tensor<16xi32, #broadcasted_lane_1d>, tensor<16xf32, #broadcasted_lane_1d>) -> tensor<16xf32, #broadcasted_lane_1d>
  %15 = builtin.unrealized_conversion_cast %14 : tensor<16xf32, #broadcasted_lane_1d> to !llvm.struct<(f32)>
  llvm.store volatile %15, %ptr : !llvm.struct<(f32)>, !llvm.ptr

  %16 = tt.call @gather_broadcasted_warp_2d(%arg13, %arg14) : (tensor<64x1xi32, #broadcasted_warp_2d>, tensor<64x1xf32, #broadcasted_warp_2d>) -> tensor<64x1xf32, #broadcasted_warp_2d>
  %17 = builtin.unrealized_conversion_cast %16 : tensor<64x1xf32, #broadcasted_warp_2d> to !llvm.struct<(f32, f32)>
  llvm.store volatile %17, %ptr : !llvm.struct<(f32, f32)>, !llvm.ptr

  tt.return
}

}
