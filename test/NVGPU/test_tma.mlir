// RUN: triton-translate %s | FileCheck %s
#SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32,  "triton_gpu.num-ctas" = 2 : i32} {
  tt.func @test_tma(%im2colOffsets0 : !llvm.struct<(i16, i16)>, %im2colOffsets1 : !llvm.struct<(i16, i16, i16)>) {
    %mbarrier = llvm.mlir.null : !llvm.ptr<i64, 3>
    %tmaDesc  = llvm.mlir.null : !llvm.ptr<i8, 1>
    %dst      = llvm.mlir.null : !llvm.ptr<i8, 3>
    %l2desc   = arith.constant 0 : i64
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %pred = arith.constant 1 : i1
    %mask = arith.constant 15 : i16

    // CHECK: void @__nv_tma_load_tiled_2d
    // CHECK: void @__nv_tma_load_tiled_3d
    // CHECK: void @__nv_tma_load_tiled_4d
    // CHECK: void @__nv_tma_load_tiled_5d
    nvgpu.tma_load_tiled %dst, %mbarrier, %tmaDesc, %l2desc, %pred, %c0, %c1 {operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 2, 0>}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, i1, i32, i32
    nvgpu.tma_load_tiled %dst, %mbarrier, %tmaDesc, %l2desc, %pred, %c0, %c1, %c2 {operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 3, 0>}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, i1, i32, i32, i32
    nvgpu.tma_load_tiled %dst, %mbarrier, %tmaDesc, %l2desc, %pred, %c0, %c1, %c2, %c3 {operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 4, 0>}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, i1, i32, i32, i32, i32
    nvgpu.tma_load_tiled %dst, %mbarrier, %tmaDesc, %l2desc, %pred, %c0, %c1, %c2, %c3, %c4 {operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 5, 0>}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, i1, i32, i32, i32, i32, i32

    // CHECK: void @__nv_tma_load_tiled_mcast_2d
    // CHECK: void @__nv_tma_load_tiled_mcast_3d
    // CHECK: void @__nv_tma_load_tiled_mcast_4d
    // CHECK: void @__nv_tma_load_tiled_mcast_5d
    nvgpu.tma_load_tiled %dst, %mbarrier, %tmaDesc, %l2desc, %pred, %c0, %c1, %mask {operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 2, 1>}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, i1, i32, i32, i16
    nvgpu.tma_load_tiled %dst, %mbarrier, %tmaDesc, %l2desc, %pred, %c0, %c1, %c2, %mask {operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 3, 1>}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, i1, i32, i32, i32, i16
    nvgpu.tma_load_tiled %dst, %mbarrier, %tmaDesc, %l2desc, %pred, %c0, %c1, %c2, %c3, %mask {operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 4, 1>}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, i1, i32, i32, i32, i32, i16
    nvgpu.tma_load_tiled %dst, %mbarrier, %tmaDesc, %l2desc, %pred, %c0, %c1, %c2, %c3, %c4, %mask {operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 5, 1>}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, i1, i32, i32, i32, i32, i32, i16

    // CHECK: tail call void @__nv_tma_load_im2col_4d
    // CHECK: tail call void @__nv_tma_load_im2col_5d
    // CHECK: tail call void @__nv_tma_load_im2col_mcast_4d
    // CHECK: tail call void @__nv_tma_load_im2col_mcast_5d
    nvgpu.tma_load_im2col %dst, %mbarrier, %tmaDesc, %l2desc, %im2colOffsets0, %pred, %c0, %c1, %c2, %c3 {mcastMask = 0 : i16}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, !llvm.struct<(i16, i16)>, i1, i32, i32, i32, i32
    nvgpu.tma_load_im2col %dst, %mbarrier, %tmaDesc, %l2desc, %im2colOffsets1, %pred, %c0, %c1, %c2, %c3, %c4 {mcastMask = 0 : i16}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, !llvm.struct<(i16, i16, i16)>, i1, i32, i32, i32, i32, i32

    nvgpu.tma_load_im2col %dst, %mbarrier, %tmaDesc, %l2desc, %im2colOffsets0, %pred, %c0, %c1, %c2, %c3 {mcastMask = 1 : i16}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, !llvm.struct<(i16, i16)>, i1, i32, i32, i32, i32
    nvgpu.tma_load_im2col %dst, %mbarrier, %tmaDesc, %l2desc, %im2colOffsets1, %pred, %c0, %c1, %c2, %c3, %c4 {mcastMask = 1 : i16}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, !llvm.struct<(i16, i16, i16)>, i1, i32, i32, i32, i32, i32

    tt.return
  }
} // end module
