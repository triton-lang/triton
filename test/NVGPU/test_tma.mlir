// RUN: triton-opt %s -split-input-file --convert-nv-gpu-to-llvm | FileCheck %s
#SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32,  "triton_gpu.num-ctas" = 2 : i32} {
  tt.func @test_tma(%im2colOffsets0 : !llvm.struct<(i16, i16)>, %im2colOffsets1 : !llvm.struct<(i16, i16, i16)>) {
    %mbarrier = llvm.mlir.zero : !llvm.ptr<i64, 3>
    %tmaDesc  = llvm.mlir.zero : !llvm.ptr<i8, 1>
    %dst      = llvm.mlir.zero : !llvm.ptr<i8, 3>
    %l2desc   = arith.constant 0 : i64
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %pred = arith.constant 1 : i1
    %mask = arith.constant 15 : i16

    // CHECK: llvm.inline_asm {{.*}} cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
    // CHECK: llvm.inline_asm {{.*}} cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
    nvgpu.tma_load_tiled %dst, %mbarrier, %tmaDesc, %l2desc, %pred, %c0, %c1 {operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 2, 0>}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, i1, i32, i32
    nvgpu.tma_load_tiled %dst, %mbarrier, %tmaDesc, %l2desc, %pred, %c0, %c1, %c2, %c3 {operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 4, 0>}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, i1, i32, i32, i32, i32

    // CHECK: llvm.inline_asm {{.*}} cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint
    // CHECK: llvm.inline_asm {{.*}} cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint
    nvgpu.tma_load_tiled %dst, %mbarrier, %tmaDesc, %l2desc, %pred, %c0, %c1, %mask {operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 2, 1>}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, i1, i32, i32, i16
    nvgpu.tma_load_tiled %dst, %mbarrier, %tmaDesc, %l2desc, %pred, %c0, %c1, %c2, %c3 {operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 4, 0>}: !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, i1, i32, i32, i32, i32

    tt.return
  }
} // end module
