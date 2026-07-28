// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s

#layout = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [2], order = [0]}>
#layout_adj = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [2], order = [0]}>
#layout_2d = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 2], warpsPerCTA = [2, 1], order = [0,1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 16 : i32} {

// CHECK-LABEL: @test_1d_simple
tt.func private @test_1d_simple(%arg0: tensor<8xi32, #layout>) -> tensor<8xi32, #layout> {
  // CHECK: [[TID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK: [[LANEID_AXIS:%.*]] = and i32 [[TID]], 7
  // CHECK: icmp eq i32 [[LANEID_AXIS]], 0
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.scan.return %1 : i32
  }) : (tensor<8xi32, #layout>) -> tensor<8xi32, #layout>
  tt.return %0 : tensor<8xi32, #layout>
}

// CHECK-LABEL: @test_1d_grouped
tt.func private @test_1d_grouped(%arg0: tensor<8xi32, #layout_adj>) -> tensor<8xi32, #layout_adj> {
  // CHECK: [[TID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK: [[LANEID_AXIS:%.*]] = and i32 [[TID]], 3
  // CHECK: icmp eq i32 [[LANEID_AXIS]], 0
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.scan.return %1 : i32
  }) : (tensor<8xi32, #layout_adj>) -> tensor<8xi32, #layout_adj>
  tt.return %0 : tensor<8xi32, #layout_adj>
}

// CHECK-LABEL: @test_2d_grouped
tt.func private @test_2d_grouped(%arg0: tensor<16x1xi32, #layout_2d>) -> tensor<16x1xi32, #layout_2d> {
  // CHECK: [[TID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK: [[LANEID_AXIS:%.*]] = and i32 [[TID]], 7
  // CHECK: icmp eq i32 [[LANEID_AXIS]], 0
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.scan.return %1 : i32
  }) : (tensor<16x1xi32, #layout_2d>) -> tensor<16x1xi32, #layout_2d>
  tt.return %0 : tensor<16x1xi32, #layout_2d>
}

// CHECK-LABEL: @test_1d_reversed
tt.func private @test_1d_reversed(%arg0: tensor<8xi32, #layout>) -> tensor<8xi32, #layout> {
  // CHECK: [[FLIP0:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %{{.*}}, i32 8, i32 31)
  // CHECK: [[FLIP1:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[FLIP0]], i32 4, i32 31)
  // CHECK: [[FLIP2:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[FLIP1]], i32 2, i32 31)
  // CHECK: [[FLIP3:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[FLIP2]], i32 1, i32 31)
  // CHECK: tail call i32 @llvm.nvvm.shfl.sync.up.i32
  // CHECK: [[UNFLIP0:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 %{{.*}}, i32 8, i32 31)
  // CHECK: [[UNFLIP1:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[UNFLIP0]], i32 4, i32 31)
  // CHECK: [[UNFLIP2:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[UNFLIP1]], i32 2, i32 31)
  // CHECK: tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[UNFLIP2]], i32 1, i32 31)
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.scan.return %1 : i32
  }) : (tensor<8xi32, #layout>) -> tensor<8xi32, #layout>
  tt.return %0 : tensor<8xi32, #layout>
}

// This just prevents the test functions from being DCE'd.
tt.func public @anchor(%ptr: !llvm.ptr, %arg0: !llvm.struct<(i32)>, %arg1: !llvm.struct<(i32, i32)>, %arg2: !llvm.struct<(i32)>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : !llvm.struct<(i32)> to tensor<8xi32, #layout>
  %1 = tt.call @test_1d_simple(%0) : (tensor<8xi32, #layout>) -> tensor<8xi32, #layout>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<8xi32, #layout> to !llvm.struct<(i32)>
  llvm.store volatile %2, %ptr : !llvm.struct<(i32)>, !llvm.ptr

  %3 = builtin.unrealized_conversion_cast %arg1 : !llvm.struct<(i32, i32)> to tensor<8xi32, #layout_adj>
  %4 = tt.call @test_1d_grouped(%3) : (tensor<8xi32, #layout_adj>) -> tensor<8xi32, #layout_adj>
  %5 = builtin.unrealized_conversion_cast %4 : tensor<8xi32, #layout_adj> to !llvm.struct<(i32, i32)>
  llvm.store volatile %5, %ptr : !llvm.struct<(i32, i32)>, !llvm.ptr

  %6 = builtin.unrealized_conversion_cast %arg2 : !llvm.struct<(i32)> to tensor<16x1xi32, #layout_2d>
  %7 = tt.call @test_2d_grouped(%6) : (tensor<16x1xi32, #layout_2d>) -> tensor<16x1xi32, #layout_2d>
  %8 = builtin.unrealized_conversion_cast %7 : tensor<16x1xi32, #layout_2d> to !llvm.struct<(i32)>
  llvm.store volatile %8, %ptr : !llvm.struct<(i32)>, !llvm.ptr

  %9 = tt.call @test_1d_reversed(%0) : (tensor<8xi32, #layout>) -> tensor<8xi32, #layout>
  %10 = builtin.unrealized_conversion_cast %9 : tensor<8xi32, #layout> to !llvm.struct<(i32)>
  llvm.store volatile %10, %ptr : !llvm.struct<(i32)>, !llvm.ptr

  tt.return
}

}
