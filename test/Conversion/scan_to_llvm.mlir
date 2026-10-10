// RUN: split-file %s %t
// RUN: triton-opt %t/scan.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm --canonicalize | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s
// RUN: triton-opt %t/scan.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm --canonicalize | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s --check-prefix=WARP
// RUN: triton-opt %t/parallel-carries.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm --canonicalize | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s --check-prefix=CARRIES
// RUN: triton-opt %t/grouped-carries.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm --canonicalize | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s --check-prefix=GROUPS
// RUN: triton-opt %t/tuple.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm --canonicalize | FileCheck %s --check-prefix=TUPLE
// RUN: triton-opt %t/tuple.mlir --allocate-amdgpu-shared-memory=arch=gfx1250 --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --canonicalize | FileCheck %s --check-prefix=AMD-TUPLE
// RUN: triton-opt %t/warp-transpose.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | FileCheck %s --check-prefix=TRANSPOSE
// RUN: triton-opt %t/warp-transpose.mlir --allocate-amdgpu-shared-memory=arch=gfx1250 --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --canonicalize | FileCheck %s --check-prefix=AMD-TRANSPOSE
// RUN: triton-opt %t/converted-totals.mlir --allocate-amdgpu-shared-memory=arch=gfx1250 --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --canonicalize | FileCheck %s --check-prefix=AMD-RETAIN
// RUN: triton-opt %t/converted-totals.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s --check-prefix=TERMINAL
// RUN: triton-opt %t/warp-transpose.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s --check-prefix=REUSE
// RUN: triton-opt %t/parallel-totals.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | FileCheck %s --check-prefix=COLUMNS
// RUN: triton-opt %t/parallel-totals.mlir --allocate-amdgpu-shared-memory=arch=gfx1250 --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --canonicalize | FileCheck %s --check-prefix=COLUMNS
// RUN: not triton-opt %t/cross-cta.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm 2>&1 | FileCheck %s --check-prefix=ERROR

// RUN: triton-opt %t/ship-chunks.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | FileCheck %s --check-prefix=SHIP

// RUN: triton-opt %t/shuffle-offsets.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | FileCheck %s --check-prefix=OFFSETS

//--- scan.mlir

#layout = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [2], order = [0]}>
#layout_reg4 = #ttg.linear<{register = [[1], [2], [64]], lane = [[4], [8], [16], [32]], warp = [[0]], block = []}>
#layout_adj = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [2], order = [0]}>
#layout_2d = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 2], warpsPerCTA = [2, 1], order = [0,1]}>

#registers = #ttg.linear<{register = [[4], [1], [2]], lane = [[0], [0], [0], [0]], warp = [[0]], block = []}>
#lanes = #ttg.linear<{register = [[8]], lane = [[4], [1], [0], [2]], warp = [[0]], block = []}>
#interleaved = #ttg.linear<{register = [[64], [1], [8], [32]], lane = [[2], [0], [16], [0]], warp = [[4]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 16 : i32} {

// CHECK-LABEL: @test_1d_simple
tt.func private @test_1d_simple(%arg0: tensor<8xi32, #layout>) -> tensor<8xi32, #layout> {
  // CHECK-COUNT-3: tail call i32 @llvm.nvvm.shfl.sync.idx.i32
  // CHECK-NOT: @llvm.nvvm.barrier
  // CHECK: ret
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.scan.return %1 : i32
  }) : (tensor<8xi32, #layout>) -> tensor<8xi32, #layout>
  tt.return %0 : tensor<8xi32, #layout>
}

// CHECK-LABEL: @test_1d_grouped
tt.func private @test_1d_grouped(%arg0: tensor<8xi32, #layout_adj>) -> tensor<8xi32, #layout_adj> {
  // CHECK: tail call i32 @llvm.nvvm.shfl.sync.idx.i32
  // CHECK-NOT: @llvm.nvvm.barrier
  // CHECK: ret
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.scan.return %1 : i32
  }) : (tensor<8xi32, #layout_adj>) -> tensor<8xi32, #layout_adj>
  tt.return %0 : tensor<8xi32, #layout_adj>
}

// CHECK-LABEL: @test_warp_register_groups
// WARP-LABEL: @test_warp_register_groups
// Normalize register ownership with shuffles, then scan the group totals.
// WARP: @llvm.nvvm.shfl.sync.idx.i32
// WARP: add i32
// WARP-NOT: @llvm.nvvm.barrier
// WARP: ret
tt.func private @test_warp_register_groups(%arg: tensor<128xi32, #layout_reg4>) -> tensor<128xi32, #layout_reg4> {
  // CHECK-COUNT-5: @llvm.nvvm.shfl.sync.idx.i32
  // CHECK-NOT: @llvm.nvvm.barrier
  // CHECK: ret
  %0 = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i32, %b: i32):
    %sum = arith.addi %a, %b : i32
    tt.scan.return %sum : i32
  }) : (tensor<128xi32, #layout_reg4>) -> tensor<128xi32, #layout_reg4>
  tt.return %0 : tensor<128xi32, #layout_reg4>
}

tt.func public @anchor_warp_register_groups(%ptr: !llvm.ptr, %arg: !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>) {
  %0 = builtin.unrealized_conversion_cast %arg : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> to tensor<128xi32, #layout_reg4>
  %1 = tt.call @test_warp_register_groups(%0) : (tensor<128xi32, #layout_reg4>) -> tensor<128xi32, #layout_reg4>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<128xi32, #layout_reg4> to !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
  llvm.store volatile %2, %ptr : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>, !llvm.ptr
  tt.return
}

// CHECK-LABEL: @test_2d_grouped
tt.func private @test_2d_grouped(%arg0: tensor<16x1xi32, #layout_2d>) -> tensor<16x1xi32, #layout_2d> {
  // CHECK: tail call i32 @llvm.nvvm.shfl.sync.idx.i32
  // CHECK: store {{.*}}, ptr addrspace(3)
  // CHECK: @llvm.nvvm.barrier
  // CHECK: load i32, ptr addrspace(3)
  // CHECK: ret
  %0 = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.scan.return %1 : i32
  }) : (tensor<16x1xi32, #layout_2d>) -> tensor<16x1xi32, #layout_2d>
  tt.return %0 : tensor<16x1xi32, #layout_2d>
}

// CHECK-LABEL: @test_1d_reversed
tt.func private @test_1d_reversed(%arg0: tensor<8xi32, #layout>) -> tensor<8xi32, #layout> {
  // CHECK-NOT: @llvm.nvvm.shfl.sync.bfly.i32
  // CHECK-COUNT-3: @llvm.nvvm.shfl.sync.idx.i32
  // CHECK-NOT: @llvm.nvvm.shfl.sync.bfly.i32
  // CHECK-NOT: @llvm.nvvm.barrier
  // CHECK: ret
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

// CHECK-LABEL: @test_registers
// WARP-LABEL: @test_registers
// With register bases [4, 1, 2], logical elements 0 and 1 live in input
// registers 0 and 2. Their inclusive prefix must remain in output register 2.
// WARP: %[[R0:.*]] = extractvalue {{.*}}, 0
// WARP: %[[R2:.*]] = extractvalue {{.*}}, 2
// WARP: %[[PREFIX1:.*]] = add i32 %[[R0]], %[[R2]]
// WARP: insertvalue {{.*}}, i32 %[[PREFIX1]], 2
// CHECK-NOT: @llvm.nvvm.shfl
// CHECK-NOT: @llvm.nvvm.barrier
// CHECK: ret
tt.func private @test_registers(%arg: tensor<8xi32, #registers>) -> tensor<8xi32, #registers> {
  %0 = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<8xi32, #registers>) -> tensor<8xi32, #registers>
  tt.return %0 : tensor<8xi32, #registers>
}

tt.func public @anchor_registers(%ptr: !llvm.ptr, %arg: !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>) {
  %0 = builtin.unrealized_conversion_cast %arg : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)> to tensor<8xi32, #registers>
  %1 = tt.call @test_registers(%0) : (tensor<8xi32, #registers>) -> tensor<8xi32, #registers>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<8xi32, #registers> to !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>
  llvm.store volatile %2, %ptr : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32)>, !llvm.ptr
  tt.return
}

// CHECK-LABEL: @test_permuted_lanes
// The low logical axis bits belong to lane bits 1, 3, 0 in that order.
// Normalize register/lane ownership, then shift in reverse logical order.
// Broadcast lane bit 2 must not affect the logical scan.
// CHECK: @llvm.nvvm.shfl.sync.idx.i32
// CHECK-NOT: @llvm.nvvm.barrier
// CHECK: ret
tt.func private @test_permuted_lanes(%arg: tensor<16xi32, #lanes>) -> tensor<16xi32, #lanes> {
  %0 = "tt.scan"(%arg) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<16xi32, #lanes>) -> tensor<16xi32, #lanes>
  tt.return %0 : tensor<16xi32, #lanes>
}

tt.func public @anchor_permuted_lanes(%ptr: !llvm.ptr, %arg: !llvm.struct<(i32, i32)>) {
  %0 = builtin.unrealized_conversion_cast %arg : !llvm.struct<(i32, i32)> to tensor<16xi32, #lanes>
  %1 = tt.call @test_permuted_lanes(%0) : (tensor<16xi32, #lanes>) -> tensor<16xi32, #lanes>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<16xi32, #lanes> to !llvm.struct<(i32, i32)>
  llvm.store volatile %2, %ptr : !llvm.struct<(i32, i32)>, !llvm.ptr
  tt.return
}

// CHECK-LABEL: @test_interleaved
// Exchange the full sequence once, scan it within each warp, then map the
// exclusive carries back to their original owners.
// CHECK-NOT: @llvm.nvvm.barrier
// CHECK: store {{.*}}, ptr addrspace(3)
// CHECK: @llvm.nvvm.barrier
// CHECK: load i32, ptr addrspace(3)
// CHECK-NOT: @llvm.nvvm.barrier
// CHECK: ret
tt.func private @test_interleaved(%arg: tensor<128xi32, #interleaved>) -> tensor<128xi32, #interleaved> {
  %0 = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<128xi32, #interleaved>) -> tensor<128xi32, #interleaved>
  tt.return %0 : tensor<128xi32, #interleaved>
}

tt.func public @anchor_interleaved(%ptr: !llvm.ptr, %arg: !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>) {
  %0 = builtin.unrealized_conversion_cast %arg : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)> to tensor<128xi32, #interleaved>
  %1 = tt.call @test_interleaved(%0) : (tensor<128xi32, #interleaved>) -> tensor<128xi32, #interleaved>
  %2 = builtin.unrealized_conversion_cast %1 : tensor<128xi32, #interleaved> to !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>
  llvm.store volatile %2, %ptr : !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>, !llvm.ptr
  tt.return
}

}

//--- cross-cta.mlir
// ERROR: scan axis distributed across CTAs is not supported
#layout = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = [[128]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func @cross_cta(%x: tensor<256xi32, #layout>) -> tensor<256xi32, #layout> {
    %0 = "tt.scan"(%x) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: i32, %b: i32):
      %sum = arith.addi %a, %b : i32
      tt.scan.return %sum : i32
    }) : (tensor<256xi32, #layout>) -> tensor<256xi32, #layout>
    tt.return %0 : tensor<256xi32, #layout>
  }
}

//--- parallel-carries.mlir

#parallel_carries = #ttg.linear<{register = [[0, 1], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]], lane = [[1, 0], [2, 0], [0, 0], [0, 0], [0, 0]], warp = [[4, 0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

// CARRIES-LABEL: @test_parallel_carries
// Scan the replicated sequence and return carries to the original owners.
// CARRIES: store {{.*}}, ptr addrspace(3)
// CARRIES: @llvm.nvvm.barrier
// CARRIES: load {{.*}}, ptr addrspace(3)
// CARRIES: @llvm.nvvm.shfl.sync.idx.i32
// CARRIES-NOT: @llvm.nvvm.barrier
// CARRIES: ret
tt.func public @test_parallel_carries(%ptr: !tt.ptr<i32>) {
  %rows = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #parallel_carries}>>
  %cols = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #parallel_carries}>>
  %r = tt.expand_dims %rows {axis = 1 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #parallel_carries}>> -> tensor<512x1xi32, #parallel_carries>
  %c = tt.expand_dims %cols {axis = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #parallel_carries}>> -> tensor<1x2xi32, #parallel_carries>
  %two = arith.constant dense<2> : tensor<512x1xi32, #parallel_carries>
  %r2 = arith.muli %r, %two : tensor<512x1xi32, #parallel_carries>
  %rr = tt.broadcast %r2 : tensor<512x1xi32, #parallel_carries> -> tensor<512x2xi32, #parallel_carries>
  %cc = tt.broadcast %c : tensor<1x2xi32, #parallel_carries> -> tensor<512x2xi32, #parallel_carries>
  %offset = arith.addi %rr, %cc : tensor<512x2xi32, #parallel_carries>
  %base = tt.splat %ptr : !tt.ptr<i32> -> tensor<512x2x!tt.ptr<i32>, #parallel_carries>
  %ptrs = tt.addptr %base, %offset : tensor<512x2x!tt.ptr<i32>, #parallel_carries>, tensor<512x2xi32, #parallel_carries>
  %arg = tt.load %ptrs : tensor<512x2x!tt.ptr<i32>, #parallel_carries>
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<512x2xi32, #parallel_carries>) -> tensor<512x2xi32, #parallel_carries>
  tt.store %ptrs, %result : tensor<512x2x!tt.ptr<i32>, #parallel_carries>
  tt.return
}

}

//--- grouped-carries.mlir

#grouped_carries = #ttg.linear<{register = [[0, 1], [16, 0], [32, 0], [64, 0], [128, 0], [256, 0]], lane = [[1, 0], [2, 0], [0, 0], [0, 0], [0, 0]], warp = [[4, 0], [8, 0]], block = []}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

// GROUPS-LABEL: @test_grouped_carries
// Replicate all 128 segment totals within each warp, including register
// sequences, and scan them after the shared-memory conversion.
// GROUPS: store {{.*}}, ptr addrspace(3)
// GROUPS: @llvm.nvvm.barrier
// GROUPS: load {{.*}}, ptr addrspace(3)
// GROUPS: @llvm.nvvm.shfl.sync.idx.i32
// GROUPS: fadd float
// GROUPS-NOT: @llvm.nvvm.barrier
// GROUPS: ret
tt.func public @test_grouped_carries(%ptr: !tt.ptr<f32>) {
  %rows = tt.make_range {start = 0 : i32, end = 512 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #grouped_carries}>>
  %cols = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #grouped_carries}>>
  %r = tt.expand_dims %rows {axis = 1 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #grouped_carries}>> -> tensor<512x1xi32, #grouped_carries>
  %c = tt.expand_dims %cols {axis = 0 : i32} : tensor<2xi32, #ttg.slice<{dim = 0, parent = #grouped_carries}>> -> tensor<1x2xi32, #grouped_carries>
  %two = arith.constant dense<2> : tensor<512x1xi32, #grouped_carries>
  %r2 = arith.muli %r, %two : tensor<512x1xi32, #grouped_carries>
  %rr = tt.broadcast %r2 : tensor<512x1xi32, #grouped_carries> -> tensor<512x2xi32, #grouped_carries>
  %cc = tt.broadcast %c : tensor<1x2xi32, #grouped_carries> -> tensor<512x2xi32, #grouped_carries>
  %offset = arith.addi %rr, %cc : tensor<512x2xi32, #grouped_carries>
  %base = tt.splat %ptr : !tt.ptr<f32> -> tensor<512x2x!tt.ptr<f32>, #grouped_carries>
  %ptrs = tt.addptr %base, %offset : tensor<512x2x!tt.ptr<f32>, #grouped_carries>, tensor<512x2xi32, #grouped_carries>
  %arg = tt.load %ptrs : tensor<512x2x!tt.ptr<f32>, #grouped_carries>
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    tt.scan.return %sum : f32
  }) : (tensor<512x2xf32, #grouped_carries>) -> tensor<512x2xf32, #grouped_carries>
  tt.store %ptrs, %result : tensor<512x2x!tt.ptr<f32>, #grouped_carries>
  tt.return
}

}

//--- tuple.mlir

// Thirty-two warp-segment totals share lanes with independent columns.
// Their scan spans registers and lanes in each warp. Reverse tuple operands
// require separate, aligned shared-memory allocations.
#tuple = #ttg.linear<{register = [[8, 0], [16, 0], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [0, 1], [0, 2], [0, 4]], warp = [[4, 0], [0, 0]], block = []}>
#single_tile = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// The i32 operand uses two scratch tiles and the i64 operand uses four.
// Each operand conversion synchronizes before loads and before scratch reuse.
// TUPLE: ttg.shared = 1024 : i32
// TUPLE-LABEL: llvm.func {{.*}}@test_scan_tuple_reverse
// TUPLE: llvm.getelementptr %arg2[512]
// TUPLE-COUNT-3: nvvm.barrier
// TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{.*}}i32>
// TUPLE-COUNT-7: nvvm.barrier
// TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{.*}}i64>
// TUPLE-NOT: nvvm.barrier
// TUPLE: llvm.return
// AMD-TUPLE: ttg.shared = 1024 : i32
// AMD-TUPLE-LABEL: llvm.func {{.*}}@test_scan_tuple_reverse
// AMD-TUPLE: llvm.getelementptr %arg2[512]
// AMD-TUPLE-COUNT-3: rocdl.s.barrier
// AMD-TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{.*}}i32>
// AMD-TUPLE-COUNT-7: rocdl.s.barrier
// AMD-TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{.*}}i64>
// AMD-TUPLE-NOT: rocdl.s.barrier
// AMD-TUPLE: llvm.return
tt.func private @test_scan_tuple_reverse(%a: tensor<128x8xi32, #tuple>, %b: tensor<128x8xi64, #tuple>) -> (tensor<128x8xi32, #tuple>, tensor<128x8xi64, #tuple>) {
  %a_out, %b_out = "tt.scan"(%a, %b) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%a1: i32, %b1: i64, %a2: i32, %b2: i64):
    %a12 = arith.muli %a1, %a2 : i32
    %a2_wide = arith.extsi %a2 : i32 to i64
    %b12 = arith.muli %b1, %a2_wide : i64
    %b_sum = arith.addi %b12, %b2 : i64
    tt.scan.return %a12, %b_sum : i32, i64
  }) : (tensor<128x8xi32, #tuple>, tensor<128x8xi64, #tuple>) -> (tensor<128x8xi32, #tuple>, tensor<128x8xi64, #tuple>)
  tt.return %a_out, %b_out : tensor<128x8xi32, #tuple>, tensor<128x8xi64, #tuple>
}
// Each operand conversion uses one store-to-load barrier.
// TUPLE-LABEL: llvm.func {{.*}}@test_scan_tuple_single_tile
// TUPLE: llvm.store {{.*}} : vector<1xi32>, !llvm.ptr<3>
// TUPLE: nvvm.barrier
// TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> i32
// TUPLE: llvm.store {{.*}} : vector<1xi64>, !llvm.ptr<3>
// TUPLE: nvvm.barrier
// TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> i64
// TUPLE-NOT: nvvm.barrier
// TUPLE: llvm.return
// AMD-TUPLE-LABEL: llvm.func {{.*}}@test_scan_tuple_single_tile
// AMD-TUPLE: llvm.store {{.*}} : vector<1xi32>, !llvm.ptr<3>
// AMD-TUPLE: rocdl.s.barrier
// AMD-TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{.*}}i32>
// AMD-TUPLE: llvm.store {{.*}} : vector<1xi64>, !llvm.ptr<3>
// AMD-TUPLE: rocdl.s.barrier
// AMD-TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{.*}}i64>
// AMD-TUPLE-NOT: rocdl.s.barrier
// AMD-TUPLE: llvm.return
tt.func private @test_scan_tuple_single_tile(%a: tensor<256xi32, #single_tile>, %b: tensor<256xi64, #single_tile>) -> (tensor<256xi32, #single_tile>, tensor<256xi64, #single_tile>) {
  %a_out, %b_out = "tt.scan"(%a, %b) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a1: i32, %b1: i64, %a2: i32, %b2: i64):
    %sum = arith.addi %a1, %a2 : i32
    %same = arith.cmpi eq, %b1, %b2 : i64
    %result = arith.select %same, %sum, %a2 : i32
    tt.scan.return %result, %b2 : i32, i64
  }) : (tensor<256xi32, #single_tile>, tensor<256xi64, #single_tile>) -> (tensor<256xi32, #single_tile>, tensor<256xi64, #single_tile>)
  tt.return %a_out, %b_out : tensor<256xi32, #single_tile>, tensor<256xi64, #single_tile>
}

}

//--- warp-transpose.mlir

// Two register/lane bit swaps would use shared memory under the default
// conversion heuristic. Scan forces shuffles in both directions, including
// packed i8 and split i64 values, without requesting an allocation offset.
#transpose = #ttg.linear<{register = [[4], [8]], lane = [[1], [2], [0], [0], [0]], warp = [[0], [0]], block = []}>
#parallel = #ttg.linear<{register = [[0, 1]], lane = [[1, 0], [2, 0], [0, 0], [0, 0], [0, 0]], warp = [[0, 0], [0, 0]], block = []}>
#native_prefix = #ttg.linear<{register = [[1], [2], [16]], lane = [[4], [8], [0], [0], [0]], warp = [[0], [0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
// TRANSPOSE: ttg.shared = 0 : i32
// AMD-TRANSPOSE: ttg.shared = 0 : i32
// TRANSPOSE-LABEL: llvm.func {{.*}}@test_scan_warp_transpose_forward
// TRANSPOSE: nvvm.shfl.sync
// TRANSPOSE: llvm.add
// TRANSPOSE: llvm.return
// AMD-TRANSPOSE-LABEL: llvm.func {{.*}}@test_scan_warp_transpose_forward
// AMD-TRANSPOSE: llvm.add
// AMD-TRANSPOSE: llvm.return
tt.func private @test_scan_warp_transpose_forward(%a: tensor<16xi8, #transpose>, %b: tensor<16xi64, #transpose>) -> (tensor<16xi8, #transpose>, tensor<16xi64, #transpose>) {
  %a_out, %b_out = "tt.scan"(%a, %b) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a1: i8, %b1: i64, %a2: i8, %b2: i64):
    %a12 = arith.muli %a1, %a2 : i8
    %a2_wide = arith.extsi %a2 : i8 to i64
    %b12 = arith.muli %b1, %a2_wide : i64
    %b_sum = arith.addi %b12, %b2 : i64
    tt.scan.return %a12, %b_sum : i8, i64
  }) : (tensor<16xi8, #transpose>, tensor<16xi64, #transpose>) -> (tensor<16xi8, #transpose>, tensor<16xi64, #transpose>)
  tt.return %a_out, %b_out : tensor<16xi8, #transpose>, tensor<16xi64, #transpose>
}
// TRANSPOSE-LABEL: llvm.func {{.*}}@test_scan_warp_transpose_reverse
// TRANSPOSE: nvvm.shfl.sync
// TRANSPOSE: llvm.add
// TRANSPOSE: llvm.return
// AMD-TRANSPOSE-LABEL: llvm.func {{.*}}@test_scan_warp_transpose_reverse
// AMD-TRANSPOSE: llvm.add
// AMD-TRANSPOSE: llvm.return
tt.func private @test_scan_warp_transpose_reverse(%a: tensor<16xi8, #transpose>, %b: tensor<16xi64, #transpose>) -> (tensor<16xi8, #transpose>, tensor<16xi64, #transpose>) {
  %a_out, %b_out = "tt.scan"(%a, %b) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%a1: i8, %b1: i64, %a2: i8, %b2: i64):
    %a12 = arith.muli %a1, %a2 : i8
    %a2_wide = arith.extsi %a2 : i8 to i64
    %b12 = arith.muli %b1, %a2_wide : i64
    %b_sum = arith.addi %b12, %b2 : i64
    tt.scan.return %a12, %b_sum : i8, i64
  }) : (tensor<16xi8, #transpose>, tensor<16xi64, #transpose>) -> (tensor<16xi8, #transpose>, tensor<16xi64, #transpose>)
  tt.return %a_out, %b_out : tensor<16xi8, #transpose>, tensor<16xi64, #transpose>
}
// Native four-register prefixes must be computed before any lane shuffles.
// Only their totals undergo the register/lane conversion.
// TRANSPOSE-LABEL: llvm.func {{.*}}@test_scan_native_prefix_forward
// TRANSPOSE-NOT: nvvm.shfl.sync
// TRANSPOSE: llvm.add
// TRANSPOSE: nvvm.shfl.sync
// TRANSPOSE: llvm.return
// AMD-TRANSPOSE-LABEL: llvm.func {{.*}}@test_scan_native_prefix_forward
// AMD-TRANSPOSE: llvm.add
// AMD-TRANSPOSE: llvm.return
tt.func private @test_scan_native_prefix_forward(%a: tensor<32xi8, #native_prefix>, %b: tensor<32xi64, #native_prefix>) -> (tensor<32xi8, #native_prefix>, tensor<32xi64, #native_prefix>) {
  %a_out, %b_out = "tt.scan"(%a, %b) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a1: i8, %b1: i64, %a2: i8, %b2: i64):
    %a12 = arith.muli %a1, %a2 : i8
    %a2_wide = arith.extsi %a2 : i8 to i64
    %b12 = arith.muli %b1, %a2_wide : i64
    %b_sum = arith.addi %b12, %b2 : i64
    tt.scan.return %a12, %b_sum : i8, i64
  }) : (tensor<32xi8, #native_prefix>, tensor<32xi64, #native_prefix>) -> (tensor<32xi8, #native_prefix>, tensor<32xi64, #native_prefix>)
  tt.return %a_out, %b_out : tensor<32xi8, #native_prefix>, tensor<32xi64, #native_prefix>
}

// TRANSPOSE-LABEL: llvm.func {{.*}}@test_scan_native_prefix_reverse
// TRANSPOSE-NOT: nvvm.shfl.sync
// TRANSPOSE: llvm.add
// TRANSPOSE: nvvm.shfl.sync
// TRANSPOSE: llvm.return
// AMD-TRANSPOSE-LABEL: llvm.func {{.*}}@test_scan_native_prefix_reverse
// AMD-TRANSPOSE: llvm.add
// AMD-TRANSPOSE: llvm.return
tt.func private @test_scan_native_prefix_reverse(%a: tensor<32xi8, #native_prefix>, %b: tensor<32xi64, #native_prefix>) -> (tensor<32xi8, #native_prefix>, tensor<32xi64, #native_prefix>) {
  %a_out, %b_out = "tt.scan"(%a, %b) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%a1: i8, %b1: i64, %a2: i8, %b2: i64):
    %a12 = arith.muli %a1, %a2 : i8
    %a2_wide = arith.extsi %a2 : i8 to i64
    %b12 = arith.muli %b1, %a2_wide : i64
    %b_sum = arith.addi %b12, %b2 : i64
    tt.scan.return %a12, %b_sum : i8, i64
  }) : (tensor<32xi8, #native_prefix>, tensor<32xi64, #native_prefix>) -> (tensor<32xi8, #native_prefix>, tensor<32xi64, #native_prefix>)
  tt.return %a_out, %b_out : tensor<32xi8, #native_prefix>, tensor<32xi64, #native_prefix>
}


// After conversion, each of four logical lanes owns four consecutive values.
// Two scan rounds update every register prefix with the preceding lanes' total.
// The register/lane conversions use butterfly shuffles.
// TRANSPOSE-LABEL: llvm.func {{.*}}@test_scan_lane_prefixes_forward
// TRANSPOSE-COUNT-2: nvvm.shfl.sync idx
// TRANSPOSE-NOT: nvvm.shfl.sync idx
// TRANSPOSE: llvm.return
// AMD-TRANSPOSE-LABEL: llvm.func {{.*}}@test_scan_lane_prefixes_forward
// AMD-TRANSPOSE: llvm.add
// AMD-TRANSPOSE: llvm.return
tt.func private @test_scan_lane_prefixes_forward(%arg: tensor<16xi32, #transpose>) -> tensor<16xi32, #transpose> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<16xi32, #transpose>) -> tensor<16xi32, #transpose>
  tt.return %result : tensor<16xi32, #transpose>
}
// TRANSPOSE-LABEL: llvm.func {{.*}}@test_scan_lane_prefixes_reverse
// TRANSPOSE-COUNT-2: nvvm.shfl.sync idx
// TRANSPOSE-NOT: nvvm.shfl.sync idx
// TRANSPOSE: llvm.return
// AMD-TRANSPOSE-LABEL: llvm.func {{.*}}@test_scan_lane_prefixes_reverse
// AMD-TRANSPOSE: llvm.add
// AMD-TRANSPOSE: llvm.return
tt.func private @test_scan_lane_prefixes_reverse(%arg: tensor<16xi32, #transpose>) -> tensor<16xi32, #transpose> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<16xi32, #transpose>) -> tensor<16xi32, #transpose>
  tt.return %result : tensor<16xi32, #transpose>
}

// Independent register groups share each round's lane lookup and predicate.
// REUSE-LABEL: llvm.func {{.*}}@test_scan_reuse_lane_lookups
// REUSE: %[[P1:.*]] = llvm.icmp
// REUSE: %[[P2:.*]] = llvm.icmp
// REUSE: nvvm.shfl.sync idx %{{.*}}, %{{.*}}, %[[SRC1:[a-zA-Z0-9_]+]],
// REUSE: llvm.select %[[P1]],
// REUSE: nvvm.shfl.sync idx %{{.*}}, %{{.*}}, %[[SRC2:[a-zA-Z0-9_]+]],
// REUSE: llvm.select %[[P2]],
// REUSE-NOT: llvm.icmp
// REUSE: nvvm.shfl.sync idx %{{.*}}, %{{.*}}, %[[SRC1]],
// REUSE: llvm.select %[[P1]],
// REUSE: nvvm.shfl.sync idx %{{.*}}, %{{.*}}, %[[SRC2]],
// REUSE: llvm.select %[[P2]],
// REUSE: llvm.return
tt.func private @test_scan_reuse_lane_lookups(%arg: tensor<4x2xi32, #parallel>) -> tensor<4x2xi32, #parallel> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<4x2xi32, #parallel>) -> tensor<4x2xi32, #parallel>
  tt.return %result : tensor<4x2xi32, #parallel>
}

}

//--- converted-totals.mlir

// Each warp owns 16 consecutive elements in four strided registers. Scan in
// register=[1,2], lane=[4,8,0,0,0], then extract its terminal total directly
// from that layout for the shared-memory exchange. Restore native ownership
// after the exchange, when applying carries to the saved prefixes.
#converted = #ttg.linear<{register = [[4], [8]], lane = [[1], [2], [0], [0], [0]], warp = [[16], [0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
// Broadcast terminal totals before converting their layout.
// Apply the inter-warp carry to every scanned register prefix after the exchange.
// TERMINAL-LABEL: llvm.func {{.*}}@test_scan_converted_totals(
// TERMINAL-COUNT-3: nvvm.shfl.sync idx
// TERMINAL-NOT: nvvm.shfl.sync
// TERMINAL: llvm.store {{.*}} : vector<1xi32>, !llvm.ptr<3>
// TERMINAL-NEXT: nvvm.barrier
// TERMINAL: llvm.load
// TERMINAL: nvvm.shfl.sync idx
// TERMINAL: nvvm.shfl.sync bfly
// TERMINAL: llvm.return
// AMD-RETAIN-LABEL: llvm.func {{.*}}@test_scan_converted_totals
// AMD-RETAIN: llvm.store
// AMD-RETAIN: llvm.load
// AMD-RETAIN: llvm.return
tt.func private @test_scan_converted_totals(%arg: tensor<32xi32, #converted>) -> tensor<32xi32, #converted> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<32xi32, #converted>) -> tensor<32xi32, #converted>
  tt.return %result : tensor<32xi32, #converted>
}

// TERMINAL-LABEL: llvm.func {{.*}}@test_scan_converted_totals_reverse(
// TERMINAL-COUNT-3: nvvm.shfl.sync idx
// TERMINAL-NOT: nvvm.shfl.sync
// TERMINAL: llvm.store {{.*}} : vector<1xi32>, !llvm.ptr<3>
// TERMINAL-NEXT: nvvm.barrier
// TERMINAL: llvm.load
// TERMINAL: nvvm.shfl.sync idx
// TERMINAL: nvvm.shfl.sync bfly
// TERMINAL: llvm.return
// AMD-RETAIN-LABEL: llvm.func {{.*}}@test_scan_converted_totals_reverse
// AMD-RETAIN: llvm.store
// AMD-RETAIN: llvm.load
// AMD-RETAIN: llvm.return
tt.func private @test_scan_converted_totals_reverse(%arg: tensor<32xi32, #converted>) -> tensor<32xi32, #converted> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<32xi32, #converted>) -> tensor<32xi32, #converted>
  tt.return %result : tensor<32xi32, #converted>
}
}

//--- parallel-totals.mlir

#parallel_totals = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
// Preserve the eight independent register columns while replicating four
// segment totals in each warp. The totals scan uses two adds per column,
// plus 40 intra-warp adds and eight carry adds, for 64 in total.
// COLUMNS-LABEL: llvm.func {{.*}}@test_scan_parallel_totals(
// COLUMNS-COUNT-64: llvm.fadd
// COLUMNS-NOT: llvm.fadd
// COLUMNS: llvm.return
tt.func private @test_scan_parallel_totals(%arg: tensor<128x8xf32, #parallel_totals>) -> tensor<128x8xf32, #parallel_totals> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    tt.scan.return %sum : f32
  }) : (tensor<128x8xf32, #parallel_totals>) -> tensor<128x8xf32, #parallel_totals>
  tt.return %result : tensor<128x8xf32, #parallel_totals>
}
}

//--- ship-chunks.mlir

#ship = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
// Convert the full register/lane sequence within the warp, then scan it.
// SHIP: ttg.shared = 0 : i32
// SHIP-LABEL: llvm.func {{.*}}@test_scan_register_lane_exchanges(
// SHIP: nvvm.shfl.sync
// SHIP: llvm.fadd
// SHIP-NOT: nvvm.barrier
// SHIP: llvm.return
tt.func private @test_scan_register_lane_exchanges(%arg: tensor<512xf32, #ship>) -> tensor<512xf32, #ship> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    tt.scan.return %sum : f32
  }) : (tensor<512xf32, #ship>) -> tensor<512xf32, #ship>
  tt.return %result : tensor<512xf32, #ship>
}
}

//--- shuffle-offsets.mlir

#groups = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#strided = #ttg.linear<{register = [], lane = [[1, 0], [0, 1], [0, 2], [0, 4], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#gapped = #ttg.linear<{register = [], lane = [[0, 1], [1, 0], [0, 2], [2, 0], [0, 4]], warp = [[0, 0], [0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
// Five scan rounds and the exclusive carry use shuffle-up.
// OFFSETS-LABEL: llvm.func {{.*}}@test_scan_shuffle_up_carry(
// OFFSETS-NOT: nvvm.shfl.sync idx
// OFFSETS-COUNT-6: nvvm.shfl.sync up
// OFFSETS-NOT: nvvm.shfl.sync
// OFFSETS: llvm.return
tt.func private @test_scan_shuffle_up_carry(%arg: tensor<64xi32, #groups>) -> tensor<64xi32, #groups> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<64xi32, #groups>) -> tensor<64xi32, #groups>
  tt.return %result : tensor<64xi32, #groups>
}

// Physical lane bit zero selects an independent row. The scan strides are
// 2, 4, 8, and 16, preserving that row bit.
// OFFSETS-LABEL: llvm.func {{.*}}@test_scan_shuffle_up_strided(
// OFFSETS-DAG: %[[TWO:.*]] = llvm.mlir.constant(2 : i32)
// OFFSETS-DAG: %[[FOUR:.*]] = llvm.mlir.constant(4 : i32)
// OFFSETS-DAG: %[[EIGHT:.*]] = llvm.mlir.constant(8 : i32)
// OFFSETS-DAG: %[[SIXTEEN:.*]] = llvm.mlir.constant(16 : i32)
// OFFSETS: nvvm.shfl.sync up %{{.*}}, %{{.*}}, %[[TWO]],
// OFFSETS: nvvm.shfl.sync up %{{.*}}, %{{.*}}, %[[FOUR]],
// OFFSETS: nvvm.shfl.sync up %{{.*}}, %{{.*}}, %[[EIGHT]],
// OFFSETS: nvvm.shfl.sync up %{{.*}}, %{{.*}}, %[[SIXTEEN]],
// OFFSETS: llvm.return
tt.func private @test_scan_shuffle_up_strided(%arg: tensor<2x16xi32, #strided>) -> tensor<2x16xi32, #strided> {
  %result = "tt.scan"(%arg) <{axis = 1 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<2x16xi32, #strided>) -> tensor<2x16xi32, #strided>
  tt.return %result : tensor<2x16xi32, #strided>
}

// The first two rounds cross gaps between scan lane bits 0, 2, and 4.
// The last round changes only bit 4 and has a uniform distance of 16.
// OFFSETS-LABEL: llvm.func {{.*}}@test_scan_shuffle_up_gapped(
// OFFSETS: %[[SIXTEEN:.*]] = llvm.mlir.constant(16 : i32)
// OFFSETS-COUNT-2: nvvm.shfl.sync idx
// OFFSETS: nvvm.shfl.sync up %{{.*}}, %{{.*}}, %[[SIXTEEN]],
// OFFSETS-NOT: nvvm.shfl.sync
// OFFSETS: llvm.return
tt.func private @test_scan_shuffle_up_gapped(%arg: tensor<4x8xi32, #gapped>) -> tensor<4x8xi32, #gapped> {
  %result = "tt.scan"(%arg) <{axis = 1 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<4x8xi32, #gapped>) -> tensor<4x8xi32, #gapped>
  tt.return %result : tensor<4x8xi32, #gapped>
}
}
