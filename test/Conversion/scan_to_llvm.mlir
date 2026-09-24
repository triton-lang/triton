// RUN: split-file %s %t
// RUN: triton-opt %t/scan.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s
// RUN: triton-opt %t/scan.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s --check-prefix=WARP
// RUN: triton-opt %t/parallel-carries.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm --canonicalize | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s --check-prefix=CARRIES
// RUN: triton-opt %t/grouped-carries.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm --canonicalize | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s --check-prefix=GROUPS
// RUN: not triton-opt %t/cross-cta.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm 2>&1 | FileCheck %s --check-prefix=ERROR

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
  // CHECK: tail call i32 @llvm.nvvm.shfl.sync.idx.i32
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
// Two groups: six local additions, sixteen lane-stage additions, two
// register-stage additions, and four interior prefixes. Scanning every
// register at every stage would require 42 additions.
// WARP-COUNT-28: add i32
// WARP-NOT: add i32
// WARP: ret
tt.func private @test_warp_register_groups(%arg: tensor<128xi32, #layout_reg4>) -> tensor<128xi32, #layout_reg4> {
  // CHECK-COUNT-9: @llvm.nvvm.shfl.sync.idx.i32
  // CHECK-NOT: @llvm.nvvm.shfl
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
  // CHECK: st.shared
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
// Reverse traversal sets the current bit and clears the preceding bits,
// preserving broadcast lane bit 2 throughout.
// CHECK: [[TID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
// CHECK: [[CLEAR0:%.*]] = and i32 [[TID]], 13
// CHECK: [[SRC0:%.*]] = or disjoint i32 [[CLEAR0]], 2
// CHECK: @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %{{.*}}, i32 [[SRC0]], i32 31)
// CHECK: [[CLEAR1:%.*]] = and i32 [[TID]], 5
// CHECK: [[SRC1:%.*]] = or disjoint i32 [[CLEAR1]], 8
// CHECK: @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %{{.*}}, i32 [[SRC1]], i32 31)
// CHECK: [[CLEAR2:%.*]] = and i32 [[TID]], 4
// CHECK: [[SRC2:%.*]] = or disjoint i32 [[CLEAR2]], 1
// CHECK: @llvm.nvvm.shfl.sync.idx.i32(i32 -1, i32 %{{.*}}, i32 [[SRC2]], i32 31)
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
// Publish all warp-local segment totals once. Later register/lane axis bits
// must not introduce another shared-memory exchange.
// CHECK-NOT: @llvm.nvvm.barrier
// CHECK: st.shared
// CHECK: @llvm.nvvm.barrier
// CHECK-NOT: @llvm.nvvm.barrier
// CHECK-NOT: st.shared
// CHECK: load i32, ptr addrspace(3)
// CHECK-NOT: @llvm.nvvm.barrier
// CHECK-NOT: st.shared
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
// Each thread owns two columns. Read the first batch for both columns before
// advancing either carry to the second batch, exposing independent chains.
// CARRIES: @llvm.nvvm.barrier
// CARRIES: load i32, ptr addrspace(3) @global_smem
// CARRIES: load i32, ptr addrspace(3) getelementptr {{.*}}i64 4)
// CARRIES: load i32, ptr addrspace(3) getelementptr {{.*}}i64 504)
// CARRIES: load i32, ptr addrspace(3) getelementptr {{.*}}i64 508)
// CARRIES-NOT: @llvm.nvvm.barrier
// CARRIES-NOT: st.shared
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
// Each group of four warp totals is combined independently. The long carry
// chain accumulates group totals, rather than every individual shared load.
// GROUPS: @llvm.nvvm.barrier
// GROUPS: %[[A:.*]] = load float, ptr addrspace(3) @global_smem
// GROUPS: %[[B:.*]] = load float, ptr addrspace(3) getelementptr {{.*}}i64 8)
// GROUPS: %[[AB:.*]] = fadd float %[[A]], %[[B]]
// GROUPS: %[[C:.*]] = load float, ptr addrspace(3) getelementptr {{.*}}i64 16)
// GROUPS: %[[ABC:.*]] = fadd float %[[AB]], %[[C]]
// GROUPS: %[[D:.*]] = load float, ptr addrspace(3) getelementptr {{.*}}i64 24)
// GROUPS: %[[TOTAL:.*]] = fadd float %[[ABC]], %[[D]]
// GROUPS: %[[E:.*]] = load float, ptr addrspace(3) getelementptr {{.*}}i64 32)
// GROUPS: %[[F:.*]] = load float, ptr addrspace(3) getelementptr {{.*}}i64 40)
// GROUPS: %[[EF:.*]] = fadd float %[[E]], %[[F]]
// GROUPS: %[[G:.*]] = load float, ptr addrspace(3) getelementptr {{.*}}i64 48)
// GROUPS: %[[EFG:.*]] = fadd float %[[EF]], %[[G]]
// GROUPS: %[[H:.*]] = load float, ptr addrspace(3) getelementptr {{.*}}i64 56)
// GROUPS: %[[NEXT:.*]] = fadd float %[[EFG]], %[[H]]
// GROUPS: fadd float %[[TOTAL]], %[[NEXT]]
// GROUPS-NOT: @llvm.nvvm.barrier
// GROUPS-NOT: st.shared
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
