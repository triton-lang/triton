// RUN: split-file %s %t
// RUN: triton-opt %t/scan.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm --canonicalize | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s
// RUN: triton-opt %t/tuple.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm --canonicalize | FileCheck %s --check-prefix=TUPLE
// RUN: triton-opt %t/tuple.mlir --allocate-amdgpu-shared-memory=arch=gfx1250 --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --canonicalize | FileCheck %s --check-prefix=AMD-TUPLE
// RUN: triton-opt %t/warp-transpose.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | FileCheck %s --check-prefix=TRANSPOSE
// RUN: triton-opt %t/warp-transpose.mlir --allocate-amdgpu-shared-memory=arch=gfx1250 --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --canonicalize | FileCheck %s --check-prefix=AMD-TRANSPOSE
// RUN: not triton-opt %t/cross-cta.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm 2>&1 | FileCheck %s --check-prefix=ERROR

// RUN: triton-opt %t/shuffle-offsets.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | FileCheck %s --check-prefix=OFFSETS
// RUN: triton-opt %t/shuffle-offsets.mlir --allocate-amdgpu-shared-memory=arch=gfx1250 --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --canonicalize | FileCheck %s --check-prefix=AMD-OFFSETS

// RUN: triton-opt %t/interwarp-groups.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | FileCheck %s --check-prefix=INTERWARP-GROUPS
// RUN: triton-opt %t/interwarp-groups.mlir --allocate-amdgpu-shared-memory=arch=gfx1250 --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --canonicalize | FileCheck %s --check-prefix=AMD-INTERWARP-GROUPS
// RUN: triton-opt %t/gfx90a.mlir --allocate-amdgpu-shared-memory=arch=gfx90a --convert-triton-amdgpu-to-llvm=gfx-arch=gfx90a --canonicalize | FileCheck %s --check-prefix=GFX90A

// RUN: triton-opt %t/tiny-totals.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | FileCheck %s --check-prefix=TINY-TOTALS
// RUN: triton-opt %t/small-lanes.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | FileCheck %s --check-prefix=SMALL-LANES --implicit-check-not="nvvm.shfl.sync idx" --implicit-check-not="nvvm.barrier"

// RUN: triton-opt %t/lane-local-carries.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | FileCheck %s --check-prefix=LOCAL
// RUN: triton-opt %t/lane-local-carries.mlir --allocate-amdgpu-shared-memory=arch=gfx1250 --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --canonicalize | FileCheck %s --check-prefix=LOCAL

// RUN: triton-opt %t/terminal-prefixes.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | FileCheck %s --check-prefix=PREFIX
// RUN: triton-opt %t/terminal-prefixes.mlir --allocate-amdgpu-shared-memory=arch=gfx1250 --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --canonicalize | FileCheck %s --check-prefix=AMD-PREFIX
// RUN: triton-opt %t/parallel-totals.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --canonicalize | FileCheck %s --check-prefix=COLUMNS
// RUN: triton-opt %t/parallel-totals.mlir --allocate-amdgpu-shared-memory=arch=gfx1250 --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 --canonicalize | FileCheck %s --check-prefix=COLUMNS

//--- scan.mlir

#layout = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [16], warpsPerCTA = [2], order = [0]}>
#layout_adj = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [16], warpsPerCTA = [2], order = [0]}>
#layout_2d = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 2], warpsPerCTA = [2, 1], order = [0,1]}>

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

// CHECK-LABEL: @test_2d_grouped
tt.func private @test_2d_grouped(%arg0: tensor<16x1xi32, #layout_2d>) -> tensor<16x1xi32, #layout_2d> {
  // CHECK: tail call i32 @llvm.nvvm.shfl.sync.idx.i32
  // CHECK: asm sideeffect "@{{.*}} st.shared::cta.
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

//--- tuple.mlir

// Thirty-two warp-segment totals share lanes with independent columns.
// Their scan spans registers and lanes in each warp. Reverse tuple operands
// require separate, aligned shared-memory allocations.
#single_tile = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#tuple = #ttg.linear<{register = [[8, 0], [16, 0], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [0, 1], [0, 2], [0, 4]], warp = [[4, 0], [0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// The i32 operand uses one scratch tile and the i64 operand uses two.
// Each operand conversion synchronizes before loads and before scratch reuse.
// TUPLE: ttg.shared = 2048 : i32
// TUPLE-LABEL: llvm.func {{.*}}@test_scan_tuple_reverse
// TUPLE: llvm.getelementptr %arg2[1024]
// TUPLE: nvvm.barrier
// TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{.*}}i32>
// TUPLE: nvvm.barrier
// TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{.*}}i64>
// TUPLE-COUNT-2: nvvm.barrier
// TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{.*}}i64>
// TUPLE-NOT: nvvm.barrier
// TUPLE: llvm.return
// AMD-TUPLE: ttg.shared = 2048 : i32
// AMD-TUPLE-LABEL: llvm.func {{.*}}@test_scan_tuple_reverse
// AMD-TUPLE: llvm.getelementptr %arg2[1024]
// AMD-TUPLE: rocdl.s.barrier
// AMD-TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{.*}}i32>
// AMD-TUPLE: rocdl.s.barrier
// AMD-TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{.*}}i64>
// AMD-TUPLE-COUNT-2: rocdl.s.barrier
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
// Both operands use the same terminal-lane predicate and are stored before
// a single store-to-load barrier.
// TUPLE-LABEL: llvm.func {{.*}}@test_scan_tuple_single_tile
// TUPLE: %[[STORE_PRED:.*]] = llvm.icmp "eq" {{.*}} : i32
// TUPLE: llvm.inline_asm {{.*}}st.shared::cta.b32{{.*}} %{{.*}}, %{{.*}}, %[[STORE_PRED]] :
// TUPLE-NOT: nvvm.barrier
// TUPLE: llvm.inline_asm {{.*}}st.shared::cta.b64{{.*}} %{{.*}}, %{{.*}}, %[[STORE_PRED]] :
// TUPLE: nvvm.barrier
// TUPLE-NOT: nvvm.barrier
// TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> i32
// TUPLE-NOT: nvvm.barrier
// TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> i64
// TUPLE-NOT: nvvm.barrier
// TUPLE: llvm.return
// AMD-TUPLE-LABEL: llvm.func {{.*}}@test_scan_tuple_single_tile
// AMD-TUPLE: llvm.store {{.*}} : vector<1xi32>, !llvm.ptr<3>
// AMD-TUPLE-NOT: rocdl.s.barrier
// AMD-TUPLE: llvm.store {{.*}} : vector<1xi64>, !llvm.ptr<3>
// AMD-TUPLE: rocdl.s.barrier
// AMD-TUPLE-NOT: rocdl.s.barrier
// AMD-TUPLE: llvm.load {{.*}} : !llvm.ptr<3> -> vector<{{.*}}i32>
// AMD-TUPLE-NOT: rocdl.s.barrier
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

}

//--- terminal-prefixes.mlir

#transpose = #ttg.linear<{register = [[4], [8]], lane = [[1], [2], [0], [0], [0]], warp = [[0], [0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
// After conversion, each of four logical lanes owns four consecutive values.
// Two rounds scan the terminal totals; one shuffle supplies the carry for
// the other register prefixes.
// The register/lane conversions use butterfly shuffles.
// PREFIX-LABEL: llvm.func {{.*}}@test_scan_lane_prefixes_forward
// PREFIX-COUNT-3: nvvm.shfl.sync idx
// PREFIX-NOT: nvvm.shfl.sync idx
// PREFIX: llvm.return
// AMD-PREFIX-LABEL: llvm.func {{.*}}@test_scan_lane_prefixes_forward
// AMD-PREFIX: llvm.add
// AMD-PREFIX: llvm.return
tt.func private @test_scan_lane_prefixes_forward(%arg: tensor<16xi32, #transpose>) -> tensor<16xi32, #transpose> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<16xi32, #transpose>) -> tensor<16xi32, #transpose>
  tt.return %result : tensor<16xi32, #transpose>
}
// PREFIX-LABEL: llvm.func {{.*}}@test_scan_lane_prefixes_reverse
// PREFIX-COUNT-3: nvvm.shfl.sync idx
// PREFIX-NOT: nvvm.shfl.sync idx
// PREFIX: llvm.return
// AMD-PREFIX-LABEL: llvm.func {{.*}}@test_scan_lane_prefixes_reverse
// AMD-PREFIX: llvm.add
// AMD-PREFIX: llvm.return
tt.func private @test_scan_lane_prefixes_reverse(%arg: tensor<16xi32, #transpose>) -> tensor<16xi32, #transpose> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<16xi32, #transpose>) -> tensor<16xi32, #transpose>
  tt.return %result : tensor<16xi32, #transpose>
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
// Five scan rounds and the exclusive carry use shuffle-down.
// OFFSETS-LABEL: llvm.func {{.*}}@test_scan_shuffle_down_carry(
// OFFSETS-NOT: nvvm.shfl.sync idx
// OFFSETS-COUNT-6: nvvm.shfl.sync down
// OFFSETS-NOT: nvvm.shfl.sync
// OFFSETS: llvm.return
// AMD-OFFSETS-LABEL: llvm.func {{.*}}@test_scan_shuffle_down_carry(
// AMD-OFFSETS-COUNT-6: rocdl.ds_bpermute
// AMD-OFFSETS-NOT: rocdl.ds_bpermute
// AMD-OFFSETS: llvm.return
tt.func private @test_scan_shuffle_down_carry(%arg: tensor<64xi32, #groups>) -> tensor<64xi32, #groups> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<64xi32, #groups>) -> tensor<64xi32, #groups>
  tt.return %result : tensor<64xi32, #groups>
}

// Physical lane bit zero selects an independent row. The scan strides are
// 2, 4, 8, and 16, preserving that row bit.
// OFFSETS-LABEL: llvm.func {{.*}}@test_scan_shuffle_down_strided(
// OFFSETS-DAG: %[[TWO:.*]] = llvm.mlir.constant(2 : i32)
// OFFSETS-DAG: %[[FOUR:.*]] = llvm.mlir.constant(4 : i32)
// OFFSETS-DAG: %[[EIGHT:.*]] = llvm.mlir.constant(8 : i32)
// OFFSETS-DAG: %[[SIXTEEN:.*]] = llvm.mlir.constant(16 : i32)
// OFFSETS: nvvm.shfl.sync down %{{.*}}, %{{.*}}, %[[TWO]],
// OFFSETS: nvvm.shfl.sync down %{{.*}}, %{{.*}}, %[[FOUR]],
// OFFSETS: nvvm.shfl.sync down %{{.*}}, %{{.*}}, %[[EIGHT]],
// OFFSETS: nvvm.shfl.sync down %{{.*}}, %{{.*}}, %[[SIXTEEN]],
// OFFSETS: llvm.return
// AMD-OFFSETS-LABEL: llvm.func {{.*}}@test_scan_shuffle_down_strided(
// AMD-OFFSETS-COUNT-4: rocdl.ds_bpermute
// AMD-OFFSETS-NOT: rocdl.ds_bpermute
// AMD-OFFSETS: llvm.return
tt.func private @test_scan_shuffle_down_strided(%arg: tensor<2x16xi32, #strided>) -> tensor<2x16xi32, #strided> {
  %result = "tt.scan"(%arg) <{axis = 1 : i32, reverse = true}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<2x16xi32, #strided>) -> tensor<2x16xi32, #strided>
  tt.return %result : tensor<2x16xi32, #strided>
}

// The first two rounds cross gaps between scan lane bits 0, 2, and 4.
// The last round changes only bit 4 and has a uniform distance of 16.
// OFFSETS-LABEL: llvm.func {{.*}}@test_scan_shuffle_down_gapped(
// OFFSETS: %[[SIXTEEN:.*]] = llvm.mlir.constant(16 : i32)
// OFFSETS-COUNT-2: nvvm.shfl.sync idx
// OFFSETS: nvvm.shfl.sync down %{{.*}}, %{{.*}}, %[[SIXTEEN]],
// OFFSETS-NOT: nvvm.shfl.sync
// OFFSETS: llvm.return
// AMD-OFFSETS-LABEL: llvm.func {{.*}}@test_scan_shuffle_down_gapped(
// AMD-OFFSETS-COUNT-3: rocdl.ds_bpermute
// AMD-OFFSETS-NOT: rocdl.ds_bpermute
// AMD-OFFSETS: llvm.return
tt.func private @test_scan_shuffle_down_gapped(%arg: tensor<4x8xi32, #gapped>) -> tensor<4x8xi32, #gapped> {
  %result = "tt.scan"(%arg) <{axis = 1 : i32, reverse = true}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<4x8xi32, #gapped>) -> tensor<4x8xi32, #gapped>
  tt.return %result : tensor<4x8xi32, #gapped>
}
}

//--- interwarp-groups.mlir

#columns = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [0, 1]}>
// Both operands share one store-to-load barrier and retain their complete table.
// INTERWARP-GROUPS: ttg.shared = 49152 : i32
// INTERWARP-GROUPS-LABEL: llvm.func {{.*}}@test_scan_interwarp_groups(
// INTERWARP-GROUPS-NOT: shfl
// INTERWARP-GROUPS: nvvm.barrier
// INTERWARP-GROUPS-NOT: {{shfl|nvvm.barrier}}
// INTERWARP-GROUPS: llvm.return
// AMD-INTERWARP-GROUPS: ttg.shared = 49152 : i32
// AMD-INTERWARP-GROUPS-LABEL: llvm.func {{.*}}@test_scan_interwarp_groups(
// AMD-INTERWARP-GROUPS-NOT: {{ds.bpermute|ds.swizzle|permlane}}
// AMD-INTERWARP-GROUPS: rocdl.s.barrier
// AMD-INTERWARP-GROUPS-NOT: {{ds.bpermute|ds.swizzle|permlane|rocdl.s.barrier}}
// AMD-INTERWARP-GROUPS: llvm.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
  tt.func private @test_scan_interwarp_groups(%values: tensor<128x32xf32, #columns>, %keys: tensor<128x32xi64, #columns>) -> (tensor<128x32xf32, #columns>, tensor<128x32xi64, #columns>) {
    %result:2 = "tt.scan"(%values, %keys) <{axis = 0 : i32, reverse = false}> ({
    ^bb0(%a: f32, %ka: i64, %b: f32, %kb: i64):
      %same = arith.cmpi eq, %ka, %kb : i64
      %sum = arith.addf %a, %b : f32
      %value = arith.select %same, %sum, %b : f32
      tt.scan.return %value, %kb : f32, i64
    }) : (tensor<128x32xf32, #columns>, tensor<128x32xi64, #columns>) -> (tensor<128x32xf32, #columns>, tensor<128x32xi64, #columns>)
    tt.return %result#0, %result#1 : tensor<128x32xf32, #columns>, tensor<128x32xi64, #columns>
  }
}

//--- gfx90a.mlir

// Register, lane, and warp bits alternate along the scan axis. Zero lane
// bases replicate values within a wave64; tuple operands have different widths.
#interleaved = #ttg.linear<{register = [[16, 0], [4, 0], [0, 1]], lane = [[1, 0], [0, 2], [8, 0], [0, 4], [0, 0], [0, 0]], warp = [[2, 0], [32, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx90a"} {
// GFX90A-LABEL: llvm.func {{.*}}@test_scan_wave64_interleaved_reverse(
// GFX90A: llvm.store
// GFX90A: rocdl.s.barrier
// GFX90A: llvm.load
// GFX90A: llvm.return
tt.func private @test_scan_wave64_interleaved_reverse(%a: tensor<64x8xi8, #interleaved>, %b: tensor<64x8xi64, #interleaved>) -> (tensor<64x8xi8, #interleaved>, tensor<64x8xi64, #interleaved>) {
  %a_out, %b_out = "tt.scan"(%a, %b) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%a1: i8, %b1: i64, %a2: i8, %b2: i64):
    %a12 = arith.muli %a1, %a2 : i8
    %a2_wide = arith.extsi %a2 : i8 to i64
    %b12 = arith.muli %b1, %a2_wide : i64
    %b_sum = arith.addi %b12, %b2 : i64
    tt.scan.return %a12, %b_sum : i8, i64
  }) : (tensor<64x8xi8, #interleaved>, tensor<64x8xi64, #interleaved>) -> (tensor<64x8xi8, #interleaved>, tensor<64x8xi64, #interleaved>)
  tt.return %a_out, %b_out : tensor<64x8xi8, #interleaved>, tensor<64x8xi64, #interleaved>
}
}

//--- tiny-totals.mlir

#tiny_totals = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
// Four warp totals are loaded together into registers, avoiding a lane scan.
// TINY-TOTALS-LABEL: llvm.func {{.*}}@test_scan_tiny_totals(
// TINY-TOTALS: nvvm.barrier
// TINY-TOTALS: llvm.load {{.*}} : !llvm.ptr<3> -> vector<4xi32>
// TINY-TOTALS: llvm.return
tt.func private @test_scan_tiny_totals(%arg: tensor<512xi32, #tiny_totals>) -> tensor<512xi32, #tiny_totals> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<512xi32, #tiny_totals>) -> tensor<512xi32, #tiny_totals>
  tt.return %result : tensor<512xi32, #tiny_totals>
}
}

//--- small-lanes.mlir

#layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
// Keep the four independent lane scans in place, without layout conversions.
// SMALL-LANES-LABEL: llvm.func {{.*}}@test_small_independent_lane_scans(
// SMALL-LANES-COUNT-12: nvvm.shfl.sync up
// SMALL-LANES-NOT: nvvm.shfl
// SMALL-LANES: llvm.return
tt.func private @test_small_independent_lane_scans(%arg: tensor<64x8xi32, #layout>) -> tensor<64x8xi32, #layout> {
  %result = "tt.scan"(%arg) <{axis = 1 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<64x8xi32, #layout>) -> tensor<64x8xi32, #layout>
  tt.return %result : tensor<64x8xi32, #layout>
}
}

//--- lane-local-carries.mlir

#lane_local = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
// LOCAL-LABEL: llvm.func {{.*}}@test_scan_lane_local_carries(
// LOCAL-NOT: {{shfl|ds.bpermute|ds.swizzle|permlane}}
// LOCAL: llvm.return
tt.func private @test_scan_lane_local_carries(%arg: tensor<8x32xi32, #lane_local>) -> tensor<8x32xi32, #lane_local> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<8x32xi32, #lane_local>) -> tensor<8x32xi32, #lane_local>
  tt.return %result : tensor<8x32xi32, #lane_local>
}
// LOCAL-LABEL: llvm.func {{.*}}@test_scan_lane_local_carries_reverse(
// LOCAL-NOT: {{shfl|ds.bpermute|ds.swizzle|permlane}}
// LOCAL: llvm.return
tt.func private @test_scan_lane_local_carries_reverse(%arg: tensor<8x32xi32, #lane_local>) -> tensor<8x32xi32, #lane_local> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<8x32xi32, #lane_local>) -> tensor<8x32xi32, #lane_local>
  tt.return %result : tensor<8x32xi32, #lane_local>
}

// LOCAL-LABEL: llvm.func {{.*}}@test_scan_lane_local_carries_broadcast_lanes(
// LOCAL-NOT: {{shfl|ds.bpermute|ds.swizzle|permlane}}
// LOCAL: llvm.return
tt.func private @test_scan_lane_local_carries_broadcast_lanes(%arg: tensor<8x16xi32, #lane_local>) -> tensor<8x16xi32, #lane_local> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<8x16xi32, #lane_local>) -> tensor<8x16xi32, #lane_local>
  tt.return %result : tensor<8x16xi32, #lane_local>
}
// LOCAL-LABEL: llvm.func {{.*}}@test_scan_lane_local_carries_broadcast_lanes_reverse(
// LOCAL-NOT: {{shfl|ds.bpermute|ds.swizzle|permlane}}
// LOCAL: llvm.return
tt.func private @test_scan_lane_local_carries_broadcast_lanes_reverse(%arg: tensor<8x16xi32, #lane_local>) -> tensor<8x16xi32, #lane_local> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<8x16xi32, #lane_local>) -> tensor<8x16xi32, #lane_local>
  tt.return %result : tensor<8x16xi32, #lane_local>
}

}

//--- parallel-totals.mlir

#parallel_totals = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
// Replicate four segment totals in each warp and scan them in registers.
// Register scans and carry propagation require 26 adds in total.
// COLUMNS-LABEL: llvm.func {{.*}}@test_scan_parallel_totals(
// COLUMNS-COUNT-26: llvm.fadd
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
