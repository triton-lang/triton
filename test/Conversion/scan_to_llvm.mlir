// RUN: split-file %s %t
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

//--- shuffle-offsets.mlir

#groups = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#strided = #ttg.linear<{register = [], lane = [[1, 0], [0, 1], [0, 2], [0, 4], [0, 8]], warp = [[0, 0], [0, 0]], block = []}>
#gapped = #ttg.linear<{register = [], lane = [[0, 1], [1, 0], [0, 2], [2, 0], [0, 4]], warp = [[0, 0], [0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:100"} {
// Five scan rounds and the exclusive carry use shuffle-up.
// OFFSETS-LABEL: llvm.func {{.*}}@test_scan_shuffle_up_carry(
// OFFSETS-NOT: nvvm.shfl.sync idx
// OFFSETS-COUNT-5: nvvm.shfl.sync up {{.*}} return_value_and_is_valid
// OFFSETS: nvvm.shfl.sync up {{.*}} : i32 -> i32
// OFFSETS-NOT: nvvm.shfl.sync
// OFFSETS: llvm.return
// AMD-OFFSETS-LABEL: llvm.func {{.*}}@test_scan_shuffle_up_carry(
// AMD-OFFSETS: [[VALID:%.*]] = llvm.icmp "sge" [[LANE:%[^,]+]],
// AMD-OFFSETS: [[SOURCE:%.*]] = llvm.select [[VALID]], {{%.*}}, [[LANE]] : i1, i32
// AMD-OFFSETS: [[ADDR:%.*]] = llvm.shl [[SOURCE]],
// AMD-OFFSETS: [[DATA:%.*]] = rocdl.ds_bpermute [[ADDR]],
// AMD-OFFSETS: [[SUM:%.*]] = llvm.add {{.*}}[[DATA]]
// AMD-OFFSETS: llvm.select [[VALID]], [[SUM]],
// AMD-OFFSETS-COUNT-5: rocdl.ds_bpermute
// AMD-OFFSETS-NOT: rocdl.ds_bpermute
// AMD-OFFSETS: llvm.return
tt.func private @test_scan_shuffle_up_carry(%arg: tensor<64xi32, #groups>) -> tensor<64xi32, #groups> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i32, %rhs: i32):
    %sum = arith.addi %lhs, %rhs : i32
    tt.scan.return %sum : i32
  }) : (tensor<64xi32, #groups>) -> tensor<64xi32, #groups>
  tt.return %result : tensor<64xi32, #groups>
}

// Both words of an i64 read the same lane. Each round uses one bounds result.
// OFFSETS-LABEL: llvm.func {{.*}}@test_scan_shuffle_up_i64(
// OFFSETS-COUNT-5: nvvm.shfl.sync up {{.*}} return_value_and_is_valid
// OFFSETS-NOT: return_value_and_is_valid
// OFFSETS: llvm.return
tt.func private @test_scan_shuffle_up_i64(%arg: tensor<64xi64, #groups>) -> tensor<64xi64, #groups> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i64, %rhs: i64):
    %sum = arith.addi %lhs, %rhs : i64
    tt.scan.return %sum : i64
  }) : (tensor<64xi64, #groups>) -> tensor<64xi64, #groups>
  tt.return %result : tensor<64xi64, #groups>
}

// Physical lane bit zero selects an independent row. Column k is in lanes
// 2*k and 2*k+1, so a shuffle by 2*d supplies the bounds for a scan by d.
// OFFSETS-LABEL: llvm.func {{.*}}@test_scan_shuffle_up_strided(
// OFFSETS-DAG: %[[TWO:.*]] = llvm.mlir.constant(2 : i32)
// OFFSETS-DAG: %[[FOUR:.*]] = llvm.mlir.constant(4 : i32)
// OFFSETS-DAG: %[[EIGHT:.*]] = llvm.mlir.constant(8 : i32)
// OFFSETS-DAG: %[[SIXTEEN:.*]] = llvm.mlir.constant(16 : i32)
// OFFSETS: nvvm.shfl.sync up %{{.*}}, %{{.*}}, %[[TWO]], {{.*}} return_value_and_is_valid
// OFFSETS: nvvm.shfl.sync up %{{.*}}, %{{.*}}, %[[FOUR]], {{.*}} return_value_and_is_valid
// OFFSETS: nvvm.shfl.sync up %{{.*}}, %{{.*}}, %[[EIGHT]], {{.*}} return_value_and_is_valid
// OFFSETS: nvvm.shfl.sync up %{{.*}}, %{{.*}}, %[[SIXTEEN]], {{.*}} return_value_and_is_valid
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
// In the last round, column >= 4 is equivalent to physical lane >= 16.
// OFFSETS-LABEL: llvm.func {{.*}}@test_scan_shuffle_up_gapped(
// OFFSETS: %[[SIXTEEN:.*]] = llvm.mlir.constant(16 : i32)
// OFFSETS-COUNT-2: nvvm.shfl.sync idx
// OFFSETS: nvvm.shfl.sync up %{{.*}}, %{{.*}}, %[[SIXTEEN]], {{.*}} return_value_and_is_valid
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
#wave64 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
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

// Each of the six wave64 scan rounds permutes both words of the i64 value.
// GFX90A-LABEL: llvm.func {{.*}}@test_scan_shuffle_up_wave64(
// GFX90A: [[VALID:%.*]] = llvm.icmp "sge" [[LANE:%[^,]+]],
// GFX90A: [[SOURCE:%.*]] = llvm.select [[VALID]], {{%.*}}, [[LANE]] : i1, i32
// GFX90A: [[ADDR:%.*]] = llvm.shl [[SOURCE]],
// GFX90A: rocdl.ds_bpermute [[ADDR]],
// GFX90A: rocdl.ds_bpermute
// GFX90A: [[SUM:%.*]] = llvm.add {{.*}} : i64
// GFX90A: llvm.select [[VALID]], [[SUM]],
// GFX90A-COUNT-10: rocdl.ds_bpermute
// GFX90A-NOT: rocdl.ds_bpermute
// GFX90A: llvm.return
tt.func private @test_scan_shuffle_up_wave64(%arg: tensor<64xi64, #wave64>) -> tensor<64xi64, #wave64> {
  %result = "tt.scan"(%arg) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%lhs: i64, %rhs: i64):
    %sum = arith.addi %lhs, %rhs : i64
    tt.scan.return %sum : i64
  }) : (tensor<64xi64, #wave64>) -> tensor<64xi64, #wave64>
  tt.return %result : tensor<64xi64, #wave64>
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
